from peft import get_peft_model, LoraConfig, AdaLoraConfig, TaskType
import gc
import os
import sys
import threading
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import (
    train_text_to_text_model,
    model_inference,
    initialize_text_to_text_model,
    transform_dataset,
    merge_llama,
)
import json
import math
from datasets import load_dataset
import wandb
from data import *
from typing import List
import torch
from copy import deepcopy
import logging
from tqdm import tqdm, trange
from typing import Tuple, List, Dict
from peft.tuners.lora.layer import Linear as LoraLinear
from eval_gsm8k import evaluate_gsm8k_model

try:
    import pynvml
except ImportError:
    pynvml = None

log = logging.getLogger(__name__)


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def strip_distributed_launcher_args(argv: List[str]) -> List[str]:
    cleaned = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--local_rank":
            skip_next = True
            continue
        if arg.startswith("--local_rank="):
            continue
        cleaned.append(arg)
    return cleaned


def distributed_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


class GpuUsageMonitor:
    def __init__(self, interval_minutes: float = 5.0):
        self.interval_seconds = max(interval_minutes * 60.0, 1.0)
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = time.time()
        self._enabled = pynvml is not None
        self._visible_indices = self._parse_visible_indices()
        self._nvml_initialized = False

    def _parse_visible_indices(self):
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            return [
                int(idx.strip())
                for idx in visible_devices.split(",")
                if idx.strip() and idx.strip().isdigit()
            ]
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))

    def _ensure_nvml(self):
        if not self._enabled or self._nvml_initialized:
            return
        pynvml.nvmlInit()
        self._nvml_initialized = True

    def _collect_sample(self):
        if not self._enabled:
            return
        try:
            self._ensure_nvml()
            total_memory_mib = 0.0
            total_utilization_pct = 0.0
            per_gpu = []
            for gpu_index in self._visible_indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                used_mib = mem.used / (1024 * 1024)
                total_memory_mib += used_mib
                total_utilization_pct += util.gpu
                per_gpu.append(
                    {
                        "gpu_index": gpu_index,
                        "memory_used_mib": round(used_mib, 2),
                        "utilization_pct": util.gpu,
                    }
                )
            self.samples.append(
                {
                    "timestamp_seconds": round(time.time() - self._start_time, 2),
                    "total_memory_used_mib": round(total_memory_mib, 2),
                    "total_utilization_pct": round(total_utilization_pct, 2),
                    "per_gpu": per_gpu,
                }
            )
        except Exception as exc:
            log.warning(f"GPU usage monitor sample failed: {exc}")

    def _run(self):
        while not self._stop_event.wait(self.interval_seconds):
            self._collect_sample()

    def start(self):
        if not self._enabled or not self._visible_indices:
            return
        self._collect_sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._collect_sample()
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

    def summary(self):
        if not self.samples:
            return {}
        total_mem = [sample["total_memory_used_mib"] for sample in self.samples]
        total_util = [sample["total_utilization_pct"] for sample in self.samples]
        return {
            "gpu_monitor_interval_seconds": self.interval_seconds,
            "gpu_monitor_num_samples": len(self.samples),
            "gpu_total_memory_used_mib_peak": round(max(total_mem), 2),
            "gpu_total_memory_used_mib_avg": round(sum(total_mem) / len(total_mem), 2),
            "gpu_total_utilization_pct_peak": round(max(total_util), 2),
            "gpu_total_utilization_pct_avg": round(sum(total_util) / len(total_util), 2),
            "gpu_usage_samples": self.samples,
        }


def write_run_report(report_path, report):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


def write_run_report_text(report_path, report):
    lines = [
        f"status: {report.get('status')}",
        f"seed: {report.get('seed')}",
        f"dataset_name: {report.get('dataset_name')}",
        f"model_name: {report.get('model_name')}",
        f"runtime_seconds: {report.get('runtime_seconds')}",
        f"gpu_total_memory_used_mib_peak: {report.get('gpu_total_memory_used_mib_peak')}",
        f"gpu_total_memory_used_mib_avg: {report.get('gpu_total_memory_used_mib_avg')}",
        f"gpu_total_utilization_pct_peak: {report.get('gpu_total_utilization_pct_peak')}",
        f"gpu_total_utilization_pct_avg: {report.get('gpu_total_utilization_pct_avg')}",
        f"gsm8k_accuracy: {report.get('gsm8k_accuracy')}",
        f"gsm8k_num_examples: {report.get('gsm8k_num_examples')}",
        f"results_dir: {report.get('results_dir')}",
        f"model_checkpoint_dir: {report.get('model_checkpoint_dir')}",
    ]
    if report.get("error"):
        lines.append(f"error: {report['error']}")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def find_all_linear_modules(model) -> List[str]:
    r"""
    Finds all available modules to apply lora.
    """
    linear_cls = torch.nn.Linear

    output_layer_names = ["lm_head", "embed_tokens"]

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(
            [output_layer in name for output_layer in output_layer_names]
        ):
            module_names.add(name.split(".")[-1])
    return list(module_names)


def find_hidden_state_size(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            return min(module.weight.shape)
    return None


@torch.no_grad()
def reinit_lora_modules(name, module, init_config, **kwargs):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    lora_r = min(module.lora_A.default.weight.shape)
    a_dim = max(module.lora_A.default.weight.shape)
    b_dim = max(module.lora_B.default.weight.shape)
    if init_config.mode == "simple":
        match init_config.lora_A:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=init_config.lora_A_std
                )
            case "kaiming":
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_A.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_A.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_A.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=1.0 / (a_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_A.default.weight)
            case _:
                raise ValueError(f"Unknown lora_A initialization: {init_config.lora_A}")
        match init_config.lora_B:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=init_config.lora_B_std
                )
            case "kaiming":
                torch.nn.init.kaiming_normal_(module.lora_B.default.weight)
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_B.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_B.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_B.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=1.0 / (b_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_B.default.weight)
            case _:
                raise ValueError(f"Unknown lora_B initialization: {init_config.lora_B}")
        if init_config.get("scale", "") == "stable":
            gamma = init_config.stable_gamma
            module.lora_B.default.weight.data *= (m**0.25) / gamma**0.5
            module.lora_A.default.weight.data *= (n**0.25) / gamma**0.5
    elif init_config.mode == "svd":
        U, S, V = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
        V = V.T
        m, n = module.weight.shape
        if init_config.scale == "default":
            S = S / module.scaling["default"]
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])).T.contiguous()
            )
        elif init_config.scale == "stable":
            gamma = init_config.stable_gamma
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * (m**0.25) / gamma**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :] * (n**0.25) / gamma**0.5).contiguous()
            )
        elif init_config.scale == "unit":
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r]).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :]).contiguous()
            )
        elif init_config.scale == "normalized":
            S_sum = S[:lora_r].sum()
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).T.contiguous()
            )
    elif init_config.mode == "gradient":
        named_grad = kwargs["named_grads"]
        grad_name = ".".join(name.split(".")[2:]) + ".weight"
        grads = named_grad[grad_name]
        U, S, V = torch.svd_lowrank(grads.cuda().float(), q=4 * lora_r, niter=4)
        V = V.T
        # set direction
        if init_config.direction == "ArBr":
            B = U[:, 0 : 2 * lora_r : 2]
            A = V[1 : 2 * lora_r : 2, :]
        elif init_config.direction == "A2rBr":
            B = U[:, :lora_r]
            A = V[lora_r : 2 * lora_r, :]
        elif init_config.direction == "ArB2r":
            B = U[:, lora_r : 2 * lora_r]
            A = V[:lora_r, :]
        scaling_factor = module.scaling["default"]
        if init_config.scale == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif init_config.scale == "unit":
            # Because A,B is orthogonal, do not need to scale
            pass
        elif init_config.scale == "stable":
            m, n = grads.shape # m: feature_out, n: feature_in
            # the scale of output is only related to the feature_out
            gamma = init_config.stable_gamma
            B = B * m**0.25 / gamma**0.5
            A = A * m**0.25 / gamma**0.5
        elif init_config.scale == "weightS":
            _, S, _ = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
            S = S / module.scaling["default"]
            avg_s = torch.sqrt(S[:lora_r]).mean().to(A.device)
            B = B * avg_s
            A = A * avg_s
        module.lora_B.default.weight = torch.nn.Parameter(B.contiguous().cuda())
        module.lora_A.default.weight = torch.nn.Parameter(A.contiguous().cuda())

    with torch.no_grad():
        # consider dtype not in init_config
        if "dtype" not in init_config:
            pass
        elif init_config.dtype == "bf16":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.bfloat16
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.bfloat16
            )
        elif init_config.dtype == "fp32":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.float32
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.float32
            )
        # If lora_A@lora_B is not zero, then we need to subtract lora_A@lora_B from the original weight matrix
        offset = (module.lora_B.default.weight @ module.lora_A.default.weight).to(
            module.weight.data.device
        )
        scaling_factor = module.scaling["default"]
        offset *= scaling_factor
        if "norm_clip" in init_config and init_config.norm_clip:
            # for numerical stability, offset's largest value must be less then weight's largest value
            ratio = torch.max(torch.abs(module.weight.data)) / torch.max(
                torch.abs(offset)
            )
            if ratio < 1:
                offset *= ratio
                module.lora_A.default.weight.data *= ratio**0.5
                module.lora_B.default.weight.data *= ratio**0.5
                log.warning(f"Clipping offset by {ratio}")
        try:
            module.weight.data -= offset
        except:
            breakpoint()


def reinit_lora(model, init_config, **kwargs):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    for name, module in tqdm(
        model.named_modules(),
        desc="Reinitializing Lora",
        total=len(list(model.named_modules())),
    ):
        if isinstance(module, LoraLinear):
            reinit_lora_modules(name, module, init_config, **kwargs)

    return model


def get_record_gradient_hook(model, record_dict):
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.cpu()
                else:
                    record_dict[n] += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook


def estimate_gradient(
    model, dataset, batch_size: int = 4
) -> Dict[str, List[torch.Tensor]]:
    r"""
    Estimate the gradient of the model on the given dataset
    """
    log.info("Estimating gradient")
    model.train()
    named_grads = {}
    hooks = []
    for name, param in model.named_parameters():
        hook = param.register_hook(get_record_gradient_hook(model, named_grads))
        hooks.append(hook)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    num = 0
    for batch in tqdm(dataloader, desc="Estimating gradient"):
        num += 1
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward()
        get_record_gradient_hook(model, named_grads)(None)  # get gradient of last layer
        # make sure the gradient is cleared
        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad = None
    for n, g in named_grads.items():
        named_grads[n] /= num
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    return named_grads


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_exp(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    model_name = cfg.model.name
    model_type = cfg.model.type
    dataset_name = cfg.dataset_name
    dataset_func = DATASET_MAP[dataset_name]
    use_peft = cfg.peft.use_peft
    if_use_rslora = cfg.peft.use_rslora
    lora_r = cfg.peft.lora_r
    lora_relative_r = cfg.peft.lora_relative_r
    lora_target_modules = cfg.peft.lora_target_modules
    train_embeddings = cfg.peft.train_embeddings
    deepspeed_enabled = bool(cfg.get("deepspeed", {}).get("enabled", False))
    deepspeed_config = cfg.get("deepspeed", {}).get("config")
    flash_attention = cfg.get("flash_attention", True)
    gradient_checkpointing = cfg.get("gradient_checkpointing", False)
    main_process = is_main_process()
    world_size = get_world_size()
    monitor = None
    start_time = time.time()
    run_report = {
        "status": "running",
        "seed": cfg.seed,
        "dataset_name": dataset_name,
        "model_name": model_name,
    }
    if cfg.dry_run:
        return
    if use_peft:
        assert (lora_r is not None) ^ (
            lora_relative_r is not None
        ), "Please specify lora_r or lora_relative_r"
        assert lora_target_modules is not None, "Please specify lora_target_modules"
    else:
        lora_r = None
        lora_target_modules = None
        lora_relative_r = None
        train_embeddings = True
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "use_peft": use_peft,
        "lora_r": lora_r,
        "lora_target_modules": str(lora_target_modules),
        "lora_relative_r": lora_relative_r,
        "train_embeddings": train_embeddings,
    }
    if cfg.wandb.name:
        name = cfg.wandb.name
    else:
        name = "_".join([f"{k}={v}" for k, v in config.items()])
    cfg.wandb.project += "_" + cfg.dataset_name
    run_dir = os.path.join("results", f"{cfg.wandb.project}/{name}/{cfg.seed}")
    report_path = os.path.join(run_dir, "run_report.json")
    report_text_path = os.path.join(run_dir, "run_report.txt")
    run_report["results_dir"] = run_dir
    wandb_mode = cfg.wandb.get("mode", "online")
    wandb_entity = os.environ.get("WANDB_ENTITY_OVERRIDE", "zeqiye-northwestern-university")
    if not main_process:
        os.environ.setdefault("WANDB_MODE", "disabled")
    wandb_enabled = main_process and wandb_mode != "disabled"
    eval_results = None
    save_dir = os.path.join(run_dir, "merged_checkpoint")
    try:
        if wandb_enabled:
            wandb.init(
                entity=wandb_entity,
                project=cfg.wandb.project,
                name=name,
                config=config,
                mode=wandb_mode,
            )
        if main_process and cfg.get("monitor", {}).get("enabled", False):
            monitor = GpuUsageMonitor(cfg.monitor.get("interval_minutes", 5.0))
            monitor.start()
        train_set, val_set, _ = dataset_func()
        run_report["train_examples"] = len(train_set) if hasattr(train_set, "__len__") else None
        run_report["val_examples"] = len(val_set) if hasattr(val_set, "__len__") else None
        distributed_full_ft = not use_peft and (deepspeed_enabled or world_size > 1)
        model, tokenizer = initialize_text_to_text_model(
            model_name,
            model_type,
            cfg.model.bf16,
            cfg.peft.use_peft,
            flash_attention=flash_attention,
            device_map=None if distributed_full_ft else ("auto" if use_peft else None),
        )
        if gradient_checkpointing and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        additional_kwargs = {}
        if use_peft and cfg.init.mode == "gradient":
            if isinstance(train_set, list):
                temp_set = train_set[: cfg.init.bsz * cfg.init.iters]
            else:
                temp_set = train_set.select(range(cfg.init.bsz * cfg.init.iters))
            transform_dataset(
                model_type=model_type,
                dataset=temp_set,
                tokenizer=tokenizer,
                max_length=cfg.init.max_length,
            )
            named_grads = estimate_gradient(model, temp_set, cfg.init.bsz)
            additional_kwargs["named_grads"] = named_grads

        if lora_target_modules == "all":
            lora_target_modules = find_all_linear_modules(model)
        else:
            lora_target_modules = list(lora_target_modules) if lora_target_modules else []
        if lora_relative_r is not None:
            hidden_size = find_hidden_state_size(model)
            lora_r = int(hidden_size * lora_relative_r)
            log.info(f"lora_r is set to {hidden_size} * {lora_relative_r} = {lora_r}")
        if use_peft and cfg.peft.get("dora", False):
            log.info("Using Dora")
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=cfg.peft.lora_alpha,
                target_modules=lora_target_modules,
                use_rslora=if_use_rslora,
                use_dora=True,
            )
            orig_model_params = sum(p.numel() for p in model.parameters())
            model = get_peft_model(model, peft_config)
            trainable_params, all_param = model.get_nb_trainable_parameters()
            rate = {
                "trainable_params": trainable_params,
                "orig_params": orig_model_params,
                "all_params": all_param,
                "trainable_ratio": trainable_params / all_param,
                "param_ratio": trainable_params / orig_model_params,
            }
        elif use_peft and cfg.peft.get("adalora", False):
            log.info("Using AdaLora")
            peft_config = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_r=lora_r,
                lora_alpha=cfg.peft.lora_alpha,
                target_modules=lora_target_modules,
                total_step=int(len(train_set)/cfg.model.real_batch_size)*cfg.model.epochs,
            )
            orig_model_params = sum(p.numel() for p in model.parameters())
            model = get_peft_model(model, peft_config)
            trainable_params, all_param = model.get_nb_trainable_parameters()
            rate = {
                "trainable_params": trainable_params,
                "orig_params": orig_model_params,
                "all_params": all_param,
                "trainable_ratio": trainable_params / all_param,
                "param_ratio": trainable_params / orig_model_params,
            }
        elif use_peft:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=cfg.peft.lora_alpha,
                target_modules=lora_target_modules,
                use_rslora=if_use_rslora,
            )
            orig_model_params = sum(p.numel() for p in model.parameters())
            model = get_peft_model(model, peft_config)
            reinit_lora(model, cfg.init, **additional_kwargs)
            if train_embeddings:
                model.lm_head.weight.requires_grad = True
            trainable_params, all_param = model.get_nb_trainable_parameters()
            rate = {
                "trainable_params": trainable_params,
                "orig_params": orig_model_params,
                "all_params": all_param,
                "trainable_ratio": trainable_params / all_param,
                "param_ratio": trainable_params / orig_model_params,
            }
            peft_save_dir = os.path.join(run_dir, "orig_checkpoint")
            if main_process:
                model.save_pretrained(peft_save_dir)
                adapter_config = json.load(open(os.path.join(peft_save_dir, "adapter_config.json")))
                adapter_config["lora_alpha"] = -adapter_config["lora_alpha"]
                json.dump(
                    adapter_config, open(os.path.join(peft_save_dir, "adapter_config.json"), "w")
                )
        else:
            all_param = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            rate = {
                "trainable_params": trainable_params,
                "orig_params": all_param,
                "all_params": all_param,
                "trainable_ratio": trainable_params / all_param,
                "param_ratio": 1,
            }
        log.info(rate)
        run_report.update(rate)
        run_report["model_checkpoint_dir"] = save_dir
        if wandb_enabled:
            wandb.summary.update(rate)
        training_loop = train_text_to_text_model
        model = training_loop(
            f"{cfg.wandb.project}/{name}",
            train_set,
            val_set,
            model,
            tokenizer,
            model_type,
            num_train_epochs=cfg.model.epochs,
            per_device_batch_size=cfg.model.per_device_batch_size,
            real_batch_size=cfg.model.real_batch_size,
            bf16=cfg.model.bf16,
            eval_epochs=cfg.model.eval_epochs,
            early_stopping_patience=cfg.model.early_stopping_patience,
            max_length=cfg.model.max_length,
            logging_steps=cfg.model.logging_steps,
            use_loraplus=cfg.peft.use_loraplus,
            loraplus_lr_ratio=cfg.peft.loraplus_lr_ratio,
            learning_rate=cfg.model.learning_rate,
            deepspeed=deepspeed_config if deepspeed_enabled else None,
            ddp_find_unused_parameters=False if world_size > 1 else None,
            gradient_checkpointing=gradient_checkpointing,
            report_to=["wandb"] if wandb_enabled else [],
            save_dir=save_dir if not use_peft else None,
            seed=cfg.seed,
        )
        if main_process:
            log.info(f"Saving model to {save_dir}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        distributed_barrier()
        if (
            main_process
            and cfg.get("evaluation", {}).get("enabled", False)
            and cfg.evaluation.get("task") == "gsm8k"
        ):
            eval_results = evaluate_gsm8k_model(
                save_dir,
                tokenizer_name=save_dir,
                flash_attention=cfg.evaluation.get("flash_attention", False),
                bf16=cfg.model.bf16,
                results_path=os.path.join(run_dir, "gsm8k_eval.json"),
            )
            run_report["gsm8k_accuracy"] = eval_results["accuracy"]
            run_report["gsm8k_num_examples"] = eval_results["num_examples"]
            if wandb_enabled:
                wandb.summary.update(
                    {
                        "gsm8k_accuracy": eval_results["accuracy"],
                        "gsm8k_num_examples": eval_results["num_examples"],
                    }
                )
        run_report["status"] = "success"
    except Exception as exc:
        run_report["status"] = "failed"
        run_report["error_type"] = type(exc).__name__
        run_report["error"] = str(exc)
        raise
    finally:
        if main_process:
            run_report["runtime_seconds"] = round(time.time() - start_time, 2)
            if monitor is not None:
                monitor.stop()
                run_report.update(monitor.summary())
            write_run_report(report_path, run_report)
            write_run_report_text(report_text_path, run_report)
            if wandb_enabled:
                wandb.summary.update(
                    {
                        k: v
                        for k, v in run_report.items()
                        if isinstance(v, (int, float, str))
                        and k not in {"status", "error", "error_type", "results_dir", "model_checkpoint_dir"}
                    }
                )
                wandb.finish(exit_code=0 if run_report["status"] == "success" else 1)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *strip_distributed_launcher_args(sys.argv[1:])]
    run_exp()
