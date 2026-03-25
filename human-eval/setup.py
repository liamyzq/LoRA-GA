import os

from setuptools import setup, find_packages


def load_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_path, "r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.strip().startswith("#")
        ]


setup(
    name="human-eval",
    py_modules=["human-eval"],
    version="1.0",
    description="",
    author="OpenAI",
    packages=find_packages(),
    install_requires=load_requirements(),
    entry_points={
        "console_scripts": [
            "evaluate_functional_correctness = human_eval.evaluate_functional_correctness:main",
        ]
    }
)
