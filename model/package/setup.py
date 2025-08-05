from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    "wandb",
    "tensorflow==2.13.0",
    "pandas",
    "numpy",
    "tqdm",
    "python-json-logger",
]

setup(
    name="capy-trainer",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="Capy Trainer Application",
)
