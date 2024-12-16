from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="financial_modelling",
    version="0.1.0",
    description="A package for financial modelling pipelines.",
    author="Paul Pelletier",
    packages=find_packages(),
    install_requires=required,  # Read dependencies from requirements.txt
    python_requires=">=3.8",
)
