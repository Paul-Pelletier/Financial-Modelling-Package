from setuptools import setup, find_packages

setup(
    name="financial_modelling",
    version="0.1.0",
    description="A package for financial modelling pipelines.",
    author="Your Name",
    packages=find_packages(),  # Automatically finds 'financial_modelling' and submodules
    install_requires=[
        "pandas",
        "numpy",
        "scipy"
    ],
    python_requires=">=3.8",
)