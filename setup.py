import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
VERSION = "2.6.1"


setup(
    name="Attentional_CPMP",
    description="Library to resolve Container Pre Marshalling Problem (CPMP) with attentional mechanism",
    long_description=README,
    long_description_content_type="text/markdown",
    version=VERSION,
    url="https://github.com/LosWillyAmigos/CPMP_With_attention.git",
    author="EDANHS and Slinking196",
    install_requires=[
        "numpy==1.26.3",
        "tensorflow==2.15.0",
        "scikit-learn==1.4.0",
        "pymongo==4.6.1",
        "matplotlib==3.8.2",
        "matplotlib_inline==0.1.6",
        "pydot==1.4.2",
        "graphviz==0.20.3",
        "jupyter==1.0.0",
        "optuna==4.1.0",
        "optuna-integration==3.6.0",
        "keras-tuner==1.4.7",
        "cpmp_ml @ git+https://github.com/rilianx/CPMP-ML.git@develop#egg=cpmp_ml"
    ],
    python_requires=">=3.10,<3.12",
    packages=find_packages(
        include=("attentional_cpmp", "attentional_cpmp.*"),
        exclude=["models", "models.*"]
    ),
)
