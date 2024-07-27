import pathlib

from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
VERSION = "1.1.0"

setup(
    name='Attentional_CPMP',
    version= VERSION,
    description='Library to work on the cpmp problem with attention mechanisms.',
    long_description= README,
    long_description_content_type='text/markdown',
    author='Matias Bugueño B., Thomas Molina S.',
    author_email='matias.bugueno@pucv.cl, thomas.molina@pucv,cl',
    url='https://github.com/LosWillyAmigos/CPMP_With_attention',
    packages=find_packages(include=("attentional_cpmp", "attentional_cpmp.*")),
    install_requires=read_requirements('requirements.txt'),
    python_requires='>=3.10',
)
