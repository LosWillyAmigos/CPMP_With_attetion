from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()

setup(
    name='Attentional_CPMP',
    version='0.1.0',
    description='Library to work on the cpmp problem with attention mechanisms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Matias Bugueño B., Thomas Molina S.',
    author_email='matias.bugueno@pucv.cl, thomas.molina@pucv,cl',
    url='https://github.com/LosWillyAmigos/CPMP_With_attention',
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt'),
    classifiers=[
    ],
    python_requires='>=3.6',
)
