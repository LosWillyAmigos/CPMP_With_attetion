from setuptools import setup, find_packages

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
    install_requires=[
        tensorflow==2.15.0
        numpy==1.26.3,
        scikit-learn==1.4.0,
        pymongo==4.6.1,
        matplotlib==3.8.2,
        matplotlib_inline==0.1.6,
        pydot==1.4.2,
        graphviz==0.20.3,
        jupyter==1.0.0,
        optuna==3.6.1,
        json==1.6.3
    ],
    classifiers=[
    ],
    python_requires='>=3.6',
)
