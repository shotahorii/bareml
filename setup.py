from setuptools import setup
from codecs import open
from os import path

__version__ = '0.0.1'

here = path.abspath(path.dirname(__file__))

# Read README.md to use it as the long description
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    reqs = f.read().split('\n')

reqs = [x.strip() for x in reqs if x.strip() != '']

setup(
    name='machinelfs',
    packages=['machinelfs'], 
    version=__version__,
    license='MIT', 
    install_requires=reqs,
    author='shotahorii',
    author_email='sh.sinker@gmail.com',
    url='https://github.com/shotahorii/ml-from-scratch', 
    description='A Python module containing various machine learning algorithms.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='machine-learning machine-learning-algorithms machine-learning-from-scratch data-science statistical-models',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)