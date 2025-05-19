from setuptools import setup, find_packages

setup(
    name="JoJo",
    version="0.1.0",
    description="A package for modeling the transit light curves of oblate planets and planets with rings",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
    ],
    python_requires='>=3.6',
)