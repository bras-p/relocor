from setuptools import setup, find_packages

setup(
    name='relocor',
    packages=[
        "relocor",
    ],
    url='',
    author="Pierre Bras",
    description='RELOCOR: REinforcement Learning based Optimal CORrelation for variance reduction',
    long_description=open('README.md').read(),
    install_requires=[
        "torch>=1.11.0",
        "gymnasium>=0.28.0",
        "stable-baselines3>=2.0.0",
        "numpy>=1.23.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
        ],
    include_package_data=True,
    license='MIT',
)