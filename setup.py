from setuptools import setup, find_packages

setup(
    name="gxcnv",
    version="0.1.0",
    description="Hybrid sWGS CNV analysis algorithm (WisecondorX + BinDel inspired)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Genolyx",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pysam>=0.19.0",
    ],
    entry_points={
        "console_scripts": [
            "gxcnv=gxcnv.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
