from setuptools import setup, find_packages

setup(
    name="pibert",
    version="0.1.0",
    author="Somyajit Chakraborty, Chen Xizhong",
    author_email="somyajit.chakraborty@example.com",
    description="Physics-Informed BERT-style Transformer for Multiscale PDE Modeling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pibert",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "full": [
            "pywt>=1.2.0",  # For wavelet transforms
        ]
    },
    entry_points={
        "console_scripts": [
            "pibert-train=pibert.cli:train",
            "pibert-eval=pibert.cli:evaluate",
        ],
    },
)
