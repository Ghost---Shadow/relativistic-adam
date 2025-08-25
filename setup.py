from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="relativistic-adam",
    version="1.0.0",
    author="Souradeep Nanda",
    description="A PyTorch optimizer that implements a relativistic gradient clipping mechanism, inspired by the theory of special relativity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ghost---Shadow/relativistic-adam",
    project_urls={
        "Bug Tracker": "https://github.com/Ghost---Shadow/relativistic-adam/issues",
        "Documentation": "https://github.com/Ghost---Shadow/relativistic-adam#readme",
        "Repository": "https://github.com/Ghost---Shadow/relativistic-adam",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
    ],
    keywords=[
        "pytorch",
        "optimizer",
        "machine-learning",
        "deep-learning",
        "gradient-clipping",
        "physics-inspired",
        "adam",
        "relativity",
    ],
    include_package_data=True,
    zip_safe=False,
)