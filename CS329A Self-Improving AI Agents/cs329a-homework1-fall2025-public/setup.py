from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="cs329_hw1",
    version="0.1.0",
    author="stanford-cs329a",
    author_email="",
    description="A toolbox for evaluating self-improvement techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stanford-cs329a/eval-toolbox",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10,<3.13",  # latex2sympy2 pins antlr4 4.7.2, which needs typing.io (removed in 3.13)
    install_requires=requirements,
)
