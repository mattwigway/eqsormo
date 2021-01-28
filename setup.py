from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eqsormo",
    version="0.8.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["statsmodels", "tqdm", "numpy", "pandas", "dill", "scipy",],
    author="Matthew Wigginton Conway",
    author_email="matt@indicatrix.org",
    description="Equilibrium sorting models in Python",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/mattwigway/eqsormo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 3 - Alpha",
    ],
)
