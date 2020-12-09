from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='eqsormo',
    version='0.3.10-pypi-2',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    requirements=['statsmodels>=0.11', 'tqdm>=4.0', 'numba>=0.51', 'numpy>=1.0', 'pandas>=1.0', 'dill>=0.3', 'scipy>=1.5'],
    author='Matthew Wigginton Conway',
    author_email='matt@indicatrix.org',
    description='Equilibrium sorting models in Python',
    long_description_type='text/markdown',
    long_description=long_description,
    url='https://github.com/mattwigway/eqsormo',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 3 - Alpha'
    ]
)
