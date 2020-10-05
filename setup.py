from setuptools import setup, find_packages

setup(
    name='eqsormo',
    version='0.3.8',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    requirements=['statsmodels>=0.11', 'tqdm>=4.0', 'numba>=0.51', 'numpy>=1.0', 'pandas>=1.0', 'dill>=0.3', 'scipy>=1.5'],
    author='Matthew Wigginton Conway',
    author_email='matt@indicatrix.org',
    description='Equilibrium sorting models in Python',
    url='https://github.com/mattwigway/eqsormo',
    classifiers=[
        'License :: OSI Approved :: Apache Software License'
    ]
)
