from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'pandas>=1.2.2',
    'numpy>=1.20.2'
]

REQUIRED_PACKAGES_SETUP = [
    'setuptools',
    'wheel',
]

setup(
    name='lifetime-value',
    description='Lifetime value function over time.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='0.0.1',
    url='https://github.com/merijnvanes/lifetime-value.git',
    author='Merijn van Es',
    author_email='merijnvanes@gmail.com',
    keywords=['lifetime value', 'ltv', 'customer lifetime value', 'clv'],
    packages=find_packages(exclude=['tests*']),
    python_requires='>=3.7',
    install_requires=REQUIRED_PACKAGES,
    setup_requires=REQUIRED_PACKAGES_SETUP,
)
