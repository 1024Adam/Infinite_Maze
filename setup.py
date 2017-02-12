from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name = 'infinite-maze',
    version = '0.1.0',
    description = 'A maze game where users try to progress as far through the maze as possible',
    long_description = readme,
    author = 'Adam Reid',
    author_email = 'adamjreid10@gmail.com',
    url = 'https://github.com/1024Adam/infinite-maze',
    packages=find_packages(exclude=('tests', 'docs'))
)
