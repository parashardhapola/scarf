from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if __name__ == "__main__":

    CLASSIFIERS = [
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        'Operating System :: POSIX :: Linux',
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.7',
    ]
    KEYWORDS = ['store']
    VERSION = open('VERSION').readline().rstrip('\n')
    setup(
        name='scarf',
        description='scarf',
        long_description=read('README.rst'),
        author='Parashar Dhapola',
        author_email='parashar.dhapola@gmail.com',
        url='https://github.com/karlssonlab/scarf',
        license='BSD 3-Clause',
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        install_requires=[x.strip() for x in open('requirements.txt')],
        version=VERSION,
        packages=find_packages(exclude=['data']),
        include_package_data=False
    )
