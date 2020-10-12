from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if __name__ == "__main__":
    classifiers = [
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        'Operating System :: POSIX :: Linux',
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.7',
    ]
    keywords = ['store']
    version = open('VERSION').readline().rstrip('\n')
    install_requires = ['pybind11'] + [x.strip() for x in open('requirements.txt')] + ['dask[array]']
    dependency_links = ['https://github.com/fraenkel-lab/pcst_fast/tarball/master#egg=pcst_fast-1.0.7']
    setup(
        name='scarf',
        description='scarf',
        long_description=read('README.rst'),
        author='Parashar Dhapola',
        author_email='parashar.dhapola@gmail.com',
        url='https://github.com/parashardhapola/scarf',
        license='BSD 3-Clause',
        classifiers=classifiers,
        keywords=keywords,
        install_requires=install_requires,
        dependency_links=dependency_links,
        version=version,
        packages=find_packages(exclude=['data', 'bin']),
        include_package_data=False
    )
