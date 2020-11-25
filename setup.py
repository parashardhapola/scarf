from setuptools import setup, find_packages
from setuptools.command.install import install
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


class PostInstallCommand(install):
    """
    Post-installation for installation mode.
    From here:
    https://github.com/benfred/py-spy/blob/290584dde76834599d66d74b64165dfe9a357ef5/setup.py#L42
    """
    def run(self):
        install.run(self)
        source_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(source_dir, "bin")
        sgtsne_name = "sgtsne"
        if not os.path.isdir(self.install_scripts):
            os.makedirs(self.install_scripts)
        source = os.path.join(build_dir, sgtsne_name)
        target = os.path.join(self.install_scripts, sgtsne_name)
        if os.path.isfile(target):
            os.remove(target)
        with open(source, 'rb') as src, open(target, 'wb') as dst:
            dst.write(src.read())
        os.chmod(target, 0o555)


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
    install_requires = ['pybind11'] + [x.strip() for x in open('requirements.txt')] + \
                       ['dask[array]', 'dask[dataframe]']
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
        packages=find_packages(exclude=['data']),
        include_package_data=False,
        cmdclass={'install': PostInstallCommand}
    )
