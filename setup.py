from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import glob


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
        if not os.path.isdir(self.install_scripts):
            os.makedirs(self.install_scripts)
        source_dir = os.path.dirname(os.path.abspath(__file__))
        source_files = glob.glob(os.path.join(source_dir, "bin") + "/*")
        for source in source_files:
            target = os.path.join(self.install_scripts, os.path.basename(source))
            if os.path.isfile(target):
                os.remove(target)
            with open(source, "rb") as src, open(target, "wb") as dst:
                dst.write(src.read())
            os.chmod(target, 0o655)


if __name__ == "__main__":
    classifiers = [
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ]
    keywords = ["store"]
    version = open("VERSION").readline().rstrip("\n")
    install_requires = (
        ["pybind11"]
        + [x.strip() for x in open("requirements.txt")]
        + ["dask[array]", "dask[dataframe]"]
    )
    dependency_links = []
    setup(
        name="scarf",
        description="Scarf",
        long_description=read("README.rst"),
        author="Parashar Dhapola",
        author_email="parashar.dhapola@gmail.com",
        url="https://github.com/parashardhapola/scarf",
        license="BSD 3-Clause",
        classifiers=classifiers,
        keywords=keywords,
        install_requires=install_requires,
        dependency_links=dependency_links,
        version=version,
        packages=find_packages(),
        include_package_data=False,
        cmdclass={"install": PostInstallCommand},
    )
