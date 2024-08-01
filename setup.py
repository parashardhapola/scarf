from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import glob


def read(f_name):
    with open(os.path.join(os.path.dirname(__file__), f_name)) as fp:
        return fp.read().rstrip("\n")


def read_lines(f_name):
    return read(f_name).split("\n")


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
    version = read("VERSION").rstrip("\n")
    core_requirements = read_lines("requirements.txt")
    extra_requirements = read_lines("requirements_extra.txt")

    setup(
        name="scarf",
        version=version,
        python_requires=">=3.11",
        description="Scarf: A scalable tool for single-cell omics data analysis",
        long_description=read("pypi_README.rst"),
        long_description_content_type="text/x-rst",
        author="Parashar Dhapola",
        author_email="parashar.dhapola@gmail.com",
        url="https://github.com/parashardhapola/scarf",
        license="BSD 3-Clause",
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: BSD License",
            "Natural Language :: English",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.12",
        ],
        keywords=["single-cell"],
        install_requires=core_requirements,
        extras_require={
            "extra": extra_requirements,
        },
        packages=find_packages(exclude=["tests*"]),
        include_package_data=False,
        cmdclass={"install": PostInstallCommand},
    )
