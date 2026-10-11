import os
from typing import ClassVar

from setuptools import Command, setup


class BuildExecutables(Command):
    user_options: ClassVar[list[tuple[str, str | None, str]]] = []

    def initialize_options(self):
        self.build_dir = None

    def finalize_options(self):
        self.set_undefined_options("build", ("build_scripts", "build_dir"))

    def get_source_files(self):
        return self.distribution.scripts

    def run(self):
        self.mkpath(self.build_dir)
        for source in self.get_source_files():
            target = os.path.join(self.build_dir, os.path.basename(source))
            self.copy_file(source, target)
            if not self.dry_run:
                os.chmod(target, 0o755)


if __name__ == "__main__":
    setup(cmdclass={"build_scripts": BuildExecutables})
