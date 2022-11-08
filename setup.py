#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from distutils.util import convert_path
from shutil import rmtree

from setuptools import Command, find_packages, setup

main_ns = {}
ver_path = convert_path("manifest/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

# Package meta-data.
NAME = "manifest-ml"
DESCRIPTION = "Manifest for Prompt Programming Foundation Models."
URL = "https://github.com/HazyResearch/manifest"
EMAIL = "lorr1@cs.stanford.edu"
AUTHOR = "Laurel Orr and Avanika Narayan"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = main_ns["__version__"]

# What packages are required for this module to be executed?
REQUIRED = [
    "dill>=0.3.5",
    "redis>=4.3.1",
    "requests>=2.27.1",
    "sqlitedict>=2.0.0",
    "tqdm>=4.64.0",
]

# What packages are optional?
EXTRAS = {
    "api": [
        "Flask>=2.1.2",
        "accelerate>=0.10.0",
        "transformers>=4.20.0",
        "torch>=1.8.0",
    ],
    "dev": [
        "autopep8>=1.6.0",
        "black>=22.3.0",
        "isort>=5.9.3",
        "flake8>=4.0.0",
        "flake8-docstrings>=1.6.0",
        "mypy>=0.950",
        "pep8-naming>=0.12.1",
        "docformatter>=1.4",
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "python-dotenv>=0.20.0",
        "sphinx-rtd-theme>=0.5.1",
        "nbsphinx>=0.8.0",
        "recommonmark>=0.7.1",
        "pre-commit>=2.14.0",
        "types-redis>=4.2.6",
        "types-requests>=2.27.29",
        "types-PyYAML>=6.0.7",
        "types-protobuf>=3.19.21",
        "types-python-dateutil>=2.8.16",
        "types-setuptools>=57.4.17",
        "sphinx-autobuild",
        "twine",
    ],
}
EXTRAS["all"] = list(set(sum(EXTRAS.values(), [])))

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="Apache 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
)
