from setuptools import find_packages, setup

REQUIRED_PACKAGES = []

setup (
    name="trainer",
    version=".1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Package to train model for deep learning final."
)