# to import this Thyroid_Disease folder as a local package we need to
# use this setup to install as a package

import setuptools

with open("README.md","r",encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.2"

REPO_NAME = "Thyroid-disease-prediction"
AUTHOR_USER_NAME = "rawatshubham09"
SRC_REPO= "Thyroid_Disease"
AUTHOR_EMAIL = "rawatshubham09@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Thyroid Disease Prediction Using MLFlow",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Traker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where='src')
)