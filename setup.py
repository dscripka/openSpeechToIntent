from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Add configuration to build Pybind11 extension
ext_modules = [
    Pybind11Extension(
        "openspeechtointent.forced_alignment",
        ["openspeechtointent/forced_alignment.cpp"],
        extra_compile_args=["-O3"],
    )
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openspeechtointent",
    version="0.1.0",
    author="David Scripka",
    author_email="david.scripka@gmail.com",
    description="A simple, but performant framework for mapping speech directly to categories and intents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dscripka/openspeechtointent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "onnxruntime",
        "pybind11"
    ],
    include_package_data=True,
    package_data={
        "openspeechtointent": ["resources/intents/default_intents.json"]
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
