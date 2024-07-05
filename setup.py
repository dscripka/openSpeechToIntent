from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Add configuration to build Pybind11 extension
ext_modules = [
    Pybind11Extension(
        "openspeechtointent.forced_alignment",
        ["openspeechtointent/forced_alignment.cpp"],
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
