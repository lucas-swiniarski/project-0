# BUILD file for the simple_transformer library

load("@rules_python//python:defs.bzl", "py_library")

# Defines the Python library target
py_library(
    # The name of our library target
    name = "simple_transformer",
    # The source files that make up this library
    srcs = [
        "__init__.py",
        "model.py",
    ],
    # Make this library visible to other packages in the workspace
    visibility = ["//visibility:public"],
    # Declare dependencies required by the source files
    deps = [
        # This references the 'torch' package fetched via pip_parse in the WORKSPACE
        "@pip_deps//torch",
    ],
)