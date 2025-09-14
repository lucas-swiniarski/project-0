load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# 1. Fetch the Python rules for Bazel
# Check for the latest version and SHA256 at https://github.com/bazelbuild/rules_python/releases
http_archive(
    name = "rules_python",
    sha256 = "954a992892f26a4435f0e3943b72bce7041b3b9a13e274f282a5bd5adcf92a54",
    strip_prefix = "rules_python-0.31.0",
    url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.31.0.tar.gz",
)

# 2. Load the Python toolchain
load("@rules_python//python:repositories.bzl", "python_register_toolchains")

# Register a specific Python version. This makes builds consistent.
# You can have multiple toolchains for different versions.
python_register_toolchains(
    name = "python3_11",
    python_version = "3.11",
)

# 3. Load pip dependencies
load("@rules_python//python:pip.bzl", "pip_parse")

# This rule parses your requirements.txt file and creates Bazel targets for each package.
# We will create the requirements.txt file next.
pip_parse(
   name = "pip_deps",
   requirements_lock = "//:requirements.txt",
)

# 4. Load the dependencies into the WORKSPACE
# This makes the pip packages available as @pip_deps//<package_name>
load("@pip_deps//:requirements.bzl", "install_deps")

install_deps()

