load("//third_party/bazel_rules/rules_python/python:py_extension.bzl", "py_extension")

package(
    default_applicable_licenses = ["//:license"],
)

py_extension(
    name = "cuda_utils",
    srcs = ["driver.c"],
    visibility = [
        "//learning/deepmind/jax/triton/ops:__subpackages__",
        "//third_party/py/triton:__subpackages__",
    ],
    deps = [
        "//third_party/python_runtime:headers",
        "//:cuda_compat",
    ],
)

filegroup(
    name = "files",
    srcs = glob(
        include = ["*.py"],
    ),
    visibility = [
        "//third_party/py/triton:__subpackages__",
    ],
)