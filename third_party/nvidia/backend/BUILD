load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//third_party/py/triton:__subpackages__",
    ],
)

pybind_extension(
    name = "cuda_utils",
    srcs = ["driver.c"],
    visibility = [
        "//learning/deepmind/jax/triton/ops:__subpackages__",
        "//third_party/py/triton:__subpackages__",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "launcher",
    srcs = [
        "launcher.cc",
    ],
    tags = [
        "config-cuda-only",
        "requires-gpu-sm80",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
    ],
)

filegroup(
    name = "files",
    srcs = glob(
        include = ["*.py"],
    ),
)
