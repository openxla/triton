load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//third_party/py/triton:__subpackages__",
    ],
)

pybind_extension(
    name = "cuda_utils",
    srcs = ["cuda_utils.cc"],
    visibility = [
        "//learning/deepmind/jax/triton/ops:__subpackages__",
        "//third_party/py/triton:__subpackages__",
    ],
    deps = [
        "//platforms/gpus/cuda/dynamic_libcuda",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
        "@llvm-project//llvm:Support",
    ],
)

filegroup(
    name = "files",
    srcs = glob(
        include = ["**/*.py"],
    ),
)
