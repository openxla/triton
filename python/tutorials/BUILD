load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

package(
    default_applicable_licenses = ["@triton//:license"],
)

_requires_gpu_sm90 = [
    "config-cuda-only",
    "requires-gpu-sm90",
]

_requires_gpu_sm100 = [
    "config-cuda-only",
    "requires-gpu-sm100-full",
    "notap",  # Blackwell support is experimental on Forge and requires special flags.
    "manual",  # Blackwell support is experimental on Forge and requires special flags.
]

_requires_config_cuda = select(
    {"@rules_cuda//cuda:is_enabled": []},
    no_match_error = "Requires --config=cuda",
)

# Package some of these tests that can be used as modules into a library, so we can use the kernel
# implementations elsewhere.
py_library(
    name = "kernels",
    testonly = True,  # Has pytest as a dependency
    srcs = [
        "06-fused-attention.py",
        "09-persistent-matmul.py",
        "10-block-scaled-matmul.py",
    ],
    strict_deps = False,
    visibility = ["//third_party/py/triton:__subpackages__"],
    deps = [
        "//third_party/py/pytest",
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
        "//third_party/py/triton/profiler",
    ],
)

py_test(
    name = "01-vector-add",
    srcs = ["01-vector-add.py"],
    tags = _requires_gpu_sm90,
    target_compatible_with = _requires_config_cuda,
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

py_test(
    name = "02-fused-softmax",
    srcs = ["02-fused-softmax.py"],
    tags = _requires_gpu_sm90,
    target_compatible_with = _requires_config_cuda,
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

py_test(
    name = "03-matrix-multiplication",
    srcs = ["03-matrix-multiplication.py"],
    tags = _requires_gpu_sm90,
    target_compatible_with = _requires_config_cuda,
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

py_test(
    name = "04-low-memory-dropout",
    srcs = ["04-low-memory-dropout.py"],
    tags = _requires_gpu_sm90,
    target_compatible_with = _requires_config_cuda,
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

py_test(
    name = "05-layer-norm",
    srcs = ["05-layer-norm.py"],
    tags = _requires_gpu_sm90,
    target_compatible_with = _requires_config_cuda,
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

py_test(
    name = "06-fused-attention",
    srcs = ["06-fused-attention.py"],
    tags = _requires_gpu_sm90,
    target_compatible_with = _requires_config_cuda,
    deps = [
        "//third_party/py/pytest",
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

py_test(
    name = "07-extern-functions",
    srcs = ["07-extern-functions.py"],
    tags = _requires_gpu_sm90 + [
        # Can't find libdevice.10.bc
        "manual",
        "notap",
    ],
    target_compatible_with = _requires_config_cuda,
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

py_test(
    name = "08-grouped-gemm",
    srcs = ["08-grouped-gemm.py"],
    tags = _requires_gpu_sm90,
    target_compatible_with = _requires_config_cuda,
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

py_test(
    name = "09-persistent-matmul",
    srcs = ["09-persistent-matmul.py"],
    tags = _requires_gpu_sm90,
    target_compatible_with = _requires_config_cuda,
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
        "//third_party/py/triton/profiler",
    ],
)

py_test(
    name = "10-block-scaled-matmul",
    srcs = ["10-block-scaled-matmul.py"],
    tags = _requires_gpu_sm100,
    target_compatible_with = _requires_config_cuda,
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
        "//third_party/py/triton/profiler",
    ],
)

py_test(
    name = "11-programmatic-dependent-launch",
    srcs = ["11-programmatic-dependent-launch.py"],
    tags = _requires_gpu_sm90,
    target_compatible_with = _requires_config_cuda,
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)
