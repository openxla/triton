load("//third_party/py/pytest:pytest_defs.bzl", "pytest_multi_tests")

package(
    default_applicable_licenses = ["//:license"],
)

_requires_gpu_sm80 = [
    "config-cuda-only",
    "requires-gpu-sm80",
]

_requires_config_cuda = select(
    {"@local_config_cuda//cuda:using_clang_allow_exec": []},
    no_match_error = "Requires --config=cuda",
)

EXCLUDE_TESTS = [
    "language/test_reproducer.py",  # this is not an actual test, but a tool for running reproducers
    "language/test_subprocess.py",  # TODO(b/320224484): fix failing test
    "runtime/test_launch.py",  # TODO(b/320226169): fix failing tests
    "tools/test_aot.py",  # TODO(b/320224484): fix failing test
    "hopper/test_persistent_warp_specialized_gemm.py",  # TODO (b/332887968): fix failing test
    "hopper/test_gemm.py",  # TODO (b/332887968): fix failing test
]

# Runs all python tests on H100
pytest_multi_tests(
    name = "hopper",
    size = "large",
    srcs = [
        "conftest.py",
        "language/conftest.py",
        "language/test_core.py",
    ],
    name_suffix = "_h100",
    shard_count = 10,
    tags = [
        "config-cuda-only",
        "requires-gpu-sm90",
    ],
    target_compatible_with = _requires_config_cuda,
    tests = glob(
        include = ["**/test_*.py"],
        exclude = [
            "language/test_core.py",  # TODO (b/332887974) only fails on H100
        ] + EXCLUDE_TESTS,
    ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

pytest_multi_tests(
    name = "language",
    size = "large",
    srcs = [
        "conftest.py",
        "language/conftest.py",
        "language/test_core.py",
    ],
    shard_count = 10,
    tags = _requires_gpu_sm80,
    target_compatible_with = _requires_config_cuda,
    tests = glob(
        include = ["language/**/test_*.py"],
        exclude = EXCLUDE_TESTS,
    ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

pytest_multi_tests(
    name = "operators",
    size = "large",
    srcs = ["conftest.py"],
    shard_count = 10,
    tags = _requires_gpu_sm80,
    target_compatible_with = _requires_config_cuda,
    tests = glob(
        include = ["operators/**/test_*.py"],
        exclude = EXCLUDE_TESTS,
    ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

pytest_multi_tests(
    name = "runtime",
    tags = _requires_gpu_sm80,
    target_compatible_with = _requires_config_cuda,
    tests = glob(
        include = ["runtime/**/test_*.py"],
        exclude = EXCLUDE_TESTS,
    ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

pytest_multi_tests(
    name = "tools",
    size = "large",
    shard_count = 10,
    tags = _requires_gpu_sm80,
    target_compatible_with = _requires_config_cuda,
    tests = glob(
        include = ["tools/**/test_*.py"],
        exclude = EXCLUDE_TESTS,
    ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)
