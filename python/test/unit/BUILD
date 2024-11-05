load("//third_party/py/pytest:pytest_defs.bzl", "pytest_multi_tests", "pytest_test")

package(
    default_applicable_licenses = ["//:license"],
)

_requires_gpu_sm80 = [
    "config-cuda-only",
    "requires-gpu-sm80",
]

_requires_config_cuda = select(
    {"@local_config_cuda//cuda:using_blaze_config_cuda": []},
    no_match_error = "Requires --config=cuda",
)

EXCLUDE_TESTS = [
    "language/test_reproducer.py",  # this is not an actual test, but a tool for running reproducers
    "language/test_subprocess.py",  # TODO(b/320224484): fix failing test
    "runtime/test_launch.py",  # TODO(b/320226169): fix failing tests
    "tools/test_aot.py",  # TODO(b/320224484): fix failing test
    "tools/test_disasm.py",  # TODO(b/320224484): fix failing test
    "hopper/test_persistent_warp_specialized_gemm.py",  # TODO (b/342348738): fix failing test
    "runtime/test_cublas.py",  # TODO(b/346755023): fix failing test
    "test_debug.py",  # TODO(b/374733875): fix failing test. Also see b/374733872.
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
        exclude = EXCLUDE_TESTS + [
            "language/test_core.py",
            "language/test_pipeliner.py",  # TODO(b/362458006): fix failing test
            "hopper/test_experimental_tma.py",  # TODO(b/362458006): fix failing test
        ],
    ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

# Shard test_core more, as it is otherwise very slow to run.
pytest_test(
    name = "hopper/language/test_core_h100",
    size = "large",
    srcs = [
        "conftest.py",
        "language/conftest.py",
    ],
    shard_count = 40,
    tags = [
        "config-cuda-only",
        "requires-gpu-sm90",
    ],
    target_compatible_with = _requires_config_cuda,
    tests = ["language/test_core.py"],
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
        exclude = EXCLUDE_TESTS + ["language/test_core.py"],
    ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

# Shard test_core more, as it is otherwise very slow to run.
pytest_test(
    name = "language/test_core",
    size = "large",
    srcs = [
        "conftest.py",
        "language/conftest.py",
    ],
    shard_count = 40,
    tags = _requires_gpu_sm80,
    target_compatible_with = _requires_config_cuda,
    tests = ["language/test_core.py"],
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

pytest_multi_tests(
    name = "instrumentation",
    size = "large",
    srcs = ["conftest.py"],
    shard_count = 10,
    tags = _requires_gpu_sm80,
    target_compatible_with = _requires_config_cuda,
    tests = glob(
        include = ["instrumentation/**/test_*.py"],
        exclude = EXCLUDE_TESTS,
    ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

pytest_multi_tests(
    name = "runtime",
    srcs = ["conftest.py"],
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

pytest_multi_tests(
    name = "unit",
    size = "large",
    srcs = ["conftest.py"],
    shard_count = 10,
    tags = _requires_gpu_sm80,
    target_compatible_with = _requires_config_cuda,
    tests = glob(
        include = ["test_*.py"],
        exclude = EXCLUDE_TESTS,
    ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)
