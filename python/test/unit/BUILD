load("//third_party/py/pytest:pytest_defs.bzl", "pytest_multi_tests")

package(
    default_applicable_licenses = ["//:license"],
)

pytest_multi_tests(
    name = "hopper",
    shard_count = 10,
    tags = [
        "config-cuda-only",
        "requires-gpu-sm80",
    ],
    tests = glob(
        include = ["hopper/**/test_*.py"],
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
    tags = [
        "config-cuda-only",
        "requires-gpu-sm80",
    ],
    tests = glob(
        include = ["language/**/test_*.py"],
        exclude = [
            "language/test_subprocess.py",  # TODO(b/320224484): fix failing test
            "language/test_reproducer.py",  # this is not an actual test, but a tool for running reproducers
        ],
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
    tags = [
        "config-cuda-only",
        "requires-gpu-sm80",
    ],
    tests = glob(
        [
            "operators/**/test_*.py",
        ],
    ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

pytest_multi_tests(
    name = "runtime",
    tags = [
        "config-cuda-only",
        "requires-gpu-sm80",
    ],
    tests =
        glob(
            include = ["runtime/**/test_*.py"],
            exclude = [
                "runtime/test_launch.py",  #TODO(b/320226169): fix failing tests
            ],
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
    tags = [
        "config-cuda-only",
        "requires-gpu-sm80",
    ],
    tests =
        glob(
            include = ["tools/**/test_*.py"],
            exclude = [
                "tools/test_aot.py",  # TODO(b/320224484): fix failing test
            ],
        ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)
