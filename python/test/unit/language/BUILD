load("//third_party/py/pytest:pytest_defs.bzl", "pytest_multi_tests")

package(
    default_applicable_licenses = ["//:license"],
)

pytest_multi_tests(
    name = "tests",
    size = "large",
    srcs = [
        "conftest.py",
        "test_core.py",
    ],
    shard_count = 10,
    tags = [
        "config-cuda-only",
        "requires-gpu-sm80",
    ],
    tests = glob(
        include = ["**/test_*.py"],
        exclude = [
            "test_subprocess.py",  # TODO(b/320224484): fix failing test
            "test_reproducer.py",  # this is not an actual test, but a tool for running reproducers
        ],
    ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)
