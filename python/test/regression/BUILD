load("//third_party/py/pytest:pytest_defs.bzl", "pytest_multi_tests")

package(
    default_applicable_licenses = ["//:license"],
)

pytest_multi_tests(
    name = "tests",
    size = "large",
    shard_count = 10,
    tags = [
        "config-cuda-only",
        "requires-gpu-sm80",
    ],
    tests = glob(
        include = ["test_*.py"],

        #TODO(b/321005767): fix failing test
        exclude = [
            "test_performance.py",
        ],
    ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)
