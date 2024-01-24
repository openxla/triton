load("//third_party/py/pytest:pytest_defs.bzl", "pytest_multi_tests")

package(
    default_applicable_licenses = ["//:license"],
)

pytest_multi_tests(
    name = "tests",
    tags = [
        "config-cuda-only",
        "requires-gpu-sm80",
    ],
    tests =
        glob(
            include = ["**/test_*.py"],
            exclude = [
                "test_launch.py",  #TODO(b/320226169): fix failing tests
            ],
        ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)
