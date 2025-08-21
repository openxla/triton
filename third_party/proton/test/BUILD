load("//third_party/py/pytest:pytest_defs.bzl", "pytest_multi_tests")

package(
    default_applicable_licenses = ["@triton//:license"],
)

pytest_multi_tests(
    name = "all_tests",
    tags = ["requires-gpu-sm90"],
    target_compatible_with = select(
        {"@local_config_cuda//cuda:using_blaze_config_cuda": []},
        no_match_error = "Requires --config=cuda",
    ),
    tests = glob(
        include = ["test_*.py"],
        exclude = [
            "test_viewer.py",  # Requires hatchet as dependency, which we do not have imported.
            "test_cmd.py",  # Tries to do filepath magic, would need patching to work in google3.
            "test_profile.py",  # Needs to dlopen cupti, but we only have a static version of it.
            "test_api.py",  # Needs to dlopen cupti, but we only have a static version of it.
            "test_lib.py",  # Needs to dlopen cupti, but we only have a static version of it.
        ],
    ),
    deps = [
        "//third_party/py/pandas",
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
        "//third_party/py/triton/profiler",
    ],
)
