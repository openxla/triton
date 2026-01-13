load("//third_party/py/pytest:pytest_defs.bzl", "pytest_multi_tests", "pytest_test")

package(
    default_applicable_licenses = ["@triton//:license"],
)

_requires_gpu_sm80 = [
    "config-cuda-only",
    "requires-gpu-sm80",
]

_requires_gpu_sm90 = [
    "config-cuda-only",
    "requires-gpu-sm90",
]

_requires_gpu_sm100 = [
    "config-cuda-only",
    "requires-gpu-sm100-only",
]

_requires_gpu_sm100_full = [
    "config-cuda-only",
    "requires-gpu-sm100-full-only",
]

_requires_config_cuda = select(
    {"@rules_cuda//cuda:is_enabled": []},
    no_match_error = "Requires --config=cuda",
)

_asan_needs_extra_link_memory = select({
    "//tools/cpp:asan_build": {"cpp_link.mem": "20g"},
    "//conditions:default": None,
})

DEVICELESS_TARGETS = [
    "unit/language/test_compile_only.py",
    "unit/language/test_frontend.py",
    "unit/runtime/test_specialize.py",
    "unit/test_filecheck.py",
    "unit/tools/test_linear_layout.py",
    "unit/plugins/test_plugin.py",
    "unit/language/test_mxfp.py",
]

EXCLUDE_TESTS = [
    # failing tests
    "unit/language/test_subprocess.py",  # TODO(b/320224484): fix failing test
    "unit/tools/test_aot.py",  # TODO(b/320224484): fix failing test
    "unit/tools/test_disasm.py",  # TODO(b/320224484): fix failing test
    "unit/runtime/test_blaslt.py",  # TODO(b/346755023): fix failing test
    "unit/test_debug.py",  # TODO(b/374733875): fix failing test. Also see b/374733872.
    "gluon/test_consan.py",  # a new dialect? named gluon. Lets exclude this for now.
    "gluon/test_frontend.py",  # a new dialect? named gluon. Lets exclude this for now.

    # valueless internally
    "backend/test_device_backend.py",  # we run backends differently in g3
    "unit/language/test_reproducer.py",  # this is not an actual test, but a tool for running reproducers
    "unit/test_knobs.py",  # testing binary and file path utilites, which do not work in google's environment
    "unit/test_perf_warning.py",  # No backtraces in non-debug builds.
    "unit/runtime/test_build.py",  # this tests OSS launcher, we implement this differently in g3

    # requires 2 GPUs to run
    "unit/runtime/test_peer_access.py",  # this requires 2 GPUs to run, we have a separate target for it

    # sharded tests
    "unit/language/test_core.py",  # this test is huge and we shard it separately
] + DEVICELESS_TARGETS  # targets that don't require a GPU to run

B200_FULL_GPU_ONLY_TESTS = [
    "unit/language/test_matmul.py",  # TODO(b/462585714):this test fails when run on B200 MIGs
]

test_suite(
    name = "a100_tests",
    tests = [
        ":ampere",
        ":unit/language/test_core_a100",
    ],
)

test_suite(
    name = "h100_tests",
    tests = [
        ":hopper",
        ":unit/language/test_core_h100",
        ":unit/runtime/test_peer_access_h100_x2",
    ],
)

test_suite(
    name = "b200_tests",
    tests = [
        ":blackwell",
        ":blackwell_full",
        ":unit/language/test_core_b200",
        ":unit/runtime/test_peer_access_b200_x2",
    ],
)

# Runs all python tests on A100.
pytest_multi_tests(
    name = "ampere",
    size = "large",
    srcs = [
        "conftest.py",
        "unit/language/test_core.py",
        "unit/language/test_mxfp.py",
    ],
    exec_properties = _asan_needs_extra_link_memory,
    name_suffix = "_a100",
    shard_count = 10,
    tags = _requires_gpu_sm80,
    target_compatible_with = _requires_config_cuda,
    tests = glob(
        include = ["**/test_*.py"],
        exclude = EXCLUDE_TESTS,
    ),
    deps = [
        "@llvm-project//llvm:FileCheck",
        "//third_party/py/expecttest",
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

# Runs all python tests on H100.
pytest_multi_tests(
    name = "hopper",
    size = "large",
    srcs = [
        "conftest.py",
        "unit/language/test_core.py",
        "unit/language/test_mxfp.py",
    ],
    exec_properties = _asan_needs_extra_link_memory,
    name_suffix = "_h100",
    shard_count = 10,
    tags = _requires_gpu_sm90,
    target_compatible_with = _requires_config_cuda,
    tests = glob(
        include = ["**/test_*.py"],
        exclude = EXCLUDE_TESTS,
    ),
    deps = [
        "@llvm-project//llvm:FileCheck",
        "//third_party/py/expecttest",
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

# Shard test_core more on H100, as it is otherwise very slow to run.
pytest_test(
    name = "unit/language/test_core_h100",
    size = "large",
    srcs = [
        "conftest.py",
    ],
    exec_properties = _asan_needs_extra_link_memory,
    shard_count = 40,
    tags = _requires_gpu_sm90,
    target_compatible_with = _requires_config_cuda,
    tests = ["unit/language/test_core.py"],
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

# Shard test_core more on A100, as it is otherwise very slow to run.
pytest_test(
    name = "unit/language/test_core_a100",
    size = "large",
    srcs = [
        "conftest.py",
    ],
    exec_properties = _asan_needs_extra_link_memory,
    shard_count = 40,
    tags = _requires_gpu_sm80,
    target_compatible_with = _requires_config_cuda,
    tests = ["unit/language/test_core.py"],
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

pytest_test(
    name = "deviceless",
    srcs = [
        "conftest.py",
        "unit/plugins/custom_stages.py",
    ],
    exec_properties = _asan_needs_extra_link_memory,
    tests = DEVICELESS_TARGETS,
    deps = [
        "@llvm-project//llvm:FileCheck",
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

# Runs all python tests on B200.
pytest_multi_tests(
    name = "blackwell",
    size = "large",
    srcs = [
        "conftest.py",
        "unit/language/test_core.py",
        "unit/language/test_mxfp.py",
    ],
    exec_properties = _asan_needs_extra_link_memory,
    name_suffix = "_b200",
    shard_count = 10,
    tags = _requires_gpu_sm100,
    target_compatible_with = _requires_config_cuda,
    tests = glob(
        include = ["**/test_*.py"],
        exclude = EXCLUDE_TESTS + B200_FULL_GPU_ONLY_TESTS,
    ),
    deps = [
        "@llvm-project//llvm:FileCheck",
        "//third_party/py/expecttest",
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

pytest_multi_tests(
    name = "blackwell_full",
    size = "large",
    srcs = [
        "conftest.py",
        "unit/language/test_core.py",
        "unit/language/test_mxfp.py",
    ],
    exec_properties = _asan_needs_extra_link_memory,
    name_suffix = "_b200",
    shard_count = 10,
    tags = _requires_gpu_sm100_full,
    target_compatible_with = _requires_config_cuda,
    tests = glob(
        include = B200_FULL_GPU_ONLY_TESTS,
    ),
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

# Shard test_core more on B200, as it is otherwise very slow to run.
pytest_test(
    name = "unit/language/test_core_b200",
    size = "large",
    srcs = [
        "conftest.py",
    ],
    exec_properties = _asan_needs_extra_link_memory,
    shard_count = 40,
    tags = _requires_gpu_sm100,
    target_compatible_with = _requires_config_cuda,
    tests = ["unit/language/test_core.py"],
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

# Requires 2 GPUs to run.
pytest_test(
    name = "unit/runtime/test_peer_access_h100_x2",
    size = "large",
    srcs = ["conftest.py"],
    exec_properties = _asan_needs_extra_link_memory,
    tags = [
        "config-cuda-only",
        "requires-gpu-sm90:2",
    ],
    target_compatible_with = _requires_config_cuda,
    tests = ["unit/runtime/test_peer_access.py"],
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)

# Requires 2 GPUs to run.
pytest_test(
    name = "unit/runtime/test_peer_access_b200_x2",
    size = "large",
    srcs = ["conftest.py"],
    exec_properties = _asan_needs_extra_link_memory,
    tags = [
        "config-cuda-only",
        "requires-gpu-sm100-full:2",
    ],
    target_compatible_with = _requires_config_cuda,
    tests = ["unit/runtime/test_peer_access.py"],
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/triton",
    ],
)
