load("//tools/build_defs/build_test:build_test.bzl", "build_test")

package(
    default_applicable_licenses = ["@triton//:license"],
    default_compatible_with = ["//buildenv/target:non_prod"],
    default_visibility = ["@triton//:__subpackages__"],
)

cc_test(
    name = "AnalysisTest",
    srcs = glob(["Analysis/*.cpp"]),
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@triton//:TritonDialects",
    ],
)

cc_test(
    name = "DialectTestCatchAll",
    srcs = glob(
        [
            "Dialect/**/*.cpp",
        ],
        exclude = [
            "Dialect/TritonGPU/DialectTest.cpp",
            "Dialect/TritonGPU/LinearLayoutConversionsTest.cpp",
            "Dialect/TritonGPU/SwizzleTest.cpp",
        ],
    ),
    copts = select({
        "@triton//:compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
        ],
    }),
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@triton//:TritonDialects",
        "@triton//:TritonTools",
    ],
)

cc_test(
    name = "DialectTest",
    srcs = [
        "Dialect/TritonGPU/DialectTest.cpp",
    ],
    copts = select({
        "@triton//:compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
        ],
    }),
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@triton//:TritonDialects",
        "@triton//:TritonTools",
    ],
)

cc_test(
    name = "LinearLayoutConversionsTest",
    srcs = [
        "Dialect/TritonGPU/LinearLayoutConversionsTest.cpp",
    ],
    copts = select({
        "@triton//:compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
        ],
    }),
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@triton//:TritonDialects",
        "@triton//:TritonTools",
    ],
)

cc_test(
    name = "SwizzleTest",
    srcs = [
        "Dialect/TritonGPU/SwizzleTest.cpp",
    ],
    copts = select({
        "@triton//:compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
        ],
    }),
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@triton//:TritonDialects",
        "@triton//:TritonTools",
    ],
)

cc_test(
    name = "ToolsTest",
    srcs = glob(["Tools/*.cpp"]),
    copts = select({
        "@triton//:compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
        ],
    }),
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@triton//:TritonTools",
    ],
)

build_test(
    name = "build_test",
    allow_empty_target = False,
    targets = [
        ":ToolsTest",
        ":SwizzleTest",
        ":LinearLayoutConversionsTest",
        ":DialectTestCatchAll",
        ":AnalysisTest",
        ":DialectTest",
    ],
)
