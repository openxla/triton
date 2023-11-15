load("//tools/build_defs/build_test:build_test.bzl", "build_test")

package(
    default_applicable_licenses = ["//:license"],
    default_compatible_with = ["//buildenv/target:gce"],
    default_visibility = ["//:__subpackages__"],
)

cc_test(
    name = "AnalysisTest",
    srcs = glob(["Analysis/*.cpp"]),
    deps = [
        "//testing/base/public:gunit_main",
        "//:TritonDialects",
    ],
)

cc_test(
    name = "ConversionTest",
    srcs = glob([
        "Conversion/**/*.cpp",
        "Conversion/**/*.h",
    ]),
    copts = select({
        "//:compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
            "-Wno-private-header",
        ],
    }),
    includes = [
        "Conversion/TritonGPUToLLVM",
    ],
    # TODO(b/293864486): get this to pass.
    tags = ["notap"],
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "//:TritonDialects",
        "//:TritonGPUToLLVM",
        "//:TritonNvidiaGPUTransforms",
    ],
)

cc_test(
    name = "DialectTest",
    srcs = glob([
        "Dialect/**/*.cpp",
    ]),
    deps = [
        "//testing/base/public:gunit_main",
        "//:TritonDialects",
    ],
)

build_test(
    name = "build_test",
    allow_empty_target = False,
    targets = [
        ":ConversionTest",
        ":AnalysisTest",
        ":DialectTest",
    ],
)
