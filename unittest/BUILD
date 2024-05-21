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
    name = "DialectTest",
    srcs = glob([
        "Dialect/**/*.cpp",
    ]),
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "//:TritonDialects",
    ],
)

cc_test(
    name = "ConversionTest",
    srcs = glob(
        [
            "Conversion/**/*.cpp",
            "Conversion/**/*.h",
        ],
        exclude = [
            "Conversion/TritonGPUToLLVM/EmitIndicesTest.cpp",
            "Conversion/TritonGPUToLLVM/DumpLayout.cpp",
            "Conversion/TritonGPUToLLVM/DumpLayout.h",
        ],
    ),
    copts = select({
        "//:compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
        ],
    }),
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "//:TritonDialects",
        "//:TritonNvidiaGPUTransforms",
        "@triton//third_party/nvidia:TritonNVIDIAGPUToLLVM",
    ],
)

cc_test(
    name = "EmitIndicesTest",
    srcs = [
        "Conversion/TritonGPUToLLVM/DumpLayout.cpp",
        "Conversion/TritonGPUToLLVM/DumpLayout.h",
        "Conversion/TritonGPUToLLVM/EmitIndicesTest.cpp",
    ],
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
    # We want this to be buildable to update LLVM, but it doesn't pass and never has, even in OSS:
    # https://github.com/openai/triton/blob/ded624282e67e5f58db332380e6ff088f276d534/unittest/Conversion/TritonGPUToLLVM/EmitIndicesTest.cpp#L677
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:LLVMDialect",
        "//:TritonDialects",
        "//:TritonGPUToLLVM",
        "//:TritonNvidiaGPUTransforms",
        "@triton//third_party/nvidia:NVGPUDialect",
        "@triton//third_party/nvidia:TritonNVIDIAGPUToLLVM",
    ],
)

build_test(
    name = "build_test",
    allow_empty_target = False,
    targets = [
        ":ConversionTest",
        ":AnalysisTest",
        ":DialectTest",
        ":EmitIndicesTest",
    ],
)
