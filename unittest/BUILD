load("//tools/build_defs/build_test:build_test.bzl", "build_test")

package(
    default_applicable_licenses = ["//:license"],
    default_compatible_with = ["//buildenv/target:non_prod"],
    default_visibility = ["//:__subpackages__"],
)

cc_test(
    name = "AnalysisTest",
    srcs = glob(["Analysis/*.cpp"]),
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "//:TritonDialects",
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
        "//:compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
        ],
    }),
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "//:TritonDialects",
    ],
)

cc_test(
    name = "DialectTest",
    srcs = [
        "Dialect/TritonGPU/DialectTest.cpp",
    ],
    copts = select({
        "//:compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
        ],
    }),
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "//:TritonDialects",
    ],
)

cc_test(
    name = "LinearLayoutConversionsTest",
    srcs = [
        "Dialect/TritonGPU/LinearLayoutConversionsTest.cpp",
    ],
    copts = select({
        "//:compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
        ],
    }),
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "//:TritonDialects",
    ],
)

cc_test(
    name = "SwizzleTest",
    srcs = [
        "Dialect/TritonGPU/SwizzleTest.cpp",
    ],
    copts = select({
        "//:compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
        ],
    }),
    deps = [
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
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
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "//:TritonDialects",
        "//:TritonNvidiaGPUTransforms",
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
    ],
)
