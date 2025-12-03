# NOTE: Do not depend on any targets from this directory,
# but use //third_party/py/triton instead.

# load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["@triton//:license"],
    default_compatible_with = [# "@build-target"],
    default_visibility = [
        "//third_party/py/triton:__pkg__",
        "@triton//python:__subpackages__",
    ],
)

cc_library(
    name = "passes",
    hdrs = ["src/passes.h"],
    includes = ["src"],
    visibility = ["@triton//third_party:__subpackages__"],
)

cc_library(
    name = "ir",
    hdrs = ["src/ir.h"],
    includes = ["src"],
    visibility = ["@triton//third_party:__subpackages__"],
    deps = [
        "@llvm-project//mlir:IR",
        "@triton//:TritonGPUToLLVM",
    ],
)

pybind_extension(
    name = "libtriton",
    srcs = [
        "src/gluon_ir.cc",
        "src/interpreter.cc",
        "src/ir.cc",
        "src/ir.h",
        "src/linear_layout.cc",
        "src/llvm.cc",
        "src/main.cc",
        "src/passes.cc",
        "src/specialize.cc",
    ],
    copts = ["-DTRITON_BACKENDS_TUPLE=(nvidia)"],
    deps = [
        ":passes",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:InstCombine",
        "@llvm-project//llvm:Instrumentation",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:UBDialect",
        "@triton//:AllPassesAndDialects",
        "@triton//:GluonDialect",
        "@triton//:GluonTransforms",
        "@triton//:TritonDialects",
        "@triton//:TritonGPUToLLVM",
        "@triton//:TritonGPUTransforms",
        "@triton//:TritonHSACO",
        "@triton//:TritonInstrumentTransforms",
        "@triton//:TritonLLVMIR",
        "@triton//:TritonNvidiaGPUTransforms",
        "@triton//:TritonPTX",
        "@triton//:TritonToTritonGPU",
        "@triton//:TritonToTritonGPUPasses",
        "@triton//:TritonTools",
        "@triton//:TritonTransforms",
        "@triton//third_party/amd:TritonAMDGPU",
        "@triton//third_party/nvidia:triton_nvidia",
        "@triton//third_party/proton:ProtonIR",
    ],
)

filegroup(
    name = "files",
    srcs = glob(
        include = ["triton/**/*.py"],
    ),
)
