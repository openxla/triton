# NOTE: Do not depend on any targets from this directory,
# but use //third_party/py/triton instead.

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
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

pybind_extension(
    name = "libtriton",
    srcs = [
        "src/interpreter.cc",
        "src/ir.cc",
        "src/llvm.cc",
        "src/main.cc",
        "src/passes.cc",
    ],
    copts = ["-DTRITON_BACKENDS_TUPLE=(nvidia)"],
    deps = [
        ":passes",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:InstCombine",
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
        "//:TritonAnalysis",
        "//:TritonDialects",
        "//:TritonGPUToLLVM",
        "//:TritonGPUTransforms",
        "//:TritonHSACO",
        "//:TritonLLVMIR",
        "//:TritonNvidiaGPUTransforms",
        "//:TritonPTX",
        "//:TritonToTritonGPU",
        "//:TritonTools",
        "//:TritonTransforms",
        "@triton//third_party/nvidia:triton_nvidia",
    ],
)

filegroup(
    name = "files",
    srcs = glob(
        include = ["triton/**/*.py"],
    ),
)
