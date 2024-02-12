# NOTE: Do not depend on any targets from this directory,
# but use //third_party/py/triton instead.

load("//third_party/pybind11/google3_utils:build_defs.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//third_party/py/triton:__pkg__",
        "//third_party/triton/python:__subpackages__",
    ],
)

cc_library(
    name = "passes",
    hdrs = ["src/passes.h"],
    includes = ["src"],
    visibility = ["//third_party/triton/third_party:__subpackages__"],
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
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        "//:NVGPUToLLVM",
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
        "//third_party/triton/third_party/nvidia:triton_nvidia",
    ],
)

pybind_extension(
    name = "triton_launcher",
    srcs = [
        "triton/compiler/triton_launcher.c",
    ],
    tags = [
        "config-cuda-only",
        "requires-gpu-sm80",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
    ],
)

filegroup(
    name = "files",
    srcs = glob(
        include = ["triton/**/*.py"],
    ),
)
