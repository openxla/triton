load("//third_party/pybind11/google3_utils:build_defs.bzl", "pybind_library")

package(
    default_applicable_licenses = ["//:license"],
)

pybind_library(
    name = "triton_nvidia",
    srcs = [
        "triton_nvidia.cc",
    ],
    visibility = [
        "//third_party/triton/python:__subpackages__",
    ],
    deps = [
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "//:NVGPUToLLVM",
        "//:TritonDialects",
        "//:TritonGPUToLLVM",
        "//:TritonNvidiaGPUTransforms",
        "//third_party/triton/python:passes",
    ],
)
