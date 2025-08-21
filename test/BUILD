# load("@rules_cc//cc:cc_library.bzl", "cc_library")

# copybara:uncomment_begin
# load("//third_party/llvm/build_defs:lit.bzl", "glob_lit_tests")
# load("//tools/build_defs/build_test:build_test.bzl", "build_test")
# 
# package(
#     default_applicable_licenses = ["@triton//:license"],
#     default_compatible_with = ["//buildenv/target:non_prod"],
#     default_visibility = ["@triton//:__subpackages__"],
# )
# 
# glob_lit_tests(
#     name = "all_tests",
#     data = [
#         "@llvm-project//llvm:FileCheck",
#         "@llvm-project//llvm:llc",
#         "@llvm-project//llvm:opt",
#         "@llvm-project//mlir:mlir-translate",
#         "@triton//:triton-llvm-opt",
#         "@triton//:triton-opt",
#         "@triton//:triton-tensor-layout",
#     ],
#     driver = "@llvm-project//mlir:run_lit.sh",
#     exclude = [
#         "Conversion/amd/dedup-by-constancy.mlir",  # AMD-specific, broken
#         "TritonGPU/amd/amd-instruction-sched.mlir",  # AMD-specific, broken with -debug-only.
#         "TritonGPU/optimize_epilogue.mlir",  # TODO: b/346283526 - AMD-specific, triggering UBSAN
#         "Conversion/allocate_warp_groups.mlir",  # warpGroupStartIds order is wrong, b/418732604
#     ],
#     test_file_exts = [
#         "mlir",
#         "ll",
#     ],
# )
# 
# build_test(
#     name = "build_test",
#     allow_empty_target = False,
#     targets = [
#         "@triton//:TritonDialects",
#         "@triton//:TritonGPUToLLVM",
#         "@triton//:TritonGPUTransforms",
#         "@triton//:TritonLLVMIR",
#         "@triton//:TritonPTX",
#         "@triton//:TritonToTritonGPU",
#         "@triton//:TritonTools",
#         "@triton//:TritonTransforms",
#         "@triton//:triton-opt",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "TritonTestAnalysis",
    srcs = glob(["lib/Analysis/*.cpp"]),
    hdrs = glob(["include/Analysis/*.h"]),
    deps = [
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@triton//:TritonDialects",
        "@triton//third_party/nvidia:TritonNVIDIAGPUToLLVM",
    ],
)

cc_library(
    name = "TritonTestDialect",
    srcs = glob(["lib/Dialect/*.cpp"]),
    deps = [
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@triton//:TritonDialects",
        "@triton//:TritonTransforms",
        "@triton//third_party/nvidia:TritonNVIDIAGPUToLLVM",
    ],
)
