# copybara:uncomment_begin
# load("//third_party/llvm/build_defs:lit.bzl", "glob_lit_tests")
# load("//tools/build_defs/build_test:build_test.bzl", "build_test")
# 
# package(
#     default_applicable_licenses = ["//:license"],
#     default_compatible_with = ["//buildenv/target:gce"],
#     default_visibility = ["//:__subpackages__"],
# )
# 
# glob_lit_tests(
#     name = "all_tests",
#     data = [
#         "@llvm-project//llvm:FileCheck",
#         "//:triton-llvm-opt",
#         "//:triton-opt",
#     ],
#     driver = "@llvm-project//mlir:run_lit.sh",
#     exclude = [
#         "Conversion/amd/dedup-by-constancy.mlir",  # AMD-specific, broken
#         "TritonGPU/combine.mlir",  # TODO: b/338346821 - needs cse or something.
#         "TritonGPU/dot-operands.mlir",  # TODO: b/283035396 - broken by cl536931041.patch
#         "TritonGPU/optimize_epilogue.mlir",  # TODO: b/346283526 - AMD-specific, triggering UBSAN
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
#         "//:TritonAnalysis",
#         "//:TritonDialects",
#         "//:TritonGPUToLLVM",
#         "//:TritonGPUTransforms",
#         "//:TritonLLVMIR",
#         "//:TritonPTX",
#         "//:TritonToTritonGPU",
#         "//:TritonTools",
#         "//:TritonTransforms",
#         "//:triton-opt",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "TritonTestAnalysis",
    srcs = glob(["lib/Analysis/*.cpp"]),
    deps = [
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "//:TritonAnalysis",
        "//:TritonDialects",
    ],
)
