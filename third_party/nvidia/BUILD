load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

package(
    # copybara:uncomment_begin
    # default_applicable_licenses = ["//:license"],
    # default_visibility = [
        # "//third_party/tensorflow/compiler/xla/service/gpu:__subpackages__",
        # "//:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    default_visibility = ["//visibility:public"],
    # copybara:comment_end
)

pybind_library(
    name = "triton_nvidia",
    srcs = [
        "triton_nvidia.cc",
    ],
    # copybara:uncomment_begin
    # visibility = [
        # "//third_party/triton/python:__subpackages__",
    # ],
    # copybara:uncomment_end
    deps = [
        ":NVGPUToLLVM",
        ":TritonNVIDIAGPUToLLVM",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "//:TritonDialects",
        "//:TritonGPUToLLVM",
        "//:TritonNvidiaGPUTransforms",
        "//third_party/triton/python:passes",
    ],
)

cc_library(
    name = "NVGPUToLLVM",
    srcs = glob([
        "lib/NVGPUToLLVM/*.cpp",
    ]),
    hdrs = glob([
        "include/NVGPUToLLVM/*.h",
    ]),
    # copybara:uncomment_begin
    # compatible_with = ["//buildenv/target:gce"],
    # copybara:uncomment_end
    copts = select({
        "//conditions:default": [
            "-Wno-unused-variable",
        ],
    }),
    includes = [
        "..",
        "include",
    ],
    deps = [
        "triton_conversion_nvgpu_to_llvm_passes_inc_gen",
        ":TritonNVIDIAGPUToLLVM",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "//:TritonDialects",
    ],
)

cc_library(
    name = "TritonNVIDIAGPUToLLVM",
    srcs = glob([
        "lib/TritonNVIDIAGPUToLLVM/*.h",
        "lib/TritonNVIDIAGPUToLLVM/**/*.cpp",
    ]),
    hdrs = glob([
        "include/TritonNVIDIAGPUToLLVM/*.h",
    ]) + [
        "lib/TritonNVIDIAGPUToLLVM/Utility.h",
    ],
    # copybara:uncomment_begin
    # compatible_with = ["//buildenv/target:gce"],
    # copybara:uncomment_end
    copts = select({
        "//conditions:default": [
            "-Wno-reorder-ctor",
            "-Wno-unused-variable",
        ],
    }),
    includes = [
        "..",
        "include",
        "lib/TritonNVIDIAGPUToLLVM",
    ],
    deps = [
        ":triton_conversion_triton_nvidia_gpu_to_llvm_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexDialect",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Transforms",
        "//:TritonAnalysis",
        "//:TritonDialects",
        "//:TritonGPUToLLVM",
        "//:triton_gpu_attr_inc_gen",
    ],
)

gentbl_cc_library(
    name = "triton_conversion_nvgpu_to_llvm_passes_inc_gen",
    # copybara:uncomment_begin
    # compatible_with = ["//buildenv/target:gce"],
    # copybara:uncomment_end
    tbl_outs = [
        (
            [
                "--gen-pass-decls",
                "--name=NVGPUToLLVM",
            ],
            "include/NVGPUToLLVM/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/NVGPUToLLVM/Passes.td",
    deps = ["//:td_files"],
)

gentbl_cc_library(
    name = "triton_conversion_triton_nvidia_gpu_to_llvm_passes_inc_gen",
    # copybara:uncomment_begin
    # compatible_with = ["//buildenv/target:gce"],
    # copybara:uncomment_end
    tbl_outs = [
        (
            [
                "--gen-pass-decls",
                "--name=TritonNVIDIAGPUToLLVM",
            ],
            "include/TritonNVIDIAGPUToLLVM/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/TritonNVIDIAGPUToLLVM/Passes.td",
    deps = ["//:td_files"],
)
