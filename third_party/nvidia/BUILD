load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

package(
    # copybara:uncomment_begin
    # default_applicable_licenses = ["//:license"],
    # default_compatible_with = ["//buildenv/target:gce"],
    # default_visibility = [
        # "//third_party/tensorflow/compiler/xla/service/gpu:__subpackages__",
        # "//:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    default_visibility = ["//visibility:public"],
    # copybara:comment_end
)

pybind_library(
    name = "cublas_headers",
    hdrs = glob([
        "include/*.h",
    ]),
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)

pybind_library(
    name = "triton_nvidia",
    srcs = [
        "triton_nvidia.cc",
    ],
    compatible_with = [],
    # copybara:uncomment_begin
    # visibility = [
        # "@triton//python:__subpackages__",
    # ],
    # copybara:uncomment_end
    deps = [
        ":NVGPUDialect",
        ":NVGPUToLLVM",
        ":TritonNVIDIAGPUToLLVM",
        ":cublas_headers",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "//:TritonDialects",
        "//:TritonGPUToLLVM",
        "//:TritonNvidiaGPUTransforms",
        "@triton//python:passes",
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
        ":NVGPUDialect",
        ":TritonNVIDIAGPUToLLVM",
        ":triton_conversion_nvgpu_to_llvm_passes_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
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
        ":NVGPUDialect",
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
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
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

td_library(
    name = "td_files",
    srcs = glob(["include/Dialect/NVGPU/IR/*.td"]),
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:CastInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:DestinationStyleOpInterfaceTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LLVMOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "nvgpu_ops_inc_gen",
    tbl_outs = [
        (
            ["--gen-llvmir-conversions"],
            "include/Dialect/NVGPU/IR/OpsConversions.inc",
        ),
        (
            ["--gen-op-decls"],
            "include/Dialect/NVGPU/IR/Ops.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "include/Dialect/NVGPU/IR/Ops.cpp.inc",
        ),
        (
            ["--gen-enum-decls"],
            "include/Dialect/NVGPU/IR/OpsEnums.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "include/Dialect/NVGPU/IR/OpsEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/NVGPU/IR/NVGPUOps.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "nvgpu_attr_inc_gen",
    tbl_outs = [
        (
            ["--gen-attrdef-decls"],
            "include/Dialect/NVGPU/IR/NVGPUAttrDefs.h.inc",
        ),
        (
            ["--gen-attrdef-defs"],
            "include/Dialect/NVGPU/IR/NVGPUAttrDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/NVGPU/IR/NVGPUAttrDefs.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "nvgpu_dialect_inc_gen",
    tbl_outs = [
        (
            ["--gen-dialect-decls"],
            "include/Dialect/NVGPU/IR/Dialect.h.inc",
        ),
        (
            ["--gen-dialect-defs"],
            "include/Dialect/NVGPU/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/NVGPU/IR/NVGPUDialect.td",
    deps = ["td_files"],
)

cc_library(
    name = "NVGPUDialect",
    srcs = glob([
        "lib/Dialect/NVGPU/IR/*.cpp",
    ]),
    hdrs = glob([
        "include/Dialect/NVGPU/IR/*.h",
    ]),
    copts = select({
        "//:compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
            "-Wno-logical-op-parentheses",
        ],
    }),
    includes = ["include"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":nvgpu_attr_inc_gen",
        ":nvgpu_dialect_inc_gen",
        ":nvgpu_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        # The following is added to make Utility compile
        "//:TritonTools",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
