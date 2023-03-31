# This package imports OpenAI's Triton (https://github.com/openai/triton).
#
# There are two versions of Triton in google3 at the moment. The older version
# can be found at //third_party/py/triton. This is the MLIR-based version close
# to head. We expect to transition users to this version in the following
# weeks.
#
# There is no SLA associated with this package and it may get broken by LLVM
# imports at any time.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
# copybara:uncomment load("//tools/build_defs/license:license.bzl", "license")
load("//:triton.bzl", "if_not_msvc")

package(
    # copybara:uncomment_begin
    # default_applicable_licenses = [":license"],
    # default_compatible_with = ["//buildenv/target:gce"],
    # default_visibility = [
        # "//third_party/tensorflow/compiler/xla:__subpackages__",
        # "//third_party/triton:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    default_visibility = ["//visibility:public"],
    # copybara:comment_end
    # TODO(csigg): fix and remove
    features = [
        "-parse_headers",
        "-use_header_modules",
    ],
)

# copybara:uncomment_begin
# license(name = "license")
# 
# licenses(["notice"])
# 
# exports_files(["LICENSE"])
# copybara:uncomment_end

config_setting(
    name = "compiler_is_msvc",
    flag_values = {
        "@bazel_tools//tools/cpp:compiler": "msvc-cl",
    },
)

td_library(
    name = "td_files",
    srcs = glob(["include/triton/**/*.td"]),
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:CastInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:DestinationStyleOpInterfaceTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "triton_dialect_inc_gen",
    tbl_outs = [
        (
            ["--gen-dialect-decls"],
            "include/triton/Dialect/Triton/IR/Dialect.h.inc",
        ),
        (
            ["--gen-dialect-defs"],
            "include/triton/Dialect/Triton/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/Triton/IR/TritonDialect.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_ops_inc_gen",
    tbl_outs = [
        (
            ["--gen-enum-decls"],
            "include/triton/Dialect/Triton/IR/OpsEnums.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "include/triton/Dialect/Triton/IR/OpsEnums.cpp.inc",
        ),
        (
            ["--gen-op-decls"],
            "include/triton/Dialect/Triton/IR/Ops.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "include/triton/Dialect/Triton/IR/Ops.cpp.inc",
        ),
        (
            ["--gen-typedef-decls"],
            "include/triton/Dialect/Triton/IR/Types.h.inc",
        ),
        (
            ["--gen-typedef-defs"],
            "include/triton/Dialect/Triton/IR/Types.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/Triton/IR/TritonOps.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_interfaces_inc_gen",
    tbl_outs = [
        (
            ["--gen-attr-interface-decls"],
            "include/triton/Dialect/Triton/IR/AttrInterfaces.h.inc",
        ),
        (
            ["--gen-attr-interface-defs"],
            "include/triton/Dialect/Triton/IR/AttrInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/Triton/IR/TritonInterfaces.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_transforms_inc_gen",
    tbl_outs = [
        (
            [
                "--gen-pass-decls",
                "--name=Triton",
            ],
            "include/triton/Dialect/Triton/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/Triton/Transforms/Passes.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_combine_inc_gen",
    # The generated file is #included without relative path.
    strip_include_prefix = "lib/Dialect/Triton/Transforms",
    tbl_outs = [
        (
            ["--gen-rewriters"],
            "lib/Dialect/Triton/Transforms/TritonCombine.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/Triton/Transforms/Combine.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_gpu_dialect_inc_gen",
    tbl_outs = [
        (
            ["--gen-dialect-decls"],
            "include/triton/Dialect/TritonGPU/IR/Dialect.h.inc",
        ),
        (
            ["--gen-dialect-defs"],
            "include/triton/Dialect/TritonGPU/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/TritonGPU/IR/TritonGPUDialect.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_gpu_ops_inc_gen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "include/triton/Dialect/TritonGPU/IR/Ops.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "include/triton/Dialect/TritonGPU/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_gpu_attr_inc_gen",
    tbl_outs = [
        (
            ["--gen-attrdef-decls"],
            "include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.h.inc",
        ),
        (
            ["--gen-attrdef-defs"],
            "include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_gpu_transforms_inc_gen",
    tbl_outs = [
        (
            [
                "--gen-pass-decls",
                "--name=TritonGPU",
            ],
            "include/triton/Dialect/TritonGPU/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/TritonGPU/Transforms/Passes.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_conversion_triton_gpu_to_llvm_passes_inc_gen",
    tbl_outs = [
        (
            [
                "--gen-pass-decls",
                "--name=TritonGPUToLLVM",
            ],
            "include/triton/Conversion/TritonGPUToLLVM/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Conversion/TritonGPUToLLVM/Passes.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_conversion_triton_to_triton_gpu_passes_inc_gen",
    tbl_outs = [
        (
            [
                "--gen-pass-decls",
                "--name=TritonToTritonGPU",
            ],
            "include/triton/Conversion/TritonToTritonGPU/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Conversion/TritonToTritonGPU/Passes.td",
    deps = ["td_files"],
)

cc_library(
    name = "TritonAnalysis",
    srcs = glob(["lib/Analysis/*.cpp"]),
    hdrs = glob(["include/triton/Analysis/*.h"]),
    includes = ["include"],
    deps = [
        ":TritonDialect",
        ":TritonGPUDialect",
        ":triton_gpu_attr_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "TritonDialect",
    srcs = glob(["lib/Dialect/Triton/IR/*.cpp"]),
    hdrs = glob(["include/triton/Dialect/Triton/IR/*.h"]),
    copts = if_not_msvc(["-Wno-unused-variable"]),
    includes = ["include"],
    deps = [
        ":triton_dialect_inc_gen",
        ":triton_interfaces_inc_gen",
        ":triton_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "TritonTransforms",
    srcs = glob(["lib/Dialect/Triton/Transforms/*.cpp"]),
    hdrs = glob(["include/triton/Dialect/Triton/Transforms/*.h"]),
    includes = ["include"],
    deps = [
        ":TritonDialect",
        ":triton_combine_inc_gen",
        ":triton_transforms_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = True,  # TritonDialect uses getCanonicalizationPatterns().
)

cc_library(
    name = "TritonGPUDialect",
    srcs = glob(["lib/Dialect/TritonGPU/IR/*.cpp"]),
    hdrs = glob([
        "include/triton/Analysis/*.h",
        "include/triton/Dialect/TritonGPU/IR/*.h",
    ]),
    copts = if_not_msvc(["-Wno-unused-variable"]),
    includes = ["include"],
    deps = [
        ":TritonDialect",
        ":triton_gpu_attr_inc_gen",
        ":triton_gpu_dialect_inc_gen",
        ":triton_gpu_ops_inc_gen",
        ":triton_gpu_transforms_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TritonGPUTransforms",
    srcs = glob([
        "lib/Dialect/TritonGPU/Transforms/*.cpp",
        "lib/Dialect/TritonGPU/Transforms/*.h",
    ]),
    hdrs = glob(["include/triton/Dialect/TritonGPU/Transforms/*.h"]),
    copts = if_not_msvc(["-Wno-unused-variable"]),
    includes = ["include"],
    deps = [
        ":TritonDialect",
        ":TritonGPUDialect",
        ":triton_gpu_transforms_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TritonGPUToLLVM",
    srcs = glob([
        "lib/Conversion/TritonGPUToLLVM/*.h",
        "lib/Conversion/TritonGPUToLLVM/**/*.cpp",
    ]) + [
        "include/triton/Conversion/MLIRTypes.h",
    ],
    hdrs = glob([
        "include/triton/Tools/Sys/*.hpp",
        "include/triton/Conversion/TritonGPUToLLVM/*.h",
    ]),
    copts = if_not_msvc(["-Wno-unused-variable"]),
    includes = [
        "include",
        "lib/Conversion/TritonGPUToLLVM",
    ],
    deps = [
        ":TritonAnalysis",
        ":TritonDialect",
        ":TritonGPUDialect",
        ":triton_conversion_triton_gpu_to_llvm_passes_inc_gen",
        ":triton_conversion_triton_to_triton_gpu_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUToROCDLTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexDialect",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TritonToTritonGPU",
    srcs = glob([
        "lib/Conversion/TritonToTritonGPU/*.h",
        "lib/Conversion/TritonToTritonGPU/*.cpp",
    ]),
    hdrs = glob(["include/triton/Conversion/TritonToTritonGPU/*.h"]),
    includes = ["include"],
    deps = [
        ":TritonDialect",
        ":TritonGPUDialect",
        ":TritonGPUTransforms",
        ":triton_conversion_triton_gpu_to_llvm_passes_inc_gen",
        ":triton_conversion_triton_to_triton_gpu_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TritonLLVMIR",
    srcs = glob([
        "lib/Target/LLVMIR/*.cpp",
    ]) + [
        "include/triton/Tools/Sys/GetEnv.hpp",
    ],
    hdrs = glob(["include/triton/Target/LLVMIR/*.h"]),
    includes = ["include"],
    deps = [
        ":TritonGPUToLLVM",
        ":TritonTransforms",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        # copybara:uncomment_begin
        # "//third_party/py/triton/google:find_cuda",
        # copybara:uncomment_end
    ],
)

cc_library(
    name = "TritonPTX",
    srcs = glob([
        "lib/Target/PTX/*.cpp",
    ]),
    hdrs = glob(["include/triton/Target/PTX/*.h"]),
    includes = ["include"],
    deps = [
        ":TritonLLVMIR",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
    ],
)

cc_library(
    name = "TritonHSACO",
    srcs = glob([
        "lib/Target/HSACO/*.cpp",
    ]),
    hdrs = glob(["include/triton/Target/HSACO/*.h"]),
    includes = ["include"],
    deps = [
        ":TritonLLVMIR",
        ":TritonTools",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ],
)

cc_library(
    name = "TritonTools",
    hdrs = ["include/triton/Tools/Sys/GetEnv.hpp"],
    includes = ["include"],
)

cc_binary(
    name = "triton-opt",
    srcs = [
        "bin/RegisterTritonDialects.h",
        "bin/triton-opt.cpp",
        "include/triton/Conversion/TritonGPUToLLVM/Passes.h",
        "include/triton/Conversion/TritonToTritonGPU/Passes.h",
    ],
    includes = ["include"],
    deps = [
        ":TritonDialect",
        ":TritonGPUDialect",
        ":TritonGPUToLLVM",
        ":TritonGPUTransforms",
        ":TritonToTritonGPU",
        ":TritonTransforms",
        ":triton_conversion_triton_gpu_to_llvm_passes_inc_gen",
        ":triton_conversion_triton_to_triton_gpu_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        # copybara:uncomment "//third_party/triton/test:TritonTestAnalysis",
    ],
)

cc_binary(
    name = "triton-translate",
    srcs = [
        "bin/triton-translate.cpp",
        "include/triton/Conversion/TritonGPUToLLVM/Passes.h",
        "include/triton/Conversion/TritonToTritonGPU/Passes.h",
    ],
    includes = ["include"],
    deps = [
        ":TritonDialect",
        ":TritonGPUDialect",
        ":TritonGPUToLLVM",
        ":TritonGPUTransforms",
        ":TritonHSACO",
        ":TritonLLVMIR",
        ":TritonPTX",
        ":TritonToTritonGPU",
        ":TritonTransforms",
        ":triton_conversion_triton_gpu_to_llvm_passes_inc_gen",
        ":triton_conversion_triton_to_triton_gpu_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
