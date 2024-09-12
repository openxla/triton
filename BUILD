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

package(
    # copybara:uncomment_begin
    # default_applicable_licenses = [":license"],
    # default_compatible_with = ["//buildenv/target:gce"],
    # default_visibility = [
        # # Add your project here if you need to depend on Triton's C++ sources.
        # # Add a point of contact we can reach out to when needed in the comment.
        # #
        # # If you need to use the Python fronted, add your project to
        # # google3/third_party/py/triton/BUILD instead.
        # #
        # # By adding your project here, you agree to the Triton SLA:  go/triton-google3-sla
        # "//third_party/py/jax:__subpackages__",  # cjfj@
        # "//third_party/tensorflow/compiler/xla:__subpackages__",  # bchetioui@
        # "//platforms/xla/experimental/gpu:__subpackages__",  # csigg@
        # # Triton-internal visibility
        # "//:__subpackages__",
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
        # copybara:comment_begin
        "@bazel_tools" +
        # copybara:comment_end
        "//tools/cpp:compiler": "msvc-cl",
    },
)

# TODO(csigg): fix, enable error upstream, remove.
_no_unused_variable = select({
    ":compiler_is_msvc": [],
    "//conditions:default": ["-Wno-unused-variable"],
})

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
        "@llvm-project//mlir:LLVMOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "triton_attr_inc_gen",
    tbl_outs = [
        (
            ["--gen-attrdef-decls"],
            "include/triton/Dialect/Triton/IR/TritonAttrDefs.h.inc",
        ),
        (
            ["--gen-attrdef-defs"],
            "include/triton/Dialect/Triton/IR/TritonAttrDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/Triton/IR/TritonAttrDefs.td",
    deps = ["td_files"],
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
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/Triton/IR/TritonOps.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_types_inc_gen",
    tbl_outs = [
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
    td_file = "include/triton/Dialect/Triton/IR/TritonTypes.td",
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
        (
            ["--gen-enum-decls"],
            "include/triton/Dialect/TritonGPU/IR/OpsEnums.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "include/triton/Dialect/TritonGPU/IR/OpsEnums.cpp.inc",
        ),
        (
            ["--gen-attr-interface-decls"],
            "include/triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.h.inc",
        ),
        (
            ["--gen-attr-interface-defs"],
            "include/triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td",
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
    name = "triton_gpu_types_inc_gen",
    tbl_outs = [
        (
            ["--gen-typedef-decls"],
            "include/triton/Dialect/TritonGPU/IR/Types.h.inc",
        ),
        (
            ["--gen-typedef-defs"],
            "include/triton/Dialect/TritonGPU/IR/Types.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td",
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
    name = "triton_nvidia_gpu_attr_inc_gen",
    tbl_outs = [
        (
            ["--gen-attrdef-decls"],
            "include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.h.inc",
        ),
        (
            ["--gen-attrdef-defs"],
            "include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.cpp.inc",
        ),
        (
            ["--gen-enum-decls"],
            "include/triton/Dialect/TritonNvidiaGPU/IR/OpsEnums.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "include/triton/Dialect/TritonNvidiaGPU/IR/OpsEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_nvidia_gpu_dialect_inc_gen",
    tbl_outs = [
        (
            ["--gen-dialect-decls"],
            "include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h.inc",
        ),
        (
            ["--gen-dialect-defs"],
            "include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_nvidia_gpu_ops_inc_gen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "include/triton/Dialect/TritonNvidiaGPU/IR/Ops.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "include/triton/Dialect/TritonNvidiaGPU/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_nvidia_gpu_types_inc_gen",
    tbl_outs = [
        (
            ["--gen-typedef-decls"],
            "include/triton/Dialect/TritonNvidiaGPU/IR/Types.h.inc",
        ),
        (
            ["--gen-typedef-defs"],
            "include/triton/Dialect/TritonNvidiaGPU/IR/Types.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUTypes.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_nvidia_gpu_transforms_inc_gen",
    tbl_outs = [
        (
            [
                "--gen-pass-decls",
                "--name=TritonNvidiaGPU",
            ],
            "include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td",
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

gentbl_cc_library(
    name = "triton_target_llvmir_passes_inc_gen",
    tbl_outs = [
        (
            [
                "--gen-pass-decls",
                "--name=TritonLLVMIR",
            ],
            "include/triton/Target/LLVMIR/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Target/LLVMIR/Passes.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "triton_conversion_triton_gpu_to_llvm_pass_inc_gen",
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
    name = "triton_type_interfaces_inc_gen",
    tbl_outs = [
        (
            ["--gen-type-interface-decls"],
            "include/triton/Dialect/Triton/IR/TritonTypeInterfaces.h.inc",
        ),
        (
            ["--gen-type-interface-defs"],
            "include/triton/Dialect/Triton/IR/TritonTypeInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton/Dialect/Triton/IR/TritonTypeInterfaces.td",
    deps = ["td_files"],
)

cc_library(
    name = "TritonAnalysis",
    srcs = [
        "lib/Analysis/Alias.cpp",
        "lib/Analysis/Allocation.cpp",
        "lib/Analysis/Membar.cpp",
        # Part of TritonDialects compilation unit to avoid circular dependencies.
        # "lib/Analysis/Utility.cpp",
        # "lib/Analysis/AxisInfo.cpp",
    ],
    hdrs = [
        "include/triton/Analysis/Alias.h",
        "include/triton/Analysis/Allocation.h",
        "include/triton/Analysis/Membar.h",
        # Part of TritonDialects compilation unit to avoid circular dependencies.
        # "include/triton/Analysis/AxisInfo.h",
        # "include/triton/Analysis/Utility.h",
        "include/triton/Conversion/MLIRTypes.h",
        "include/triton/Conversion/TritonGPUToLLVM/AsmFormat.h",
        "include/triton/Conversion/TritonGPUToLLVM/Utility.h",
        "include/triton/Dialect/TritonGPU/Transforms/Utility.h",
    ],
    copts = _no_unused_variable,
    deps = [
        ":TritonDialects",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TritonDialects",
    srcs = glob([
        "lib/Dialect/Triton/IR/*.cpp",
        "lib/Dialect/TritonGPU/IR/*.cpp",
        "lib/Dialect/TritonNvidiaGPU/IR/*.cpp",
        "lib/Tools/*.cpp",
    ]) + [
        "include/triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h",  # Avoid circular dependency.
        "lib/Analysis/AxisInfo.cpp",  # Avoid circular dependency.
        "lib/Analysis/Utility.cpp",  # Avoid circular dependency.
        "lib/Dialect/TritonGPU/Transforms/Utility.cpp",  # Avoid circular dependency.
    ],
    hdrs = glob([
        "include/triton/Dialect/Triton/IR/*.h",
        "include/triton/Dialect/TritonGPU/IR/*.h",
        "include/triton/Dialect/TritonNvidiaGPU/IR/*.h",
        "include/triton/Tools/*.h",
    ]) + [
        "include/triton/Analysis/AxisInfo.h",  # Avoid circular dependency.
        "include/triton/Analysis/Utility.h",  # Avoid circular dependency.
        "include/triton/Dialect/TritonGPU/Transforms/Utility.h",  # Avoid circular dependency.
    ],
    copts = select({
        ":compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-unused-variable",
            "-Wno-logical-op-parentheses",
        ],
    }),
    includes = ["include"],
    deps = [
        ":triton_dialect_inc_gen",
        ":triton_gpu_attr_inc_gen",
        ":triton_gpu_dialect_inc_gen",
        ":triton_gpu_ops_inc_gen",
        ":triton_gpu_types_inc_gen",
        ":triton_interfaces_inc_gen",
        ":triton_nvidia_gpu_attr_inc_gen",
        ":triton_nvidia_gpu_dialect_inc_gen",
        ":triton_nvidia_gpu_ops_inc_gen",
        ":triton_nvidia_gpu_types_inc_gen",
        ":triton_ops_inc_gen",
        ":triton_types_inc_gen",
        ":triton_type_interfaces_inc_gen",
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
        "@triton//third_party/nvidia:NVGPUDialect",
        # The following is added to make Utility compile
        ":TritonTools",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@triton//third_party/f2reduce",
    ],
)

cc_library(
    name = "TritonTransforms",
    srcs = glob(["lib/Dialect/Triton/Transforms/*.cpp"]),
    hdrs = glob(["include/triton/Dialect/Triton/Transforms/*.h"]),
    copts = _no_unused_variable,
    deps = [
        ":TritonDialects",
        ":triton_combine_inc_gen",
        ":triton_transforms_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = True,  # TritonDialect uses getCanonicalizationPatterns().
)

cc_library(
    name = "TritonGPUTransforms",
    srcs = glob(
        [
            "lib/Dialect/TritonGPU/Transforms/*.cpp",
            "lib/Dialect/TritonGPU/Transforms/*.h",
            "lib/Dialect/TritonGPU/Transforms/Pipeliner/*.cpp",
            "lib/Dialect/TritonGPU/Transforms/Pipeliner/*.h",
        ],
        exclude = ["lib/Dialect/TritonGPU/Transforms/Utility.cpp"],
    ),
    hdrs = glob(
        [
            "include/triton/Dialect/TritonGPU/Transforms/*.h",
        ],
        exclude = ["include/triton/Dialect/TritonGPU/Transforms/Utility.h"],
    ) + [
        "include/triton/Tools/Sys/GetEnv.hpp",
    ],
    copts = select({
        ":compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-reorder-ctor",
            "-Wno-return-type",
            "-Wno-unused-variable",
        ],
    }),
    deps = [
        ":TritonAnalysis",
        ":TritonDialects",
        ":TritonGPUToLLVM",
        ":triton_gpu_transforms_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TritonGPUToLLVM",
    srcs = glob([
        "lib/Conversion/TritonGPUToLLVM/*.h",
        "lib/Conversion/TritonGPUToLLVM/**/*.cpp",
    ]),
    hdrs = glob([
        "include/triton/Tools/Sys/*.hpp",
        "include/triton/Conversion/TritonGPUToLLVM/*.h",
    ]),
    copts = select({
        "//conditions:default": [
            "-Wno-unused-variable",
        ],
    }),
    includes = ["include"],
    deps = [
        ":TritonAnalysis",
        ":TritonDialects",
        ":triton_conversion_triton_gpu_to_llvm_pass_inc_gen",
        ":triton_gpu_attr_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TritonNvidiaGPUTransforms",
    srcs = glob([
        "lib/Dialect/TritonNvidiaGPU/Transforms/*.cpp",
    ]),
    hdrs = glob([
        "include/triton/Dialect/TritonNvidiaGPU/Transforms/*.h",
    ]),
    copts = select({
        ":compiler_is_msvc": [],
        "//conditions:default": [
            "-Wno-ctad-maybe-unsupported",
            "-Wno-logical-op-parentheses",
            "-Wno-non-virtual-dtor",
            "-Wno-return-type",
            "-Wno-unused-variable",
        ],
    }),
    includes = ["include"],
    deps = [
        ":TritonDialects",
        ":TritonGPUTransforms",
        ":triton_nvidia_gpu_transforms_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
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
        ":TritonDialects",
        ":TritonGPUTransforms",
        ":triton_conversion_triton_to_triton_gpu_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TritonLLVMIR",
    srcs = glob([
        "lib/Target/LLVMIR/*.cpp",
        "lib/Target/LLVMIR/*.h",
    ]),
    hdrs = glob(["include/triton/Target/LLVMIR/*.h"]),
    copts = _no_unused_variable,
    deps = [
        ":TritonTransforms",
        ":triton_target_llvmir_passes_inc_gen",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:BinaryFormat",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:InstCombine",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        # copybara:uncomment "//third_party/py/triton/google:find_cuda",
    ],
)

cc_library(
    name = "TritonPTX",
    srcs = glob([
        "lib/Target/PTX/*.cpp",
    ]),
    hdrs = glob(["include/triton/Target/PTX/*.h"]),
    deps = ["@llvm-project//llvm:Support"],
)

cc_library(
    name = "TritonHSACO",
    srcs = glob([
        "lib/Target/HSACO/*.cpp",
    ]),
    hdrs = glob(["include/triton/Target/HSACO/*.h"]),
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
)

cc_library(
    name = "AllPassesAndDialects",
    srcs = [
        "include/triton/Conversion/TritonToTritonGPU/Passes.h",
        "include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h",
    ],
    hdrs = ["bin/RegisterTritonDialects.h"],
    includes = ["."],  # because it includes third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h
    deps = [
        ":TritonDialects",
        ":TritonGPUToLLVM",
        ":TritonGPUTransforms",
        ":TritonLLVMIR",
        ":TritonNvidiaGPUTransforms",
        ":TritonToTritonGPU",
        ":TritonTransforms",
        ":triton_conversion_triton_to_triton_gpu_passes_inc_gen",
        ":triton_nvidia_gpu_transforms_inc_gen",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@triton//test:TritonTestAnalysis",
        "@triton//third_party/amd:TritonAMDGPUToLLVM",
        "@triton//third_party/amd:TritonAMDGPUTransforms",
        "@triton//third_party/nvidia:NVGPUDialect",
        "@triton//third_party/nvidia:NVGPUToLLVM",
        "@triton//third_party/nvidia:TritonNVIDIAGPUToLLVM",
    ],
)

cc_binary(
    name = "triton-opt",
    srcs = [
        "bin/triton-opt.cpp",
    ],
    deps = [
        ":AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
    ],
)

cc_binary(
    name = "triton-llvm-opt",
    srcs = [
        "bin/triton-llvm-opt.cpp",
        "lib/Target/LLVMIR/LLVMPasses.h",
    ],
    deps = [
        ":TritonLLVMIR",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Option",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

# See go/triton-debug for usage.
cc_binary(
    name = "triton-reduce",
    srcs = ["bin/triton-reduce.cpp"],
    deps = [
        ":AllPassesAndDialects",
        "@llvm-project//mlir:MlirReduceLib",
    ],
)

cc_binary(
    name = "triton-tensor-layout",
    srcs = ["bin/triton-tensor-layout.cpp"],
    deps = [
        ":AllPassesAndDialects",
        ":TritonDialects",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
    ],
)

filegroup(
    name = "metadata-file",
    srcs = ["METADATA"],
)
