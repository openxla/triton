# load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    # copybara:uncomment_begin
    # default_applicable_licenses = ["@triton//:license"],
    # default_compatible_with = [# "@build-target"],
    # default_visibility = [
        # "@triton//:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    default_visibility = ["//visibility:public"],
    # copybara:comment_end
)

_no_unused_variable = select({
    "@triton//:compiler_is_msvc": [],
    "//conditions:default": ["-Wno-unused-variable"],
})

td_library(
    name = "td_files",
    srcs = glob(["Dialect/include/**/*.td"]),
    includes = [
        "..",  # because 'include proton/Dialect/include/<proper path>'
        "Dialect/include",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@triton//:td_files",
    ],
)

gentbl_cc_library(
    name = "proton_ops_inc_gen",
    tbl_outs = {
        "Dialect/include/Dialect/Proton/IR/Dialect.h.inc": [
            "--gen-dialect-decls",
            "--dialect=proton",
        ],
        "Dialect/include/Dialect/Proton/IR/Dialect.cpp.inc": [
            "--gen-dialect-defs",
            "--dialect=proton",
        ],
        "Dialect/include/Dialect/Proton/IR/Ops.h.inc": ["--gen-op-decls"],
        "Dialect/include/Dialect/Proton/IR/Ops.cpp.inc": ["--gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/include/Dialect/Proton/IR/ProtonOps.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "proton_attrdefs_inc_gen",
    tbl_outs = {
        "Dialect/include/Dialect/Proton/IR/OpsEnums.h.inc": ["--gen-enum-decls"],
        "Dialect/include/Dialect/Proton/IR/OpsEnums.cpp.inc": ["--gen-enum-defs"],
        "Dialect/include/Dialect/Proton/IR/ProtonAttrDefs.h.inc": ["--gen-attrdef-decls"],
        "Dialect/include/Dialect/Proton/IR/ProtonAttrDefs.cpp.inc": ["--gen-attrdef-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/include/Dialect/Proton/IR/ProtonAttrDefs.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "proton_gpu_ops_inc_gen",
    tbl_outs = {
        "Dialect/include/Dialect/ProtonGPU/IR/Dialect.h.inc": [
            "--gen-dialect-decls",
            "--dialect=proton_gpu",
        ],
        "Dialect/include/Dialect/ProtonGPU/IR/Dialect.cpp.inc": [
            "--gen-dialect-defs",
            "--dialect=proton_gpu",
        ],
        "Dialect/include/Dialect/ProtonGPU/IR/Ops.h.inc": ["--gen-op-decls"],
        "Dialect/include/Dialect/ProtonGPU/IR/Ops.cpp.inc": ["--gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUOps.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "proton_gpu_attrdefs_inc_gen",
    tbl_outs = {
        "Dialect/include/Dialect/ProtonGPU/IR/OpsEnums.h.inc": ["--gen-enum-decls"],
        "Dialect/include/Dialect/ProtonGPU/IR/OpsEnums.cpp.inc": ["--gen-enum-defs"],
        "Dialect/include/Dialect/ProtonGPU/IR/AttrDefs.h.inc": ["--gen-attrdef-decls"],
        "Dialect/include/Dialect/ProtonGPU/IR/AttrDefs.cpp.inc": ["--gen-attrdef-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUAttrDefs.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "proton_gpu_types_inc_gen",
    tbl_outs = {
        "Dialect/include/Dialect/ProtonGPU/IR/Types.h.inc": ["--gen-typedef-decls"],
        "Dialect/include/Dialect/ProtonGPU/IR/Types.cpp.inc": ["--gen-typedef-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUTypes.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "proton_to_proton_gpu_inc_gen",
    tbl_outs = {"Dialect/include/Conversion/ProtonToProtonGPU/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=ProtonToProtonGPU",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/include/Conversion/ProtonToProtonGPU/Passes.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "proton_gpu_transforms_inc_gen",
    tbl_outs = {"Dialect/include/Dialect/ProtonGPU/Transforms/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=ProtonGPU",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/include/Dialect/ProtonGPU/Transforms/Passes.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "proton_gpu_to_llvm_inc_gen",
    tbl_outs = {"Dialect/include/Conversion/ProtonGPUToLLVM/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=ProtonGPUToLLVM",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/include/Conversion/ProtonGPUToLLVM/Passes.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "proton_amd_gpu_to_llvm_inc_gen",
    tbl_outs = {"Dialect/include/Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=ProtonAMDGPUToLLVM",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/include/Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.td",
    deps = ["td_files"],
)

gentbl_cc_library(
    name = "proton_nvidia_gpu_to_llvm_inc_gen",
    tbl_outs = {"Dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=ProtonNvidiaGPUToLLVM",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.td",
    deps = ["td_files"],
)

cc_library(
    name = "ProtonIR",
    srcs = glob([
        "Dialect/lib/Dialect/Proton/IR/*.cpp",
    ]),
    hdrs = glob([
        "Dialect/include/Dialect/Proton/IR/*.h",
    ]),
    includes = [
        "..",  # because '#include "proton/Dialect/include/<proper path>"'
        "Dialect/include",
    ],
    deps = [
        ":proton_attrdefs_inc_gen",
        ":proton_ops_inc_gen",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Support",
        "@triton//:TritonDialects",
    ],
)

cc_library(
    name = "ProtonGPUIR",
    srcs = glob([
        "Dialect/lib/Dialect/ProtonGPU/IR/*.cpp",
    ]),
    hdrs = glob([
        "Dialect/include/Dialect/ProtonGPU/IR/*.h",
    ]),
    copts = _no_unused_variable,
    includes = [
        "..",  # because '#include "proton/Dialect/include/<proper path>"'
        "Dialect/include",
    ],
    deps = [
        ":ProtonIR",
        ":proton_gpu_attrdefs_inc_gen",
        ":proton_gpu_ops_inc_gen",
        ":proton_gpu_types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Support",
        "@triton//:TritonDialects",
    ],
)

cc_library(
    name = "ProtonAnalysis",
    srcs = glob(["Dialect/lib/Analysis/*.cpp"]),
    hdrs = glob(["Dialect/include/Analysis/*.h"]),
    includes = ["Dialect/include"],
    deps = [
        ":ProtonIR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@triton//:TritonDialects",
    ],
)

cc_library(
    name = "ProtonToProtonGPU",
    srcs = glob(["Dialect/lib/ProtonToProtonGPU/*.cpp"]),
    hdrs = glob(["Dialect/include/Conversion/ProtonToProtonGPU/*.h"]),
    copts = _no_unused_variable,
    deps = [
        ":ProtonAnalysis",
        ":ProtonGPUIR",
        ":ProtonIR",
        ":proton_to_proton_gpu_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@triton//:TritonDialects",
        "@triton//:TritonGPUToLLVM",
    ],
)

cc_library(
    name = "ProtonGPUTransforms",
    srcs = glob(["Dialect/lib/Dialect/ProtonGPU/Transforms/*.cpp"]),
    hdrs = glob(["Dialect/include/Dialect/ProtonGPU/Transforms/*.h"]),
    includes = ["Dialect/include"],
    deps = [
        ":ProtonGPUIR",
        ":proton_gpu_transforms_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "ProtonGPUToLLVM",
    srcs = glob([
        "Dialect/lib/ProtonGPUToLLVM/*.cpp",
        "Dialect/lib/ProtonGPUToLLVM/**/*.cpp",
    ]),
    hdrs = glob([
        "Dialect/include/Conversion/ProtonGPUToLLVM/*.h",
        "Dialect/include/Conversion/ProtonGPUToLLVM/**/*.h",
    ]),
    copts = _no_unused_variable,
    deps = [
        ":ProtonGPUIR",
        ":proton_amd_gpu_to_llvm_inc_gen",
        ":proton_gpu_to_llvm_inc_gen",
        ":proton_nvidia_gpu_to_llvm_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AMDGPUUtils",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUToROCDLTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@triton//:TritonDialects",
        "@triton//:TritonGPUToLLVM",
        "@triton//third_party/amd:TritonAMDGPU",
        "@triton//third_party/amd:TritonAMDGPUToLLVM",
        "@triton//third_party/nvidia:TritonNVIDIAGPUToLLVM",
    ],
)

pybind_extension(
    name = "triton_proton",
    srcs = [
        "Dialect/triton_proton.cc",
    ],
    visibility = [
        "//third_party/py/triton:__subpackages__",
    ],
    deps = [
        ":ProtonAnalysis",
        ":ProtonGPUIR",
        ":ProtonGPUToLLVM",
        ":ProtonGPUTransforms",
        ":ProtonIR",
        ":ProtonToProtonGPU",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@triton//python:passes",
    ],
)

pybind_extension(
    name = "libproton",
    srcs = glob([
        "csrc/**/*.cpp",
        "csrc/**/*.h",
        "common/**/*.cpp",
        "common/**/*.h",
    ]),
    copts = [
        "-Wno-reorder-ctor",
        "-Wno-implicit-fallthrough",
        "-Wno-pessimizing-move",
    ],
    includes = [
        "common/include",
        "csrc/include",
    ],
    visibility = [
        "//third_party/py/triton:__subpackages__",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "//third_party/json",
        "@triton//third_party/amd:backend_headers",
    ],
)
