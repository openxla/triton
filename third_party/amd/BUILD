# load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    # copybara:uncomment_begin
    # default_applicable_licenses = ["@triton//:license"],
    # default_compatible_with = ["# @build-target"],
    # default_visibility = [
        # "//third_party/tensorflow/compiler/xla/backends/gpu/codegen/triton:__subpackages__",
        # "//third_party/tensorflow/compiler/xla/service/gpu/tests:__subpackages__",
        # "@triton//:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    default_visibility = ["//visibility:public"],
    # copybara:comment_end
)

# TODO(csigg): fix, enable error upstream, remove.
_no_unused_variable = select({
    "@triton//:compiler_is_msvc": [],
    "//conditions:default": ["-Wno-unused-variable"],
})

cc_library(
    name = "TritonAMDGPUTransforms",
    srcs = glob(
        [
            "lib/TritonAMDGPUTransforms/**/*.h",
            "lib/TritonAMDGPUTransforms/**/*.cpp",
        ],
        exclude = [
            "lib/TritonAMDGPUTransforms/MfmaGroup.cpp",  # Avoid circular dependency.
            "lib/TritonAMDGPUTransforms/WmmaGroup.cpp",  # Avoid circular dependency.
        ],
    ) + [
        # Work around dependencies on private headers.
        "lib/TritonAMDGPUToLLVM/TargetInfo.h",
        "lib/TritonAMDGPUToLLVM/Utility.h",
    ],
    hdrs = glob([
        "include/TritonAMDGPUTransforms/**/*.h",
    ]),
    copts = _no_unused_variable,
    includes = [
        "include",
        "lib/TritonAMDGPUTransforms",
    ],
    deps = [
        ":Analysis",
        ":TritonAMDGPU",
        ":TritonAMDGPUToLLVM",
        ":triton_conversion_amdgpu_transforms_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ConvertToLLVM",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:UBDialect",
        "@triton//:TritonDialects",
        "@triton//:TritonGPUToLLVM",
        "@triton//:TritonGPUTransforms",
        "@triton//:TritonTools",
    ],
)

cc_library(
    name = "TritonAMDGPU",
    srcs = glob([
        "lib/Dialect/TritonAMDGPU/**/*.h",
        "lib/Dialect/TritonAMDGPU/**/*.cpp",
    ]),
    hdrs = glob([
        "include/Dialect/TritonAMDGPU/**/*.h",
        "include/Utils/**/*.h",
    ]),
    copts = _no_unused_variable,
    includes = [
        "..",
        "include",
    ],
    deps = [
        ":triton_amdgpu_attr_def_inc_gen",
        ":triton_amdgpu_dialect_inc_gen",
        ":triton_amdgpu_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@triton//:TritonDialects",
        "@triton//:TritonGPUToLLVM",
        "@triton//:TritonTools",
    ],
)

cc_library(
    name = "TritonAMDGPUToLLVM",
    srcs = glob(
        [
            "lib/TritonAMDGPUToLLVM/**/*.h",
            "lib/TritonAMDGPUToLLVM/**/*.cpp",
            # TritonAMDGPUToLLVM and TritonAMDGPUDialectToLLVM have interdependencies, easiest way to
            # deal with circular dependencies is to just compile both in a single unit.
            "lib/TritonAMDGPUDialectToLLVM/**/*.h",
            "lib/TritonAMDGPUDialectToLLVM/**/*.cpp",
        ],
        exclude = [
            "lib/TritonAMDGPUToLLVM/AsyncUtility.h",
            "lib/TritonAMDGPUToLLVM/PatternTritonGPUOpToLLVM.h",
        ],
    ) + [
        "include/TritonAMDGPUTransforms/MfmaGroup.h",  # Avoid circular dependency.
        "include/TritonAMDGPUTransforms/WmmaGroup.h",  # Avoid circular dependency.
        "lib/TritonAMDGPUTransforms/MfmaGroup.cpp",  # Avoid circular dependency.
        "lib/TritonAMDGPUTransforms/WmmaGroup.cpp",  # Avoid circular dependency.
    ],
    hdrs = glob([
        "include/TritonAMDGPUToLLVM/**/*.h",
    ]) + [
        "lib/TritonAMDGPUToLLVM/AsyncUtility.h",
        "lib/TritonAMDGPUToLLVM/PatternTritonGPUOpToLLVM.h",
    ],
    copts = _no_unused_variable + ["-Wno-implicit-fallthrough"],
    includes = [
        "include",
        "lib/TritonAMDGPUToLLVM",
    ],
    deps = [
        ":Analysis",
        ":TritonAMDGPU",
        ":triton_conversion_amdgpu_to_llvm_passes_inc_gen",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:AMDGPUDialect",
        "@llvm-project//mlir:AMDGPUUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:ConvertToLLVM",
        "@llvm-project//mlir:DialectUtils",
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
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:UBToLLVM",
        "@triton//:TritonDialects",
        "@triton//:TritonGPUToLLVM",
        "@triton//:TritonTools",
    ],
)

td_library(
    name = "td_files",
    srcs = glob(["include/**/*.td"]),
    includes = ["include"],
    deps = ["@triton//:td_files"],
)

gentbl_cc_library(
    name = "triton_amdgpu_ops_inc_gen",
    tbl_outs = {
        "include/Dialect/TritonAMDGPU/IR/OpsConversions.inc": [
            "--gen-llvmir-conversions",
        ],
        "include/Dialect/TritonAMDGPU/IR/Ops.h.inc": [
            "--gen-op-decls",
        ],
        "include/Dialect/TritonAMDGPU/IR/Ops.cpp.inc": [
            "--gen-op-defs",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_amdgpu_dialect_inc_gen",
    tbl_outs = {
        "include/Dialect/TritonAMDGPU/IR/Dialect.h.inc": [
            "--gen-dialect-decls",
            "--dialect=amdgpu",
        ],
        "include/Dialect/TritonAMDGPU/IR/Dialect.cpp.inc": [
            "--gen-dialect-defs",
            "--dialect=amdgpu",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/TritonAMDGPU/IR/TritonAMDGPUDialect.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_amdgpu_attr_def_inc_gen",
    tbl_outs = {
        "include/Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.h.inc": ["--gen-attrdef-decls"],
        "include/Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.cpp.inc": ["--gen-attrdef-defs"],
        "include/Dialect/TritonAMDGPU/IR/TritonAMDGPUEnums.h.inc": ["--gen-enum-decls"],
        "include/Dialect/TritonAMDGPU/IR/TritonAMDGPUEnums.cpp.inc": ["--gen-enum-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_conversion_amdgpu_to_llvm_passes_inc_gen",
    tbl_outs = {"include/TritonAMDGPUToLLVM/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=TritonAMDGPUToLLVM",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/TritonAMDGPUToLLVM/Passes.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_conversion_amdgpu_transforms_passes_inc_gen",
    tbl_outs = {"include/TritonAMDGPUTransforms/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=TritonAMDGPU",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/TritonAMDGPUTransforms/Passes.td",
    deps = [":td_files"],
)

cc_library(
    name = "Analysis",
    srcs = glob(["lib/Analysis/*.cpp"]),
    hdrs = glob(["include/Analysis/*.h"]),
    copts = _no_unused_variable + select({
        "@triton//:compiler_is_msvc": [],
        "//conditions:default": ["-Wno-implicit-fallthrough"],
    }),
    includes = [
        "../../",
        "include",
    ],
    deps = [
        ":TritonAMDGPU",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferIntRangeCommon",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@triton//:TritonDialects",
    ],
)

cc_library(
    name = "TestAMDAnalysis",
    srcs = glob(["test/lib/Analysis/*.cpp"]),
    deps = [
        ":Analysis",
        ":TritonAMDGPUToLLVM",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
        "@triton//:TritonDialects",
        "@triton//test:TritonTestAnalysis",
    ],
)

cc_library(
    name = "backend_headers",
    defines = [
        "__HIP_PLATFORM_AMD__",
    ],
    includes = [
        "backend/include",
    ],
    textual_hdrs = glob(["backend/include/**/*.h"]),
)
