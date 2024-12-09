load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    # copybara:uncomment_begin
    # default_applicable_licenses = ["//:license"],
    # default_compatible_with = ["//buildenv/target:non_prod"],
    # default_visibility = [
        # "//third_party/tensorflow/compiler/xla/service/gpu/fusions/triton:__subpackages__",
        # "//:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    default_visibility = ["//visibility:public"],
    # copybara:comment_end
)

# TODO(csigg): fix, enable error upstream, remove.
_no_unused_variable = select({
    "//:compiler_is_msvc": [],
    "//conditions:default": ["-Wno-unused-variable"],
})

cc_library(
    name = "TritonAMDGPUTransforms",
    srcs = glob([
        "lib/TritonAMDGPUTransforms/**/*.h",
        "lib/TritonAMDGPUTransforms/**/*.cpp",
    ]) + [
        "include/TritonAMDGPUToLLVM/TargetUtils.h",
        "lib/TritonAMDGPUToLLVM/SchedInstructions.h",
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
        ":TritonAMDGPU",
        ":triton_conversion_amdgpu_transforms_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ConvertToLLVM",
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
        "//:TritonAnalysis",
        "//:TritonDialects",
        "//:TritonGPUToLLVM",
        "//:TritonGPUTransforms",
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
    ]),
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
        "@llvm-project//mlir:TensorDialect",
        "//:TritonDialects",
        "//:TritonGPUToLLVM",
    ],
)

cc_library(
    name = "TritonAMDGPUToLLVM",
    srcs = glob([
        "lib/TritonAMDGPUToLLVM/**/*.h",
        "lib/TritonAMDGPUToLLVM/**/*.cpp",
        # TritonAMDGPUToLLVM and TritonAMDGPUDialectToLLVM have interdependencies, easiest way to
        # deal with circular dependencies is to just compile both in a single unit.
        "lib/TritonAMDGPUDialectToLLVM/**/*.h",
        "lib/TritonAMDGPUDialectToLLVM/**/*.cpp",
    ]),
    hdrs = glob([
        "include/TritonAMDGPUToLLVM/**/*.h",
    ]),
    copts = _no_unused_variable,
    includes = [
        "include",
        "lib/TritonAMDGPUToLLVM",
    ],
    deps = [
        ":TritonAMDGPU",
        ":TritonAMDGPUTransforms",
        ":triton_conversion_amdgpu_to_llvm_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//mlir:AMDGPUDialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:ConvertToLLVM",
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
        "//:TritonAnalysis",
        "//:TritonDialects",
        "//:TritonGPUToLLVM",
    ],
)

td_library(
    name = "td_files",
    srcs = glob(["include/**/*.td"]),
    includes = ["include"],
    deps = ["//:td_files"],
)

gentbl_cc_library(
    name = "triton_amdgpu_ops_inc_gen",
    tbl_outs = [
        (
            [
                "--gen-llvmir-conversions",
            ],
            "include/Dialect/TritonAMDGPU/IR/OpsConversions.inc",
        ),
        (
            [
                "--gen-op-decls",
            ],
            "include/Dialect/TritonAMDGPU/IR/Ops.h.inc",
        ),
        (
            [
                "--gen-op-defs",
            ],
            "include/Dialect/TritonAMDGPU/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_amdgpu_dialect_inc_gen",
    tbl_outs = [
        (
            [
                "--gen-dialect-decls",
                "--dialect=amdgpu",
            ],
            "include/Dialect/TritonAMDGPU/IR/Dialect.h.inc",
        ),
        (
            [
                "--gen-dialect-defs",
                "--dialect=amdgpu",
            ],
            "include/Dialect/TritonAMDGPU/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/TritonAMDGPU/IR/TritonAMDGPUDialect.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_amdgpu_attr_def_inc_gen",
    tbl_outs = [
        (
            ["--gen-attrdef-decls"],
            "include/Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.h.inc",
        ),
        (
            ["--gen-attrdef-defs"],
            "include/Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_conversion_amdgpu_to_llvm_passes_inc_gen",
    tbl_outs = [
        (
            [
                "--gen-pass-decls",
                "--name=TritonAMDGPUToLLVM",
            ],
            "include/TritonAMDGPUToLLVM/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/TritonAMDGPUToLLVM/Passes.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_conversion_amdgpu_transforms_passes_inc_gen",
    tbl_outs = [
        (
            [
                "--gen-pass-decls",
                "--name=TritonAMDGPU",
            ],
            "include/TritonAMDGPUTransforms/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/TritonAMDGPUTransforms/Passes.td",
    deps = [":td_files"],
)
