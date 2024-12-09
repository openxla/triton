// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm=compute-capability=80 2>&1 | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 3072 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ampere_s8_to_fp16_conversion_opIdx1(%1 : tensor<16x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) attributes {noinline = false} {
    // CHECK-LABEL: ampere_s8_to_fp16_conversion_opIdx1
    // CHECK: llvm.sitofp %{{.*}} : i8 to f16
    %2 = arith.sitofp %1 : tensor<16x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> to tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 3072 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ampere_s8_to_fp16_conversion_opIdx0(%1 : tensor<32x16xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>) attributes {noinline = false} {
    // CHECK-LABEL: @ampere_s8_to_fp16_conversion_opIdx0
    // CHECK: llvm.sitofp %{{.*}} : i8 to f16
    %2 = arith.sitofp %1 : tensor<32x16xi8, #ttg.dot_op<{opIdx = 0 , parent = #mma, kWidth = 4}>> to tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}
