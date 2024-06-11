// RUN: triton-opt %s -split-input-file -canonicalize | FileCheck %s


// CHECK-LABEL: @test_canonicalize_convert_view
// CHECK-SAME: (%[[ARG:.+]]: tensor<64x64xf32
//   CHECK-NOT:   triton_gpu.convert_layout
//       CHECK:   %[[V:.+]] = tt.reshape %[[ARG]] {allow_reorder = true}
//       CHECK:   tt.return %[[V]]
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>

module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.target" = "cuda:80"} {
tt.func @test_canonicalize_convert_view(%arg0: tensor<64x64xf32, #blocked0>) -> tensor<4096xf32, #blocked1> {
    %c = triton_gpu.convert_layout %arg0 : tensor<64x64xf32, #blocked0> -> tensor<64x64xf32, #blocked2>
    %r = tt.reshape %c {allow_reorder = true} : tensor<64x64xf32, #blocked2> -> tensor<4096xf32, #blocked1>
    tt.return %r : tensor<4096xf32, #blocked1>
}
}  // end module

// -----

// test that the convert doesn't get combined with view if the resulting operations
// is an expensive view which would require moving data across threads.
// CHECK-LABEL: @test_canonicalize_convert_expensive_view
// CHECK-SAME: (%[[ARG:.+]]: tensor<256x16xf32
//       CHECK:   %[[C:.+]] = triton_gpu.convert_layout %[[ARG]]
//       CHECK:   %[[V:.+]] = tt.reshape %[[C]] {allow_reorder = true}
//       CHECK:   tt.return %[[V]]
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.target" = "cuda:80"} {
tt.func @test_canonicalize_convert_expensive_view(%arg0: tensor<256x16xf32, #blocked0>) -> tensor<4096xf32, #blocked1> {
    %c = triton_gpu.convert_layout %arg0 : tensor<256x16xf32, #blocked0> -> tensor<256x16xf32, #blocked2>
    %r = tt.reshape %c {allow_reorder = true} : tensor<256x16xf32, #blocked2> -> tensor<4096xf32, #blocked1>
    tt.return %r : tensor<4096xf32, #blocked1>
}
}  // end module

// -----

// CHECK-LABEL: @test_canonicalize_convert_histogram
// CHECK-SAME: (%[[ARG:.+]]: tensor<256xi32
//   CHECK-NOT:   triton_gpu.convert_layout
//       CHECK:   %[[V:.+]] = tt.histogram %[[ARG]]
//   CHECK-NOT:   triton_gpu.convert_layout
//       CHECK:   tt.return %[[V]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.target" = "cuda:80"} {
tt.func @test_canonicalize_convert_histogram(%arg0: tensor<256xi32, #blocked1>) -> tensor<512xi32, #blocked2> {
    %0 = triton_gpu.convert_layout %arg0 : tensor<256xi32, #blocked1> -> tensor<256xi32, #blocked>
    %1 = tt.histogram %0 : tensor<256xi32, #blocked> -> tensor<512xi32, #blocked>
    %2 = triton_gpu.convert_layout %1 : tensor<512xi32, #blocked> -> tensor<512xi32, #blocked2>
    tt.return %2 : tensor<512xi32, #blocked2>
}
}  // end module

// -----

// CHECK: #[[$BLOCKED:.*]] = #triton_gpu.blocked
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @cvt_from_dot_op_into_local_allow_not_canonicalized(%in: tensor<256x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> !tt.memdesc<256x32xf32, #shared1> {
    // CHECK-LABEL: cvt_from_dot_op_into_local_allow_not_canonicalized
    %cvt_in = triton_gpu.convert_layout %in : tensor<256x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<256x32xf32, #blocked>
    %alloc = triton_gpu.local_alloc %cvt_in : (tensor<256x32xf32, #blocked>) -> !tt.memdesc<256x32xf32, #shared1>
    // CHECK: %[[ALLOC:.*]] = triton_gpu.local_alloc {{.*}} (tensor<{{.*}}, #[[$BLOCKED]]{{.*}}>) ->
    tt.return %alloc : !tt.memdesc<256x32xf32, #shared1>
  }
} // end module

