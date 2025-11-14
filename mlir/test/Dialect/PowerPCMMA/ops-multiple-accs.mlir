// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @two_accumulators
func.func @two_accumulators(
    %a0: vector<4xf32>, %a1: vector<4xf32>,
    %b0: vector<4xf32>, %b1: vector<4xf32>
) -> (!ppc_mma.acc, !ppc_mma.acc) {
  // Simulate computing two 4x4 tiles in parallel
  // CHECK: %[[ACC0:.*]] = ppc_mma.xvf32ger %arg0, %arg2
  %acc0 = ppc_mma.xvf32ger %a0, %b0
    : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  // CHECK: %[[ACC1:.*]] = ppc_mma.xvf32ger %arg1, %arg3
  %acc1 = ppc_mma.xvf32ger %a1, %b1
    : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  return %acc0, %acc1 : !ppc_mma.acc, !ppc_mma.acc
}

// CHECK-LABEL: func @eight_accumulators
func.func @eight_accumulators(
    %a: vector<4xf32>, %b: vector<4xf32>
) -> (!ppc_mma.acc, !ppc_mma.acc, !ppc_mma.acc, !ppc_mma.acc,
      !ppc_mma.acc, !ppc_mma.acc, !ppc_mma.acc, !ppc_mma.acc) {
  // Simulate OpenBLAS pattern with 8 ACCs
  // CHECK-COUNT-8: ppc_mma.xvf32ger
  %acc0 = ppc_mma.xvf32ger %a, %b : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  %acc1 = ppc_mma.xvf32ger %a, %b : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  %acc2 = ppc_mma.xvf32ger %a, %b : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  %acc3 = ppc_mma.xvf32ger %a, %b : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  %acc4 = ppc_mma.xvf32ger %a, %b : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  %acc5 = ppc_mma.xvf32ger %a, %b : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  %acc6 = ppc_mma.xvf32ger %a, %b : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  %acc7 = ppc_mma.xvf32ger %a, %b : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  return %acc0, %acc1, %acc2, %acc3, %acc4, %acc5, %acc6, %acc7 
    : !ppc_mma.acc, !ppc_mma.acc, !ppc_mma.acc, !ppc_mma.acc,
      !ppc_mma.acc, !ppc_mma.acc, !ppc_mma.acc, !ppc_mma.acc
}

// CHECK-LABEL: func @mixed_operations_multiple_accs
func.func @mixed_operations_multiple_accs(
    %a0: vector<4xf32>, %a1: vector<4xf32>,
    %b0: vector<4xf32>, %b1: vector<4xf32>
) -> (vector<4xf32>, vector<4xf32>) {
  // Initialize two ACCs
  // CHECK: %[[ACC0:.*]] = ppc_mma.xvf32ger
  %acc0 = ppc_mma.xvf32ger %a0, %b0
    : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  // CHECK: %[[ACC1:.*]] = ppc_mma.xvf32ger
  %acc1 = ppc_mma.xvf32ger %a1, %b1
    : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  // Accumulate into both
  // CHECK: %[[ACC0_2:.*]] = ppc_mma.xvf32gerpp %[[ACC0]]
  %acc0_2 = ppc_mma.xvf32gerpp %acc0, %a0, %b0
    : (!ppc_mma.acc, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  // CHECK: %[[ACC1_2:.*]] = ppc_mma.xvf32gerpp %[[ACC1]]
  %acc1_2 = ppc_mma.xvf32gerpp %acc1, %a1, %b1
    : (!ppc_mma.acc, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  // Extract from both
  // CHECK: ppc_mma.disassemble_acc %[[ACC0_2]]
  %r00, %r01, %r02, %r03 = ppc_mma.disassemble_acc %acc0_2
    : (!ppc_mma.acc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
  
  // CHECK: ppc_mma.disassemble_acc %[[ACC1_2]]
  %r10, %r11, %r12, %r13 = ppc_mma.disassemble_acc %acc1_2
    : (!ppc_mma.acc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
  
  return %r00, %r10 : vector<4xf32>, vector<4xf32>
}