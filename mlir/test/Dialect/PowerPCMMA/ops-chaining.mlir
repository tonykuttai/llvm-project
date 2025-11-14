// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @matmul_4x4_single_tile
func.func @matmul_4x4_single_tile(
    %a0: vector<4xf32>, %a1: vector<4xf32>, %a2: vector<4xf32>, %a3: vector<4xf32>,
    %b0: vector<4xf32>, %b1: vector<4xf32>, %b2: vector<4xf32>, %b3: vector<4xf32>
) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
  // Initialize accumulator with first outer product
  // CHECK: %[[ACC0:.*]] = ppc_mma.xvf32ger
  %acc0 = ppc_mma.xvf32ger %a0, %b0
    : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  // Accumulate remaining products
  // CHECK: %[[ACC1:.*]] = ppc_mma.xvf32gerpp %[[ACC0]]
  %acc1 = ppc_mma.xvf32gerpp %acc0, %a1, %b1
    : (!ppc_mma.acc, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  // CHECK: %[[ACC2:.*]] = ppc_mma.xvf32gerpp %[[ACC1]]
  %acc2 = ppc_mma.xvf32gerpp %acc1, %a2, %b2
    : (!ppc_mma.acc, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  // CHECK: %[[ACC3:.*]] = ppc_mma.xvf32gerpp %[[ACC2]]
  %acc3 = ppc_mma.xvf32gerpp %acc2, %a3, %b3
    : (!ppc_mma.acc, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  // Extract results
  // CHECK: ppc_mma.disassemble_acc %[[ACC3]]
  %r0, %r1, %r2, %r3 = ppc_mma.disassemble_acc %acc3
    : (!ppc_mma.acc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
  
  return %r0, %r1, %r2, %r3 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
}

// CHECK-LABEL: func @accumulator_reuse
func.func @accumulator_reuse(%lhs: vector<4xf32>, %rhs: vector<4xf32>) -> !ppc_mma.acc {
  // Test reusing the same accumulator multiple times (like k-loop)
  // CHECK: %[[ACC0:.*]] = ppc_mma.xvf32ger
  %acc0 = ppc_mma.xvf32ger %lhs, %rhs
    : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  // CHECK: %[[ACC1:.*]] = ppc_mma.xvf32gerpp %[[ACC0]]
  %acc1 = ppc_mma.xvf32gerpp %acc0, %lhs, %rhs
    : (!ppc_mma.acc, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  // CHECK: %[[ACC2:.*]] = ppc_mma.xvf32gerpp %[[ACC1]]
  %acc2 = ppc_mma.xvf32gerpp %acc1, %lhs, %rhs
    : (!ppc_mma.acc, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  // CHECK: %[[ACC3:.*]] = ppc_mma.xvf32gerpp %[[ACC2]]
  %acc3 = ppc_mma.xvf32gerpp %acc2, %lhs, %rhs
    : (!ppc_mma.acc, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  return %acc3 : !ppc_mma.acc
}

// CHECK-LABEL: func @assemble_disassemble_roundtrip
func.func @assemble_disassemble_roundtrip(
    %r0: vector<4xf32>, %r1: vector<4xf32>, 
    %r2: vector<4xf32>, %r3: vector<4xf32>
) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
  // Assemble then immediately disassemble (identity)
  // CHECK: %[[ACC:.*]] = ppc_mma.assemble_acc
  %acc = ppc_mma.assemble_acc %r0, %r1, %r2, %r3
    : (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  
  // CHECK: ppc_mma.disassemble_acc %[[ACC]]
  %out0, %out1, %out2, %out3 = ppc_mma.disassemble_acc %acc
    : (!ppc_mma.acc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
  
  return %out0, %out1, %out2, %out3 
    : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
}