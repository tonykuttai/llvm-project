// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @acc_type_return
func.func @acc_type_return(%a: vector<4xf32>, %b: vector<4xf32>) -> !ppc_mma.acc {
  // CHECK: -> !ppc_mma.acc
  %acc = ppc_mma.xvf32ger %a, %b
    : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  return %acc : !ppc_mma.acc
}

// CHECK-LABEL: func @acc_type_argument
func.func @acc_type_argument(%acc: !ppc_mma.acc) -> !ppc_mma.acc {
  // CHECK: !ppc_mma.acc
  return %acc : !ppc_mma.acc
}

// CHECK-LABEL: func @multiple_acc_types
func.func @multiple_acc_types() -> (!ppc_mma.acc, !ppc_mma.acc) {
  // CHECK: -> (!ppc_mma.acc, !ppc_mma.acc)
  %cst = arith.constant dense<0.0> : vector<4xf32>
  %acc0 = ppc_mma.xvf32ger %cst, %cst : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  %acc1 = ppc_mma.xvf32ger %cst, %cst : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  return %acc0, %acc1 : !ppc_mma.acc, !ppc_mma.acc
}

// CHECK-LABEL: func @acc_type_in_tuple
func.func @acc_type_in_tuple(%acc0: !ppc_mma.acc, %acc1: !ppc_mma.acc) 
    -> (!ppc_mma.acc, !ppc_mma.acc, vector<4xf32>) {
  // CHECK: -> (!ppc_mma.acc, !ppc_mma.acc, vector<4xf32>)
  %cst = arith.constant dense<1.0> : vector<4xf32>
  return %acc0, %acc1, %cst : !ppc_mma.acc, !ppc_mma.acc, vector<4xf32>
}

// CHECK-LABEL: func @acc_type_printing
func.func @acc_type_printing() {
  %cst = arith.constant dense<0.0> : vector<4xf32>
  // Verify type printing works correctly
  // CHECK: !ppc_mma.acc
  %acc = ppc_mma.xvf32ger %cst, %cst : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  return
}

// CHECK-LABEL: func @multiple_acc_operations
func.func @multiple_acc_operations(%a: vector<4xf32>, %b: vector<4xf32>) 
    -> (!ppc_mma.acc, !ppc_mma.acc, !ppc_mma.acc) {
  // CHECK: -> (!ppc_mma.acc, !ppc_mma.acc, !ppc_mma.acc)
  %acc0 = ppc_mma.xvf32ger %a, %b : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  %acc1 = ppc_mma.xvf32ger %a, %b : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  %acc2 = ppc_mma.xvf32ger %a, %b : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  return %acc0, %acc1, %acc2 : !ppc_mma.acc, !ppc_mma.acc, !ppc_mma.acc
}