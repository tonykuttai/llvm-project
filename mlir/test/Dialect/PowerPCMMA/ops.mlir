// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: func @xvf32ger_basic
func.func @xvf32ger_basic(%lhs: vector<4xf32>, %rhs: vector<4xf32>) -> !ppc_mma.acc {
  // CHECK: ppc_mma.xvf32ger
  // CHECK-SAME: (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  %acc = ppc_mma.xvf32ger %lhs, %rhs 
    : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  return %acc : !ppc_mma.acc
}

// CHECK-LABEL: func @xvf32gerpp_basic
func.func @xvf32gerpp_basic(%acc_in: !ppc_mma.acc, %lhs: vector<4xf32>, %rhs: vector<4xf32>) -> !ppc_mma.acc {
  // CHECK: ppc_mma.xvf32gerpp
  // CHECK-SAME: (!ppc_mma.acc, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  %acc_out = ppc_mma.xvf32gerpp %acc_in, %lhs, %rhs
    : (!ppc_mma.acc, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  return %acc_out : !ppc_mma.acc
}

// CHECK-LABEL: func @disassemble_acc_basic
func.func @disassemble_acc_basic(%acc: !ppc_mma.acc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
  // CHECK: ppc_mma.disassemble_acc
  // CHECK-SAME: (!ppc_mma.acc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
  %r0, %r1, %r2, %r3 = ppc_mma.disassemble_acc %acc
    : (!ppc_mma.acc) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
  return %r0, %r1, %r2, %r3 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
}

// CHECK-LABEL: func @assemble_acc_basic
func.func @assemble_acc_basic(%r0: vector<4xf32>, %r1: vector<4xf32>, %r2: vector<4xf32>, %r3: vector<4xf32>) -> !ppc_mma.acc {
  // CHECK: ppc_mma.assemble_acc
  // CHECK-SAME: (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  %acc = ppc_mma.assemble_acc %r0, %r1, %r2, %r3
    : (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  return %acc : !ppc_mma.acc
}