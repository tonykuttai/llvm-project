// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
// Wrong vector size for xvf32ger
func.func @xvf32ger_wrong_size(%lhs: vector<8xf32>, %rhs: vector<4xf32>) -> !ppc_mma.acc {
  // expected-error @+1 {{must be vector of 32-bit float values of length 4}}
  %acc = ppc_mma.xvf32ger %lhs, %rhs
    : (vector<8xf32>, vector<4xf32>) -> !ppc_mma.acc
  return %acc : !ppc_mma.acc
}

// -----
// Wrong element type for xvf32ger
func.func @xvf32ger_wrong_elt_type(%lhs: vector<4xi32>, %rhs: vector<4xf32>) -> !ppc_mma.acc {
  // expected-error @+1 {{must be vector of 32-bit float values of length 4}}
  %acc = ppc_mma.xvf32ger %lhs, %rhs
    : (vector<4xi32>, vector<4xf32>) -> !ppc_mma.acc
  return %acc : !ppc_mma.acc
}

// -----
// xvf32gerpp with wrong accumulator type
func.func @xvf32gerpp_wrong_acc_type(%acc_in: vector<16xf32>,
                                     %lhs: vector<4xf32>, %rhs: vector<4xf32>)
    -> !ppc_mma.acc {
  // expected-error @+2 {{invalid kind of type specified}}
  %acc_out = ppc_mma.xvf32gerpp %acc_in, %lhs, %rhs
    : (vector<16xf32>, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
  return %acc_out : !ppc_mma.acc
}

// -----
// disassemble_acc with wrong accumulator type
func.func @disassemble_wrong_acc_type(%acc: vector<16xf32>)
    -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
  // expected-error @+2 {{invalid kind of type specified}}
  %r0, %r1, %r2, %r3 = ppc_mma.disassemble_acc %acc
    : (vector<16xf32>) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
  return %r0, %r1, %r2, %r3
    : (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
}

// -----
// assemble_acc with wrong row vector type
func.func @assemble_wrong_row_type(%r0: vector<4xf32>, %r1: vector<4xf32>,
                                   %r2: vector<4xf32>, %r3: vector<2xf32>)
    -> !ppc_mma.acc {
  // expected-error @+1 {{must be vector of 32-bit float values of length 4}}
  %acc = ppc_mma.assemble_acc %r0, %r1, %r2, %r3
    : (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<2xf32>) -> !ppc_mma.acc
  return %acc : !ppc_mma.acc
}