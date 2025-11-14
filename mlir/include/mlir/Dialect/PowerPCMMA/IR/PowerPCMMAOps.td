//===- PowerPCMMAOps.td - PowerPC MMA dialect ops ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POWERPCCMMA_OPS
#define POWERPCCMMA_OPS

include "mlir/Dialect/PowerPCMMA/IR/PowerPCMMADialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Base Operation Class
//===----------------------------------------------------------------------===//

class PowerPCMMA_Op<string mnemonic, list<Trait> traits = []> :
    Op<PowerPCMMA_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// F32 Operations (4x4 outer product)
//===----------------------------------------------------------------------===//

def XvF32GerOp : PowerPCMMA_Op<"xvf32ger", [Pure]> {
  let summary = "F32 4x4 outer product (initialize accumulator)";
  let description = [{
    Performs a 4x4 single-precision floating-point outer product.
    This operation initializes (primes) the accumulator register.
    
    Equivalent to: __builtin_mma_xvf32ger(&acc, va, vb)
    Hardware: xvf32ger ACC, VSX, VSX
    
    Example:
```mlir
    %acc = ppc_mma.xvf32ger %lhs, %rhs 
      : (vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
```
  }];
  
  let arguments = (ins
    VectorOfLengthAndType<[4], [F32]>:$lhs,
    VectorOfLengthAndType<[4], [F32]>:$rhs
  );
  
  let results = (outs AccType:$acc);
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($acc)
  }];
}

def XvF32GerPPOp : PowerPCMMA_Op<"xvf32gerpp", [Pure]> {
  let summary = "F32 4x4 outer product with accumulation";
  let description = [{
    Performs a 4x4 single-precision floating-point outer product and
    accumulates the result into an existing accumulator.
    
    Equivalent to: __builtin_mma_xvf32gerpp(&acc, va, vb)
    Hardware: xvf32gerpp ACC, VSX, VSX
    
    Example:
```mlir
    %acc1 = ppc_mma.xvf32gerpp %acc0, %lhs, %rhs
      : (!ppc_mma.acc, vector<4xf32>, vector<4xf32>) -> !ppc_mma.acc
```
  }];
  
  let arguments = (ins
    AccType:$acc_in,
    VectorOfLengthAndType<[4], [F32]>:$lhs,
    VectorOfLengthAndType<[4], [F32]>:$rhs
  );
  
  let results = (outs AccType:$acc_out);
  
  let assemblyFormat = [{
    $acc_in `,` $lhs `,` $rhs attr-dict `:`
    `(` type($acc_in) `,` type($lhs) `,` type($rhs) `)` `->` type($acc_out)
  }];
}

//===----------------------------------------------------------------------===//
// Accumulator Management Operations
//===----------------------------------------------------------------------===//

def DisassembleAccOp : PowerPCMMA_Op<"disassemble_acc", [Pure]> {
  let summary = "Extract 4x4 result from accumulator";
  let description = [{
    Extracts the 4x4 result from an accumulator as four row vectors.
    
    The details of primed/unprimed state and any required xxmfacc 
    instructions are handled during lowering to LLVM/PowerPC.
    
    Equivalent to: __builtin_mma_disassemble_acc(result, &acc)
    
    Example:
```mlir
    %row0, %row1, %row2, %row3 = ppc_mma.disassemble_acc %acc
      : (!ppc_mma.acc) -> (vector<4xf32>, vector<4xf32>, 
                            vector<4xf32>, vector<4xf32>)
```
  }];
  
  let arguments = (ins AccType:$acc);
  
  let results = (outs
    VectorOfLengthAndType<[4], [F32]>:$row0,
    VectorOfLengthAndType<[4], [F32]>:$row1,
    VectorOfLengthAndType<[4], [F32]>:$row2,
    VectorOfLengthAndType<[4], [F32]>:$row3
  );
  
  let assemblyFormat = [{
    $acc attr-dict `:` `(` type($acc) `)` `->` 
    `(` type($row0) `,` type($row1) `,` type($row2) `,` type($row3) `)`
  }];
}

def AssembleAccOp : PowerPCMMA_Op<"assemble_acc", [Pure]> {
  let summary = "Construct accumulator from 4 row vectors";
  let description = [{
    Creates an accumulator from four row vectors.
    
    Used when converting from vector<4x4xf32> to !ppc_mma.acc.
    
    Example:
```mlir
    %acc = ppc_mma.assemble_acc %row0, %row1, %row2, %row3
      : (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
      -> !ppc_mma.acc
```
  }];
  
  let arguments = (ins
    VectorOfLengthAndType<[4], [F32]>:$row0,
    VectorOfLengthAndType<[4], [F32]>:$row1,
    VectorOfLengthAndType<[4], [F32]>:$row2,
    VectorOfLengthAndType<[4], [F32]>:$row3
  );
  
  let results = (outs AccType:$acc);
  
  let assemblyFormat = [{
    $row0 `,` $row1 `,` $row2 `,` $row3 attr-dict `:`
    `(` type($row0) `,` type($row1) `,` type($row2) `,` type($row3) `)` 
    `->` type($acc)
  }];
}

#endif // POWERPCCMMA_OPS