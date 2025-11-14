//===- PowerPCMMADialect.td - PowerPC MMA dialect ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PowerPC MMA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPCCMMA_DIALECT
#define POWERPCCMMA_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// PowerPCMMA Dialect Definition
//===----------------------------------------------------------------------===//

def PowerPCMMA_Dialect : Dialect {
  let name = "ppc_mma";
  let summary = "PowerPC Matrix-Multiply Assist (MMA) dialect";
  let description = [{
    The PowerPC MMA dialect provides operations for the Matrix-Multiply Assist
    (MMA) facility introduced in POWER10 processors.
    
    MMA instructions provide hardware-accelerated matrix multiplication
    operations using special accumulator registers that hold 4x4 tiles of
    f32 values.
    
    Key operations:
    - xvf32ger: Initialize 4x4 accumulator (outer product)
    - xvf32gerpp: Accumulate to 4x4 accumulator (outer product + accumulate)
    - disassemble_acc: Extract 4 rows from accumulator
  }];
  
  let cppNamespace = "::mlir::ppc_mma";
  
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 0;
}

//===----------------------------------------------------------------------===//
// PowerPCMMA Type Definitions
//===----------------------------------------------------------------------===//

class PowerPCMMA_Type<string name, string typeMnemonic>
    : TypeDef<PowerPCMMA_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def AccType : PowerPCMMA_Type<"Acc", "acc"> {
  let summary = "MMA accumulator type";
  let description = [{
    Represents a 512-bit MMA accumulator register (ACC).
    
    An accumulator holds a 4x4 tile of f32 values (16 floats total).
    Accumulators can be in two states:
    - Primed: Active accumulator that can receive xvf32gerpp operations
    - Unprimed: Passive state for reading results
    
    The hardware manages the primed/unprimed state automatically.
  }];
}

#endif // POWERPCCMMA_DIALECT