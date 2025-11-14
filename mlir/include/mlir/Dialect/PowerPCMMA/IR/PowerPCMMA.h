//===- PowerPCMMA.h - PowerPC MMA dialect -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POWERPCCMMA_IR_POWERPCCMMA_H
#define MLIR_DIALECT_POWERPCCMMA_IR_POWERPCCMMA_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// PowerPCMMA Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PowerPCMMA/IR/PowerPCMMAOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// PowerPCMMA Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/PowerPCMMA/IR/PowerPCMMAOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// PowerPCMMA Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/PowerPCMMA/IR/PowerPCMMAOps.h.inc"

#endif // MLIR_DIALECT_POWERPCCMMA_IR_POWERPCCMMA_Hode 