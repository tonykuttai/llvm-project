//===- PowerPCMMADialect.cpp - PowerPC MMA dialect -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PowerPCMMA/IR/PowerPCMMA.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ppc_mma;

//===----------------------------------------------------------------------===//
// PowerPCMMA Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PowerPCMMA/IR/PowerPCMMAOpsDialect.cpp.inc"

void PowerPCMMADialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/PowerPCMMA/IR/PowerPCMMAOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/PowerPCMMA/IR/PowerPCMMAOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// PowerPCMMA Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/PowerPCMMA/IR/PowerPCMMAOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// PowerPCMMA Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/PowerPCMMA/IR/PowerPCMMAOps.cpp.inc"