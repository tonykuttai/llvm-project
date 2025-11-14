add_mlir_dialect_library(MLIRPowerPCMMADialect
  PowerPCMMADialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/PowerPCMMA

  DEPENDS
  MLIRPowerPCMMAOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)