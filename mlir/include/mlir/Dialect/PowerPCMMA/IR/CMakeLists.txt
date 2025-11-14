set(LLVM_TARGET_DEFINITIONS PowerPCMMAOps.td)

# Generate dialect operations
add_mlir_dialect(PowerPCMMAOps ppc_mma)

# Generate type declarations and definitions
mlir_tablegen(PowerPCMMAOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=ppc_mma)
mlir_tablegen(PowerPCMMAOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ppc_mma)
add_public_tablegen_target(MLIRPowerPCMMAOpsTypesIncGen)

# Generate documentation
add_mlir_doc(PowerPCMMAOps PowerPCMMAOps Dialects/ -gen-dialect-doc)

# Ensure headers are generated before other targets use them
add_dependencies(mlir-headers MLIRPowerPCMMAOpsTypesIncGen)