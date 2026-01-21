import Config

# Configuration for ExMaxsimCpu
#
# The Rust NIF can be configured via environment variables:
#
# - RAYON_NUM_THREADS: Control Rayon thread pool size (default: number of CPUs)
# - OPENBLAS_NUM_THREADS: Set to 1 to avoid oversubscription with Rayon
# - MKL_NUM_THREADS: Set to 1 if using Intel MKL instead of OpenBLAS
# - VECLIB_MAXIMUM_THREADS: Set to 1 on macOS if not using Accelerate's threading
#
# For libxsmm builds:
# - LIBXSMM_DIR or LIBXSMM_LIB_DIR: Path to libxsmm installation

# Rustler configuration (optional, uses defaults if not specified)
# config :ex_maxsim_cpu, ExMaxsimCpu.Nif,
#   crate: :maxsim_cpu,
#   mode: :release
