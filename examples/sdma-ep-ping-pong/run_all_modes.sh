#!/bin/bash
# Run all SDMA-EP ping-pong benchmark modes and save results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="${SCRIPT_DIR}/build/sdma-ep-ping-pong"
OUTPUT_DIR="${SCRIPT_DIR}/results"

# Default parameters
SRC_GPU=0
DST_GPU=1
ITERATIONS=1000
WARMUP=10

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --src-gpu)
      SRC_GPU="$2"
      shift 2
      ;;
    --dst-gpu)
      DST_GPU="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--src-gpu N] [--dst-gpu N] [--iterations N] [--warmup N] [--output-dir DIR]"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "SDMA-EP Ping-Pong Benchmark Suite"
echo "=========================================="
echo "Source GPU:   ${SRC_GPU}"
echo "Dest GPU:     ${DST_GPU}"
echo "Iterations:   ${ITERATIONS}"
echo "Warmup:       ${WARMUP}"
echo "Output Dir:   ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Run each mode (placing Mode 6 right after Mode 2 for comparison)
for MODE in 1 2 6 3 4 5; do
  MODE_NAMES=("" "device-initiated" "device-triggered-xio" "reverse-offload-sdma" "reverse-offload-hipmemcpyasync" "chained-sdma" "device-triggered-hip")
  MODE_NAME="${MODE_NAMES[$MODE]}"

  echo "Running Mode ${MODE}: ${MODE_NAME}..."

  JSON_FILE="${OUTPUT_DIR}/mode${MODE}_${MODE_NAME}.json"

  mpirun -n 2 "${BENCHMARK}" \
    --mode "${MODE}" \
    --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}" \
    --output-format json \
    --output-file "${JSON_FILE}"


  echo "  → Saved: ${JSON_FILE}"
  echo ""
done

echo "=========================================="
echo "All modes completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "To plot results, run:"
echo "  python3 ${SCRIPT_DIR}/plot_results.py ${OUTPUT_DIR}"
echo "=========================================="
