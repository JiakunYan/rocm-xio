# SDMA Endpoint Ping-Pong Latency Benchmark

Measures round-trip latency of different SDMA communication patterns between two AMD GPUs.

## Prerequisites

- Two AMD GPUs with XGMI / Infinity Fabric P2P connectivity
- ROCm-XIO installed
- Root access (required for `/dev/kfd` access)

## Building

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build
```

## Usage

```bash
sudo ./build/sdma-ep-ping-pong --mode <1|2|3|4|5|6> [options]
```

### Benchmark Modes

#### Mode 1: Device-Initiated Ping-Pong (Baseline)
GPU-to-GPU communication initiated entirely from device code. This represents the baseline latency for SDMA operations.

```bash
sudo ./build/sdma-ep-ping-pong --mode 1
```

**Pattern:**
- GPU0 sends data to GPU1 via `putSignal`
- GPU1 waits for signal, then responds with `putSignal` back to GPU0
- GPU0 measures full round-trip time

#### Mode 2: Device-Triggered (XIO wait_flag_then_put)
GPU triggers host SDMA to perform the transfer using the XIO `wait_flag_then_put` primitive.

```bash
sudo ./build/sdma-ep-ping-pong --mode 2
```

**Pattern:**
- GPU updates a trigger flag in device memory
- Host SDMA queue (pre-enqueued) wakes up via `wait_flag_then_put`, performs the DMA, and signals the peer GPU
- GPU measures time from trigger to response

#### Mode 3: Reverse Offload (CPU Proxy + SDMA)
GPU writes to system memory, CPU proxy thread monitors and responds using host-initiated SDMA.

```bash
sudo ./build/sdma-ep-ping-pong --mode 3
```

**Pattern:**
- GPU writes request counter to system memory
- CPU thread busy-waits on system memory
- CPU responds via host-initiated SDMA `put_signal`
- GPU polls for response signal

#### Mode 4: Reverse Offload (CPU Proxy + hipMemcpy)
Same as Mode 3, but uses `hipMemcpyAsync` instead of SDMA for the response.

```bash
sudo ./build/sdma-ep-ping-pong --mode 4
```

**Use case:** Compare SDMA vs HIP runtime overhead for host-to-device transfers.

#### Mode 5: Chained SDMA (GPU Signal → Host SDMA)
Ping GPU signals the pong host via SDMA, which immediately responds with a host-side SDMA transfer—no pong kernel required.

```bash
sudo ./build/sdma-ep-ping-pong --mode 5
```

**Pattern:**
- Ping GPU signals a trigger flag in peer memory
- Peer host SDMA queue waits on that trigger, performs the DMA back to the ping GPU, and signals completion
- Only the ping GPU measures latency

#### Mode 6: Device-Triggered (HIP stream wait + hipMemcpyAsync)
Replaces the XIO `wait_flag_then_put` path with HIP stream primitives and runtime copies.

```bash
sudo ./build/sdma-ep-ping-pong --mode 6
```

**Pattern:**
- GPU writes iteration IDs into a trigger flag
- A non-blocking stream waits on those values (`hipStreamWaitValue32`) and performs the device-to-device copy
- A host-generated counter is written to the peer signal via `hipMemcpyAsync` after the payload transfer completes

### Command-Line Options

- `--mode <1|2|3|4|5|6>` - Select benchmark mode (default: 1)
- `--src-gpu <N>` - Source GPU ID (default: 0)
- `--dst-gpu <N>` - Destination GPU ID (default: 1)
- `--iterations <N>` - Number of measurement iterations (default: 1000)
- `--warmup <N>` - Number of warmup iterations (default: 10)
- `--output-format <format>` - Output format: text, json, csv (default: text)
- `--output-file <path>` - Output file path (required for json/csv)
- `--help` - Show help message

### Transfer Sizes

The benchmark automatically sweeps over multiple transfer sizes:
- 64 bytes - Protocol overhead dominates
- 256 bytes - Small message latency
- 1 KB - Medium message
- 4 KB - Page-sized transfer
- 16 KB - Larger transfer

## Running All Modes

Use the provided bash script to run all modes and save results:

```bash
./run_all_modes.sh [options]

Options:
  --src-gpu N        Source GPU (default: 0)
  --dst-gpu N        Destination GPU (default: 1)
  --iterations N     Number of iterations (default: 1000)
  --warmup N         Warmup iterations (default: 10)
  --output-dir DIR   Output directory (default: ./results)
```

This will create JSON files for modes 1, 2 (XIO), 6 (HIP), 3, 4, and 5 in the results directory.

## Plotting Results

After running all modes, generate plots using the Python script:

```bash
# Install dependencies if needed
pip install matplotlib numpy

# Generate plots
python3 plot_results.py results/ --output latency_comparison.png --individual --table
```

Options:
- `--output FILE` - Save comparison plot to file
- `--individual` - Generate individual breakdown plots for each mode
- `--table` - Print summary table to console

## Example Output

**Text output:**
```
Source GPU 0: AMD Instinct MI300X
Dest   GPU 1: AMD Instinct MI300X
Iterations: 1000 (warmup: 10)
Transfer sizes: 64 256 1024 4096 16384 bytes

=== Mode 1: Device-Initiated Ping-Pong ===

Device-Initiated [64 bytes]:
  Min: 1.234 us (987 cycles)
  Max: 2.346 us (1876 cycles)
  Avg: 1.457 us (1165.42 cycles)

Device-Initiated [256 bytes]:
  Min: 1.289 us (1031 cycles)
  Max: 2.401 us (1920 cycles)
  Avg: 1.512 us (1209.56 cycles)
...
```

**JSON output** (`--output-format json --output-file results.json`):
```json
{
  "benchmark": "sdma-ep-ping-pong",
  "mode": 1,
  "src_gpu": 0,
  "dst_gpu": 1,
  "iterations": 1000,
  "warmup": 10,
  "results": [
    {
      "mode_name": "Device-Initiated",
      "transfer_size_bytes": 64,
      "min_latency_us": 1.234,
      "max_latency_us": 2.346,
      "avg_latency_us": 1.457,
      "min_cycles": 987,
      "max_cycles": 1876,
      "avg_cycles": 1165.42
    }
  ]
}
```

**CSV output** (`--output-format csv --output-file results.csv`):
```csv
mode_name,transfer_size_bytes,min_latency_us,max_latency_us,avg_latency_us,min_cycles,max_cycles,avg_cycles
"Device-Initiated",64,1.234,2.346,1.457,987,1876,1165.42
"Device-Initiated",256,1.289,2.401,1.512,1031,1920,1209.56
...
```

## Interpretation

- **Mode 1**: pure device-to-device baseline
- **Mode 2 vs Mode 6**: compares XIO `wait_flag_then_put` against HIP stream wait + hipMemcpy for GPU-triggered transfers
- **Mode 3 vs Mode 4**: contrasts SDMA vs hipMemcpy for reverse offload through a CPU proxy
- **Mode 5**: demonstrates chained SDMA, where the ping GPU signals the host SDMA response without a pong kernel
- Smaller transfer sizes isolate protocol overhead; larger sizes show bandwidth trends

## Notes

- Warmup iterations ensure caches are populated and queues are initialized
- CPU proxy threads use busy-wait polling for lowest latency measurement
- All data buffers use uncached memory (`hipDeviceMallocUncached`) for SDMA visibility
- Signal buffers also use uncached memory for coherence
- Timing is measured using `wall_clock64()` GPU instruction for cycle-accurate measurement
