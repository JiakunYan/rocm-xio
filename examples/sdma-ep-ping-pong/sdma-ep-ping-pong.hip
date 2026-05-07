/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * SDMA Endpoint Ping-Pong Latency Benchmark
 *
 * Measures round-trip latency of different SDMA communication patterns:
 * - Mode 1: Device-initiated ping-pong (baseline)
 * - Mode 2: Device-triggered (XIO wait_flag_then_put)
 * - Mode 3: Reverse offload with CPU proxy (SDMA or hipMemcpy)
 * - Mode 4: Reverse offload with CPU proxy using hipMemcpy
 * - Mode 5: Chained SDMA (GPU signal → Host SDMA response)
 * - Mode 6: Device-triggered (HIP stream wait + hipMemcpyAsync)
 *
 * Prerequisites:
 *   - Two AMD GPUs with XGMI / Infinity Fabric P2P
 *   - rocm-xio installed (find_package(rocm-xio))
 *   - Run as root (hsakmt requires /dev/kfd access)
 *
 * Usage:
 *   sudo ./sdma-ep-ping-pong --mode <1|2|3|4> [options]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <thread>
#include <memory>
#include <vector>
#include <algorithm>
#include <getopt.h>

#include <mpi.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include "xio.h"
#include "endpoints/sdma-ep/sdma-ep.h"
#include "endpoints/sdma-ep/sdma-host-queue.h"
#include "endpoints/sdma-ep/sdma_device.hpp"

// HIP_CHECK already defined in xio.h

struct BenchmarkArgs {
  int myGpu = -1;  // Will be set to MPI rank
  int peerGpu = -1;  // Will be set to peer rank
  int mode = 1;
  std::vector<size_t> transferSizes = {64, 256, 1024, 4096, 16384, 65536, 262144};
  int iterations = 1000;
  int warmup = 10;
  std::string outputFormat = "text";  // text, json, csv
  std::string outputFile = "";
  int myRank = 0;
  int nRanks = 1;
};

// Common memory buffers for all modes
struct PingPongBuffers {
  // Local buffers (allocated on myGpu)
  uint8_t* mySrcBuf = nullptr;
  uint8_t* myDstBuf = nullptr;
  uint64_t* mySignal = nullptr;

  // Peer buffers (IPC-opened from peer's memory)
  uint8_t* peerSrcBuf = nullptr;
  uint8_t* peerDstBuf = nullptr;
  uint64_t* peerSignal = nullptr;

  // Mode 2/5 specific: trigger flag (device memory, polled by host SDMA)
  uint32_t* myTriggerFlag = nullptr;

  // Mode 3 specific: request queue (system memory, polled by CPU)
  uint64_t* myRequestQueue = nullptr;

  // Mode 4 specific: host-accessible signal (pinned host memory)
  uint64_t* myHostSignal = nullptr;
  uint64_t* peerHostSignal = nullptr;

  // Latency measurement
  unsigned long long* latencies = nullptr;

  size_t bufferSize = 0;
  int totalIters = 0;
};

// ============================================================================
// Utility Functions
// ============================================================================

struct LatencyStats {
  const char* modeName;
  size_t transferSize;
  double minUs;
  double maxUs;
  double avgUs;
  unsigned long long minCycles;
  unsigned long long maxCycles;
  double avgCycles;
};

void reportLatency(const char* modeName, size_t transferSize,
                   unsigned long long* latencies, int count,
                   double cycleTimeNs, std::vector<LatencyStats>& allStats) {
  unsigned long long minLat = ULLONG_MAX;
  unsigned long long maxLat = 0;
  unsigned long long sumLat = 0;

  for (int i = 0; i < count; i++) {
    minLat = std::min(minLat, latencies[i]);
    maxLat = std::max(maxLat, latencies[i]);
    sumLat += latencies[i];
  }

  double avgLat = static_cast<double>(sumLat) / count;

  // Convert to microseconds
  double minUs = (minLat * cycleTimeNs) / 1000.0;
  double maxUs = (maxLat * cycleTimeNs) / 1000.0;
  double avgUs = (avgLat * cycleTimeNs) / 1000.0;

  // Store stats for later output
  allStats.push_back({modeName, transferSize, minUs, maxUs, avgUs,
                      minLat, maxLat, avgLat});

  printf("%s [%zu bytes]:\n", modeName, transferSize);
  printf("  Min: %.3f us (%llu cycles)\n", minUs, minLat);
  printf("  Max: %.3f us (%llu cycles)\n", maxUs, maxLat);
  printf("  Avg: %.3f us (%.2f cycles)\n\n", avgUs, avgLat);
}

double getGpuCycleTimeNs(int gpuId) {
  int clockRateKhz = 0;
  // wall_clock64() uses a fixed timestamp counter, not the GPU core clock
  HIP_CHECK(hipDeviceGetAttribute(&clockRateKhz, hipDeviceAttributeWallClockRate,
                                  gpuId));
  printf("clock rate %i\n", clockRateKhz);
  return 1e6 / clockRateKhz;  // ns per cycle
}

// Common setup function used by all modes
int setupPingPongBuffers(const BenchmarkArgs& args, size_t transferSize,
                         PingPongBuffers& buffers) {
  HIP_CHECK(hipSetDevice(args.myGpu));

  buffers.bufferSize = transferSize;
  buffers.totalIters = args.warmup + args.iterations;

  // Allocate local buffers on myGpu
  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&buffers.mySrcBuf),
                                  transferSize, hipDeviceMallocUncached));
  HIP_CHECK(hipMemset(buffers.mySrcBuf, 0xAB + args.myRank, transferSize));

  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&buffers.myDstBuf),
                                  transferSize, hipDeviceMallocUncached));
  HIP_CHECK(hipMemset(buffers.myDstBuf, 0x00, transferSize));

  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&buffers.mySignal),
                                  sizeof(uint64_t), hipDeviceMallocUncached));
  HIP_CHECK(hipMemset(buffers.mySignal, 0, sizeof(uint64_t)));

  // Export IPC handles
  hipIpcMemHandle_t mySrcHandle, myDstHandle, mySignalHandle;
  HIP_CHECK(hipIpcGetMemHandle(&mySrcHandle, buffers.mySrcBuf));
  HIP_CHECK(hipIpcGetMemHandle(&myDstHandle, buffers.myDstBuf));
  HIP_CHECK(hipIpcGetMemHandle(&mySignalHandle, buffers.mySignal));

  // Exchange IPC handles with peer via MPI
  hipIpcMemHandle_t peerSrcHandle, peerDstHandle, peerSignalHandle;
  MPI_Sendrecv(&mySrcHandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, 1 - args.myRank, 0,
               &peerSrcHandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, 1 - args.myRank, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&myDstHandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, 1 - args.myRank, 1,
               &peerDstHandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, 1 - args.myRank, 1,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&mySignalHandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, 1 - args.myRank, 2,
               &peerSignalHandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, 1 - args.myRank, 2,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Open peer's memory
  int prevDevice = -1;
  if (args.mode == 4) {
    HIP_CHECK(hipGetDevice(&prevDevice));
    HIP_CHECK(hipSetDevice(args.peerGpu));
  }
  HIP_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void**>(&buffers.peerSrcBuf), peerSrcHandle,
                                hipIpcMemLazyEnablePeerAccess));
  HIP_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void**>(&buffers.peerDstBuf), peerDstHandle,
                                hipIpcMemLazyEnablePeerAccess));
  HIP_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void**>(&buffers.peerSignal), peerSignalHandle,
                                hipIpcMemLazyEnablePeerAccess));
  if (args.mode == 4) {
    if (prevDevice >= 0 && prevDevice != args.peerGpu) {
      HIP_CHECK(hipSetDevice(prevDevice));
    }
  } 
  // Allocate latencies (only rank 0 measures)
  if (args.myRank == 0) {
    HIP_CHECK(hipMalloc(&buffers.latencies, buffers.totalIters * sizeof(unsigned long long)));
  }

  return 0;
}

void cleanupPingPongBuffers(PingPongBuffers& buffers) {
  // Close IPC handles
  if (buffers.peerSrcBuf) HIP_CHECK(hipIpcCloseMemHandle(buffers.peerSrcBuf));
  if (buffers.peerDstBuf) HIP_CHECK(hipIpcCloseMemHandle(buffers.peerDstBuf));
  if (buffers.peerSignal) HIP_CHECK(hipIpcCloseMemHandle(buffers.peerSignal));

  // Free local memory
  if (buffers.mySrcBuf) HIP_CHECK(hipFree(buffers.mySrcBuf));
  if (buffers.myDstBuf) HIP_CHECK(hipFree(buffers.myDstBuf));
  if (buffers.mySignal) HIP_CHECK(hipFree(buffers.mySignal));
  if (buffers.myTriggerFlag) HIP_CHECK(hipFree(buffers.myTriggerFlag));
  if (buffers.myRequestQueue) xio::freeHostMemory(buffers.myRequestQueue, XIO_HOST_MEM_DEFAULT);
  if (buffers.myHostSignal) HIP_CHECK(hipHostFree(buffers.myHostSignal));
  if (buffers.latencies) HIP_CHECK(hipFree(buffers.latencies));
}

// ============================================================================
// Mode 1: Device-Initiated Ping-Pong
// ============================================================================

__global__ void deviceInitiatedPingKernel(
    sdma_ep::SdmaQueueHandle* myQueue, sdma_ep::SdmaQueueHandle* peerQueue,
    void* myBuf, void* peerBuf, size_t size, uint64_t* mySignal,
    uint64_t* peerSignal, unsigned long long* latencies, int iterations) {
  // Load queue handle into register to avoid repeated memory accesses
  sdma_ep::SdmaQueueHandle myQueueReg = *myQueue;

  for (int i = 0; i < iterations; i++) {
    unsigned long long t0 = wall_clock64();

    // Send data + signal to peer
    sdma_ep::putSignal(myQueueReg, peerBuf, myBuf, size, peerSignal);

    // Wait for peer's response
    sdma_ep::waitSignal(mySignal, static_cast<uint64_t>(i + 1));

    latencies[i] = wall_clock64() - t0;
  }
}

__global__ void deviceInitiatedPongKernel(
    sdma_ep::SdmaQueueHandle* myQueue, void* myBuf, void* peerBuf, size_t size,
    uint64_t* mySignal, uint64_t* peerSignal, int iterations) {
  // Load queue handle into register to avoid repeated memory accesses
  sdma_ep::SdmaQueueHandle myQueueReg = *myQueue;

  for (int i = 0; i < iterations; i++) {
    // Wait for peer's ping
    sdma_ep::waitSignal(mySignal, static_cast<uint64_t>(i + 1));

    // Send data + signal back to peer
    sdma_ep::putSignal(myQueueReg, peerBuf, myBuf, size, peerSignal);
  }
}

int runDeviceInitiated(const BenchmarkArgs& args, std::vector<LatencyStats>& allStats) {
  if (args.myRank == 0) {
    printf("\n=== Mode 1: Device-Initiated Ping-Pong ===\n\n");
  }

  // Enable bidirectional peer access
  xio::enablePeerAccess(args.myGpu, args.peerGpu);

  // Each rank creates ONE queue from its own GPU to peer GPU
  sdma_ep::SdmaQueueInfo myQueue = {};
  int rc = sdma_ep::createQueue(args.myGpu, args.peerGpu, &myQueue);
  if (rc != 0) {
    fprintf(stderr, "Rank %d: ERROR: createQueue %d->%d failed (%d)\n",
            args.myRank, args.myGpu, args.peerGpu, rc);
    return 1;
  }

  double cycleTimeNs = getGpuCycleTimeNs(args.myGpu);

  for (size_t transferSize : args.transferSizes) {
    // Use common buffer setup
    PingPongBuffers buffers = {};
    setupPingPongBuffers(args, transferSize, buffers);

    MPI_Barrier(MPI_COMM_WORLD);

    // Both ranks use the SAME kernel - each only has myQueue
    // Mode 1: GPU writes to peer's Src buffer and signals peer's signal
    if (args.myRank == 0) {
      // Rank 0: ping first
      hipLaunchKernelGGL(
          deviceInitiatedPingKernel, dim3(1), dim3(1), 0, nullptr,
          static_cast<sdma_ep::SdmaQueueHandle*>(myQueue.deviceHandle),
          nullptr,  // peerQueue not needed - we only use myQueue
          buffers.mySrcBuf, buffers.peerSrcBuf, transferSize,
          buffers.mySignal, buffers.peerSignal, buffers.latencies,
          buffers.totalIters);
    } else {
      // Rank 1: pong (waits then responds)
      hipLaunchKernelGGL(deviceInitiatedPongKernel, dim3(1), dim3(1), 0,
                         nullptr,
                         static_cast<sdma_ep::SdmaQueueHandle*>(myQueue.deviceHandle),
                         buffers.mySrcBuf, buffers.peerSrcBuf, transferSize,
                         buffers.mySignal, buffers.peerSignal,
                         buffers.totalIters);
    }

    HIP_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 0 reports results
    if (args.myRank == 0) {
      unsigned long long* hostLatencies =
          new unsigned long long[args.iterations];
      HIP_CHECK(hipMemcpy(hostLatencies, buffers.latencies + args.warmup,
                          args.iterations * sizeof(unsigned long long),
                          hipMemcpyDeviceToHost));

      reportLatency("Device-Initiated", transferSize, hostLatencies,
                    args.iterations, cycleTimeNs, allStats);

      delete[] hostLatencies;
    }

    cleanupPingPongBuffers(buffers);
  }

  sdma_ep::destroyQueue(&myQueue);

  return 0;
}

// ============================================================================
// Mode 2: Host-Populated-Device-Triggered
// ============================================================================

// Rank 0 (ping): triggers host SDMA, waits for response
__global__ void deviceTriggerPingKernel(volatile uint32_t* triggerFlag,
                                        uint64_t* mySignal,
                                        unsigned long long* latencies,
                                        int iterations) {
  for (int i = 0; i < iterations; i++) {
    unsigned long long t0 = wall_clock64();

    // Trigger local host SDMA
    *triggerFlag = i + 1;
    __threadfence_system();

    // Wait for peer's host SDMA response (writes to mySignal)
    sdma_ep::waitSignal(mySignal, static_cast<uint64_t>(i + 1));

    latencies[i] = wall_clock64() - t0;
  }
}

// Rank 1 (pong): waits for peer's host SDMA, then triggers own
__global__ void deviceTriggerPongKernel(volatile uint32_t* triggerFlag,
                                        uint64_t* mySignal,
                                        int iterations) {
  for (int i = 0; i < iterations; i++) {
    // Wait for peer's host SDMA to signal us
    sdma_ep::waitSignal(mySignal, static_cast<uint64_t>(i + 1));

    // Trigger local host SDMA to respond
    *triggerFlag = i + 1;
    __threadfence_system();
  }
}

void preEnqueueWaitThenPut(sdma_ep::SdmaQueueInfo* queueInfo,
                            uint32_t* triggerFlag, void* src, void* dst,
                            size_t size, uint64_t* responseSignal, int iterations) {
  auto handle = sdma_ep::getHostHandle(queueInfo->srcDeviceId, queueInfo->dstDeviceId, 0);

  // Pre-enqueue all wait_then_put + signal operations
  // These execute asynchronously when GPU updates triggerFlag
  for (int i = 0; i < iterations; i++) {
    // wait_flag_then_put: SDMA polls triggerFlag, then copies when flag == i+1
    handle.wait_flag_then_put(triggerFlag, static_cast<uint32_t>(i + 1), src,
                              dst, size);

    // Signal GPU that response is ready (atomic ADD by 1)
    handle.signal(responseSignal, static_cast<uint64_t>(1));
  }

  // Submit all operations to SDMA queue
  // handle.quiet();
}

void enqueueStreamWaitThenMemcpy(int gpuId, hipStream_t stream,
                                 uint32_t* triggerFlag, void* src, void* dst,
                                 size_t size, uint64_t* responseSignal,
                                 int iterations) {
  int prevDevice = -1;
  HIP_CHECK(hipGetDevice(&prevDevice));
  HIP_CHECK(hipSetDevice(gpuId));
  for (int i = 0; i < iterations; i++) {
    HIP_CHECK(hipStreamWaitValue32(stream, triggerFlag, static_cast<uint32_t>(i + 1), 0));
    HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipStreamWriteValue64(stream, responseSignal, static_cast<uint64_t>(i + 1), 0));
  }
  if (prevDevice >= 0 && prevDevice != gpuId) {
    HIP_CHECK(hipSetDevice(prevDevice));
  }
}

int runHostTriggered(const BenchmarkArgs& args, std::vector<LatencyStats>& allStats) {
  if (args.myRank == 0) {
    printf("\n=== Mode 2: Device-Triggered (XIO) ===\n\n");
  }

  // Enable bidirectional peer access
  xio::enablePeerAccess(args.myGpu, args.peerGpu);

  // Create host queue for my host -> peer GPU transfers
  sdma_ep::SdmaQueueInfo queueInfo = {};
  int rc = sdma_ep::createHostQueue(args.myGpu, args.peerGpu, &queueInfo);
  if (rc != 0) {
    fprintf(stderr, "Rank %d: ERROR: createHostQueue failed (%d)\n",
            args.myRank, rc);
    return 1;
  }

  double cycleTimeNs = getGpuCycleTimeNs(args.myGpu);

  for (size_t transferSize : args.transferSizes) {
    // Use common buffer setup
    PingPongBuffers buffers = {};
    setupPingPongBuffers(args, transferSize, buffers);

    // Mode 2 specific: allocate trigger flag in device memory
    HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&buffers.myTriggerFlag),
                                    sizeof(uint32_t), hipDeviceMallocUncached));
    HIP_CHECK(hipMemset(buffers.myTriggerFlag, 0, sizeof(uint32_t)));

    // Pre-enqueue all wait_then_put operations (NOT on critical path)
    // Host SDMA polls myTriggerFlag, writes to peerDstBuf, signals peerSignal
    preEnqueueWaitThenPut(&queueInfo, buffers.myTriggerFlag,
                          buffers.mySrcBuf, buffers.peerDstBuf, transferSize,
                          buffers.peerSignal, buffers.totalIters);

    MPI_Barrier(MPI_COMM_WORLD);

    // Launch GPU kernel - ping-pong pattern
    // Each rank: GPU triggers LOCAL host SDMA → host SDMA writes to PEER GPU
    if (args.myRank == 0) {
      // Rank 0: ping (triggers host first, measures latency)
      hipLaunchKernelGGL(deviceTriggerPingKernel, dim3(1), dim3(1), 0, nullptr,
                         buffers.myTriggerFlag, buffers.mySignal, buffers.latencies,
                         buffers.totalIters);
    } else {
      // Rank 1: pong (waits for peer's host SDMA, then triggers own)
      hipLaunchKernelGGL(deviceTriggerPongKernel, dim3(1), dim3(1), 0, nullptr,
                         buffers.myTriggerFlag, buffers.mySignal, buffers.totalIters);
    }

    HIP_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 0 reports results
    if (args.myRank == 0) {
      unsigned long long* hostLatencies =
          new unsigned long long[args.iterations];
      HIP_CHECK(hipMemcpy(hostLatencies, buffers.latencies + args.warmup,
                          args.iterations * sizeof(unsigned long long),
                          hipMemcpyDeviceToHost));

      reportLatency("Device-Triggered (XIO)", transferSize, hostLatencies,
                    args.iterations, cycleTimeNs, allStats);

      delete[] hostLatencies;
    }

    cleanupPingPongBuffers(buffers);
  }

  sdma_ep::destroyQueue(&queueInfo);

  return 0;
}

int runStreamWaitMemcpy(const BenchmarkArgs& args, std::vector<LatencyStats>& allStats) {
  if (args.myRank == 0) {
    printf("\n=== Mode 6: Device-Triggered (HIP) ===\n\n");
  }

  xio::enablePeerAccess(args.myGpu, args.peerGpu);

  HIP_CHECK(hipSetDevice(args.myGpu));
  hipStream_t waitStream;
  HIP_CHECK(hipStreamCreateWithFlags(&waitStream, hipStreamNonBlocking));

  double cycleTimeNs = getGpuCycleTimeNs(args.myGpu);

  for (size_t transferSize : args.transferSizes) {
    PingPongBuffers buffers = {};
    setupPingPongBuffers(args, transferSize, buffers);

    HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&buffers.myTriggerFlag),
                                    sizeof(uint32_t), hipDeviceMallocUncached));
    HIP_CHECK(hipMemset(buffers.myTriggerFlag, 0, sizeof(uint32_t)));

    enqueueStreamWaitThenMemcpy(args.myGpu, waitStream, buffers.myTriggerFlag,
                                buffers.mySrcBuf, buffers.peerDstBuf, transferSize,
                                buffers.peerSignal, buffers.totalIters);

    MPI_Barrier(MPI_COMM_WORLD);

    if (args.myRank == 0) {
      hipLaunchKernelGGL(deviceTriggerPingKernel, dim3(1), dim3(1), 0, nullptr,
                         buffers.myTriggerFlag, buffers.mySignal, buffers.latencies,
                         buffers.totalIters);
    } else {
      hipLaunchKernelGGL(deviceTriggerPongKernel, dim3(1), dim3(1), 0, nullptr,
                         buffers.myTriggerFlag, buffers.mySignal, buffers.totalIters);
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipStreamSynchronize(waitStream));

    MPI_Barrier(MPI_COMM_WORLD);

    if (args.myRank == 0) {
      unsigned long long* hostLatencies =
          new unsigned long long[args.iterations];
      HIP_CHECK(hipMemcpy(hostLatencies, buffers.latencies + args.warmup,
                          args.iterations * sizeof(unsigned long long),
                          hipMemcpyDeviceToHost));

      reportLatency("Device-Triggered (HIP)", transferSize, hostLatencies,
                    args.iterations, cycleTimeNs, allStats);

      delete[] hostLatencies;
    }

    HIP_CHECK(hipStreamSynchronize(waitStream));
    cleanupPingPongBuffers(buffers);
  }

  HIP_CHECK(hipStreamDestroy(waitStream));

  return 0;
}

// ============================================================================
// Mode 5: Chained SDMA (GPU signal → Host SDMA response, no pong kernel)
// ============================================================================

// Rank 0 (ping): GPU signals pong-side host, waits for response
__global__ void chainedPingKernel(sdma_ep::SdmaQueueHandle* myQueue,
                                  uint32_t* peerTriggerFlag,
                                  uint64_t* mySignal,
                                  unsigned long long* latencies,
                                  int iterations) {
  sdma_ep::SdmaQueueHandle myQueueReg = *myQueue;

  for (int i = 0; i < iterations; i++) {
    unsigned long long t0 = wall_clock64();

    // Signal pong-side host to trigger SDMA response (32-bit signal)
    sdma_ep::signal(myQueueReg, peerTriggerFlag);

    // Wait for pong-side host SDMA response
    sdma_ep::waitSignal(mySignal, static_cast<uint64_t>(i + 1));

    latencies[i] = wall_clock64() - t0;
  }
}

// Rank 1 (pong): No kernel - just host thread waiting and responding
void chainedPongHost(sdma_ep::SdmaQueueInfo* queueInfo,
                     uint32_t* myTriggerFlag,
                     void* src, void* dst, size_t size,
                     uint64_t* responseSignal,
                     int iterations) {
  auto handle = sdma_ep::getHostHandle(queueInfo->srcDeviceId, queueInfo->dstDeviceId, 0);

  // Pre-enqueue all wait_then_put + signal operations
  for (int i = 0; i < iterations; i++) {
    // Wait for ping GPU signal, then immediately respond with SDMA
    handle.wait_flag_then_put(myTriggerFlag, static_cast<uint32_t>(i + 1),
                              src, dst, size);
    handle.signal(responseSignal, static_cast<uint64_t>(1));
  }
}

int runChained(const BenchmarkArgs& args, std::vector<LatencyStats>& allStats) {
  if (args.myRank == 0) {
    printf("\n=== Mode 5: Chained SDMA (GPU → Host SDMA) ===\n\n");
  }

  xio::enablePeerAccess(args.myGpu, args.peerGpu);

  // Create host-initiated queue
  sdma_ep::SdmaQueueInfo queueInfo = {};
  int rc = sdma_ep::createHostQueue(args.myGpu, args.peerGpu, &queueInfo);
  if (rc != 0) {
    fprintf(stderr, "Rank %d: ERROR: createHostQueue failed (%d)\n",
            args.myRank, rc);
    return 1;
  }

  // Only rank 0 needs device-initiated queue for signaling
  sdma_ep::SdmaQueueInfo myQueue = {};
  if (args.myRank == 0) {
    rc = sdma_ep::createQueue(args.myGpu, args.peerGpu, &myQueue);
    if (rc != 0) {
      fprintf(stderr, "Rank 0: ERROR: createQueue failed (%d)\n", rc);
      return 1;
    }
  }

  double cycleTimeNs = getGpuCycleTimeNs(args.myGpu);

  for (size_t transferSize : args.transferSizes) {
    PingPongBuffers buffers = {};
    setupPingPongBuffers(args, transferSize, buffers);

    // Allocate trigger flag (used by ping GPU to signal pong host)
    HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&buffers.myTriggerFlag),
                                    sizeof(uint32_t), hipDeviceMallocUncached));
    HIP_CHECK(hipMemset(buffers.myTriggerFlag, 0, sizeof(uint32_t)));

    // Exchange trigger flag IPC handles
    hipIpcMemHandle_t myTriggerHandle;
    HIP_CHECK(hipIpcGetMemHandle(&myTriggerHandle, buffers.myTriggerFlag));

    uint32_t* peerTriggerFlag = nullptr;
    hipIpcMemHandle_t peerTriggerHandle;
    MPI_Sendrecv(&myTriggerHandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, 1 - args.myRank, 0,
                 &peerTriggerHandle, sizeof(hipIpcMemHandle_t), MPI_BYTE, 1 - args.myRank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    HIP_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void**>(&peerTriggerFlag),
                                  peerTriggerHandle, hipIpcMemLazyEnablePeerAccess));

    HIP_CHECK(hipSetDevice(args.myGpu));

    // Rank 1: Pre-enqueue all wait_then_put operations before barrier
    if (args.myRank == 1) {
      chainedPongHost(&queueInfo, buffers.myTriggerFlag,
                      buffers.mySrcBuf, buffers.peerDstBuf, transferSize,
                      buffers.peerSignal, buffers.totalIters);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 0: Launch ping kernel after rank 1 has pre-enqueued
    if (args.myRank == 0) {
      hipLaunchKernelGGL(chainedPingKernel, dim3(1), dim3(1), 0, nullptr,
                         static_cast<sdma_ep::SdmaQueueHandle*>(myQueue.deviceHandle),
                         peerTriggerFlag, buffers.mySignal, buffers.latencies,
                         buffers.totalIters);
    }

    HIP_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 0 reports results
    if (args.myRank == 0) {
      unsigned long long* hostLatencies = new unsigned long long[args.iterations];
      HIP_CHECK(hipMemcpy(hostLatencies, buffers.latencies + args.warmup,
                          args.iterations * sizeof(unsigned long long),
                          hipMemcpyDeviceToHost));

      reportLatency("Chained SDMA", transferSize, hostLatencies,
                    args.iterations, cycleTimeNs, allStats);

      delete[] hostLatencies;
    }

    // Cleanup
    HIP_CHECK(hipIpcCloseMemHandle(peerTriggerFlag));
    if (buffers.myTriggerFlag) {
      HIP_CHECK(hipFree(buffers.myTriggerFlag));
      buffers.myTriggerFlag = nullptr;
    }
    cleanupPingPongBuffers(buffers);
  }

  if (args.myRank == 0) {
    sdma_ep::destroyQueue(&myQueue);
  }
  sdma_ep::destroyQueue(&queueInfo);

  return 0;
}

// ============================================================================
// Mode 3: Reverse Offload with CPU Proxy
// ============================================================================

// Rank 0 (ping): triggers CPU, waits for response
__global__ void reverseOffloadPingKernel(volatile uint64_t* cpuRequestQueue,
                                         uint64_t* mySignal,
                                         unsigned long long* latencies,
                                         int iterations) {
  for (int i = 0; i < iterations; i++) {
    unsigned long long t0 = wall_clock64();

    // Trigger local CPU proxy
    cpuRequestQueue[0] = i + 1;
    __threadfence_system();

    // Wait for peer's CPU response (peer CPU writes to mySignal)
    sdma_ep::waitSignal(mySignal, static_cast<uint64_t>(i + 1));

    latencies[i] = wall_clock64() - t0;
  }
}

// Rank 1 (pong): waits for peer's CPU, then responds via own CPU
__global__ void reverseOffloadPongKernel(volatile uint64_t* cpuRequestQueue,
                                         uint64_t* mySignal,
                                         int iterations) {
  for (int i = 0; i < iterations; i++) {
    // Wait for peer's CPU to signal us
    sdma_ep::waitSignal(mySignal, static_cast<uint64_t>(i + 1));

    // Trigger local CPU proxy to respond
    cpuRequestQueue[0] = i + 1;
    __threadfence_system();
  }
}

void cpuProxySdma(sdma_ep::SdmaQueueInfo* queueInfo, uint64_t* requestQueue,
                  void* src, void* dst, size_t size, uint64_t* responseSignal,
                  int iterations) {
  auto handle = sdma_ep::getHostHandle(queueInfo->srcDeviceId, queueInfo->dstDeviceId, 0);

  // Make local volatile pointer for polling
  volatile uint64_t* volatileQueue = requestQueue;

  for (int i = 0; i < iterations; i++) {
    // Busy-wait poll for lowest latency
    while (volatileQueue[0] < static_cast<uint64_t>(i + 1)) {
      /* tight spin */
    }

    // Respond via host-initiated SDMA (data + signal increment)
    // Signal is atomic ADD, so increment by 1 each iteration
    handle.put_signal(src, dst, size, responseSignal,
                      static_cast<uint64_t>(1));
  }
  handle.quiet();
}

void cpuProxyHipMemcpy(int localGpuId, int peerGpuId, uint64_t* requestQueue,
                       void* src, void* dst, size_t size,
                       uint64_t* responseSignal, int iterations,
                       hipStream_t* peerStream, uint64_t* hostSignal) {
  int prevDevice = -1;
  HIP_CHECK(hipGetDevice(&prevDevice));
  HIP_CHECK(hipSetDevice(localGpuId));

  volatile uint64_t* volatileQueue = requestQueue;

  for (int i = 0; i < iterations; i++) {
    while (volatileQueue[0] < static_cast<uint64_t>(i + 1)) {
      /* tight spin */
    }

    // For ping-pong we only need to signal, not copy data
    // (copying to IPC-opened memory via hipMemcpy might not work)
    HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, *peerStream));

    *hostSignal = static_cast<uint64_t>(i + 1);
    HIP_CHECK(hipSetDevice(peerGpuId));
    HIP_CHECK(hipMemcpyAsync(responseSignal, hostSignal, sizeof(uint64_t),
                             hipMemcpyHostToDevice, *peerStream));
    HIP_CHECK(hipStreamSynchronize(*peerStream));
    HIP_CHECK(hipSetDevice(localGpuId));

  }

  if (prevDevice >= 0 && prevDevice != localGpuId) {
    HIP_CHECK(hipSetDevice(prevDevice));
  }
}

int runReverseOffload(const BenchmarkArgs& args, bool useHipMemcpy, std::vector<LatencyStats>& allStats) {
  if (args.myRank == 0) {
    if (useHipMemcpy) {
      printf("\n=== Mode 4: Reverse Offload (CPU + hipMemcpy) ===\n\n");
    } else {
      printf("\n=== Mode 3: Reverse Offload (CPU + SDMA) ===\n\n");
    }
  }

  // Enable bidirectional peer access
  xio::enablePeerAccess(args.myGpu, args.peerGpu);

  // Create host queue for Mode 3a
  sdma_ep::SdmaQueueInfo queueInfo = {};
  int rc = 0;
  if (!useHipMemcpy) {
    rc = sdma_ep::createHostQueue(args.myGpu, args.peerGpu, &queueInfo);
    if (rc != 0) {
      fprintf(stderr, "Rank %d: ERROR: createHostQueue failed (%d)\n",
              args.myRank, rc);
      return 1;
    }
  }

  double cycleTimeNs = getGpuCycleTimeNs(args.myGpu);

  // Create comm stream for hipMemcpyAsync in this thread (peer GPU owns the signal)
  hipStream_t com_stream;
  int prevDevice = -1;
  HIP_CHECK(hipGetDevice(&prevDevice));
  HIP_CHECK(hipSetDevice(args.peerGpu));
  HIP_CHECK(hipStreamCreate(&com_stream));
  if (prevDevice >= 0 && prevDevice != args.peerGpu) {
    HIP_CHECK(hipSetDevice(prevDevice));
  }

  auto hostSignalValue = std::make_unique<uint64_t>(0);
  HIP_CHECK(hipSetDevice(args.myGpu));

  for (size_t transferSize : args.transferSizes) {
    // Use common buffer setup
    PingPongBuffers buffers = {};
    setupPingPongBuffers(args, transferSize, buffers);

    // Mode 3 specific: allocate request queue in system memory
    HIP_CHECK(xio::allocHostMemory(sizeof(uint64_t),
                                   reinterpret_cast<void**>(&buffers.myRequestQueue),
                                   "myRequestQueue", XIO_HOST_MEM_DEFAULT));
    buffers.myRequestQueue[0] = 0;

    printf("Rank %d: Mode 3 setup - myRequestQueue=%p (value=%lu), mySignal=%p, peerSignal=%p\n",
           args.myRank, buffers.myRequestQueue, buffers.myRequestQueue[0],
           buffers.mySignal, buffers.peerSignal);

    MPI_Barrier(MPI_COMM_WORLD);

    // Launch CPU proxy thread (monitors LOCAL myRequestQueue, writes to PEER signal)
    std::thread proxy;
    if (useHipMemcpy) {
      proxy = std::thread(cpuProxyHipMemcpy, args.myGpu, args.peerGpu, buffers.myRequestQueue,
                          buffers.mySrcBuf, buffers.peerDstBuf, transferSize,
                          buffers.peerSignal, buffers.totalIters, &com_stream, hostSignalValue.get());
    } else {
      proxy = std::thread(cpuProxySdma, &queueInfo, buffers.myRequestQueue,
                          buffers.mySrcBuf, buffers.peerDstBuf, transferSize,
                          buffers.peerSignal, buffers.totalIters);
    }

    // Launch GPU kernel - ping-pong pattern
    // Each rank: GPU triggers LOCAL CPU → CPU writes to PEER signal
    // GPU always polls mySignal (device memory) - difference is only HOW CPU writes
    if (args.myRank == 0) {
      // Rank 0: ping (triggers CPU first, measures latency)
      hipLaunchKernelGGL(reverseOffloadPingKernel, dim3(1), dim3(1), 0, nullptr,
                         buffers.myRequestQueue, buffers.mySignal, buffers.latencies,
                         buffers.totalIters);
    } else {
      // Rank 1: pong (waits for peer's CPU, then triggers own CPU)
      hipLaunchKernelGGL(reverseOffloadPongKernel, dim3(1), dim3(1), 0, nullptr,
                         buffers.myRequestQueue, buffers.mySignal,
                         buffers.totalIters);
    }

    HIP_CHECK(hipDeviceSynchronize());
    proxy.join();

    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 0 reports results
    if (args.myRank == 0) {
      unsigned long long* hostLatencies =
          new unsigned long long[args.iterations];
      HIP_CHECK(hipMemcpy(hostLatencies, buffers.latencies + args.warmup,
                          args.iterations * sizeof(unsigned long long),
                          hipMemcpyDeviceToHost));

      const char* modeName =
          useHipMemcpy ? "Reverse Offload (hipMemcpy)" : "Reverse Offload (SDMA)";
      reportLatency(modeName, transferSize, hostLatencies, args.iterations,
                    cycleTimeNs, allStats);

      delete[] hostLatencies;
    }

    cleanupPingPongBuffers(buffers);
  }

  HIP_CHECK(hipSetDevice(args.peerGpu));
  hostSignalValue.reset();
  HIP_CHECK(hipStreamDestroy(com_stream));
  HIP_CHECK(hipSetDevice(args.myGpu));

  return 0;
}

// ============================================================================
// Main and Argument Parsing
// ============================================================================

void writeJsonOutput(const std::string& filename, const BenchmarkArgs& args, std::vector<LatencyStats>& allStats) {
  FILE* f = fopen(filename.c_str(), "w");
  if (!f) {
    fprintf(stderr, "ERROR: Cannot open %s for writing\n", filename.c_str());
    return;
  }

  fprintf(f, "{\n");
  fprintf(f, "  \"benchmark\": \"sdma-ep-ping-pong\",\n");
  fprintf(f, "  \"mode\": %d,\n", args.mode);
  fprintf(f, "  \"num_ranks\": %d,\n", args.nRanks);
  fprintf(f, "  \"gpu_0\": 0,\n");
  fprintf(f, "  \"gpu_1\": 1,\n");
  fprintf(f, "  \"iterations\": %d,\n", args.iterations);
  fprintf(f, "  \"warmup\": %d,\n", args.warmup);
  fprintf(f, "  \"results\": [\n");

  for (size_t i = 0; i < allStats.size(); i++) {
    const auto& s = allStats[i];
    fprintf(f, "    {\n");
    fprintf(f, "      \"mode_name\": \"%s\",\n", s.modeName);
    fprintf(f, "      \"transfer_size_bytes\": %zu,\n", s.transferSize);
    fprintf(f, "      \"min_latency_us\": %.3f,\n", s.minUs);
    fprintf(f, "      \"max_latency_us\": %.3f,\n", s.maxUs);
    fprintf(f, "      \"avg_latency_us\": %.3f,\n", s.avgUs);
    fprintf(f, "      \"min_cycles\": %llu,\n", s.minCycles);
    fprintf(f, "      \"max_cycles\": %llu,\n", s.maxCycles);
    fprintf(f, "      \"avg_cycles\": %.2f\n", s.avgCycles);
    fprintf(f, "    }%s\n", (i < allStats.size() - 1) ? "," : "");
  }

  fprintf(f, "  ]\n");
  fprintf(f, "}\n");
  fclose(f);
  printf("Results written to: %s\n", filename.c_str());
}

void writeCsvOutput(const std::string& filename, const BenchmarkArgs& args, std::vector<LatencyStats>& allStats) {
  FILE* f = fopen(filename.c_str(), "w");
  if (!f) {
    fprintf(stderr, "ERROR: Cannot open %s for writing\n", filename.c_str());
    return;
  }

  fprintf(f, "mode_name,transfer_size_bytes,min_latency_us,max_latency_us,"
             "avg_latency_us,min_cycles,max_cycles,avg_cycles\n");

  for (const auto& s : allStats) {
    fprintf(f, "\"%s\",%zu,%.3f,%.3f,%.3f,%llu,%llu,%.2f\n",
            s.modeName, s.transferSize, s.minUs, s.maxUs, s.avgUs,
            s.minCycles, s.maxCycles, s.avgCycles);
  }

  fclose(f);
  printf("Results written to: %s\n", filename.c_str());
}

void printUsage(const char* progName) {
  printf("Usage: mpirun -n 2 %s [options]\n", progName);
  printf("Options:\n");
  printf("  --mode <1|2|3|4|5>   Benchmark mode (default: 1)\n");
  printf("                       1 = Device-initiated ping-pong\n");
  printf("                       2 = Device-triggered (XIO wait_flag_then_put)\n");
  printf("                       3 = Reverse offload (CPU + SDMA)\n");
  printf("                       4 = Reverse offload (CPU + hipMemcpy)\n");
  printf("                       5 = Chained SDMA (GPU signal → Host SDMA)\n");
  printf("                       6 = Device-triggered (HIP stream wait + hipMemcpyAsync)\n");
  printf("  --iterations <N>     Number of iterations (default: 1000)\n");
  printf("  --warmup <N>         Number of warmup iterations (default: 10)\n");
  printf("  --output-format <format>  Output format: text, json, csv (default: text)\n");
  printf("  --output-file <path> Output file path (default: stdout for text)\n");
  printf("  --help               Show this help message\n");
  printf("\nNote: Requires exactly 2 MPI ranks. GPUs are mapped as rank->GPU.\n");
}

BenchmarkArgs parseArgs(int argc, char** argv) {
  BenchmarkArgs args;

  static struct option longOptions[] = {
      {"mode", required_argument, 0, 'm'},
      {"iterations", required_argument, 0, 'i'},
      {"warmup", required_argument, 0, 'w'},
      {"output-format", required_argument, 0, 'f'},
      {"output-file", required_argument, 0, 'o'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:i:w:f:o:h", longOptions, nullptr)) !=
         -1) {
    switch (opt) {
    case 'm':
      args.mode = atoi(optarg);
      break;
    case 'i':
      args.iterations = atoi(optarg);
      break;
    case 'w':
      args.warmup = atoi(optarg);
      break;
    case 'f':
      args.outputFormat = optarg;
      if (args.outputFormat != "text" && args.outputFormat != "json" &&
          args.outputFormat != "csv") {
        fprintf(stderr, "ERROR: Invalid output format '%s'\n", optarg);
        printUsage(argv[0]);
        exit(1);
      }
      break;
    case 'o':
      args.outputFile = optarg;
      break;
    case 'h':
      printUsage(argv[0]);
      exit(0);
    default:
      printUsage(argv[0]);
      exit(1);
    }
  }

  return args;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int nRanks, myRank;
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  if (nRanks != 2) {
    if (myRank == 0) {
      fprintf(stderr, "ERROR: This benchmark requires exactly 2 MPI ranks\n");
      fprintf(stderr, "Usage: mpirun -n 2 %s [options]\n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  BenchmarkArgs args = parseArgs(argc, argv);
  args.myRank = myRank;
  args.nRanks = nRanks;
  args.myGpu = myRank;  // Rank 0 -> GPU 0, Rank 1 -> GPU 1
  args.peerGpu = 1 - myRank;  // Rank 0 peers with GPU 1, Rank 1 with GPU 0

  // Validate GPUs
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    if (myRank == 0) {
      printf("SKIP: need >= 2 GPUs, found %d\n", deviceCount);
    }
    MPI_Finalize();
    return 0;
  }

  if (args.myGpu >= deviceCount || args.peerGpu >= deviceCount) {
    if (myRank == 0) {
      fprintf(stderr, "ERROR: Invalid GPU IDs (available: 0-%d)\n",
              deviceCount - 1);
    }
    MPI_Finalize();
    return 1;
  }

  // Set device for this rank
  HIP_CHECK(hipSetDevice(args.myGpu));

  hipDeviceProp_t myProp;
  HIP_CHECK(hipGetDeviceProperties(&myProp, args.myGpu));

  if (myRank == 0) {
    hipDeviceProp_t peerProp;
    HIP_CHECK(hipGetDeviceProperties(&peerProp, args.peerGpu));
    printf("Rank 0 GPU %d: %s\n", args.myGpu, myProp.name);
    printf("Rank 1 GPU %d: %s\n", args.peerGpu, peerProp.name);

    int canAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, args.myGpu, args.peerGpu));
    if (!canAccess) {
      printf("SKIP: GPU %d cannot access GPU %d\n", args.myGpu, args.peerGpu);
      MPI_Finalize();
      return 0;
    }

    printf("Iterations: %d (warmup: %d)\n", args.iterations, args.warmup);
    printf("Transfer sizes: ");
    for (size_t size : args.transferSizes) {
      printf("%zu ", size);
    }
    printf("bytes\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Initialize endpoint
  int rc = sdma_ep::initEndpoint();
  if (rc != 0) {
    fprintf(stderr, "ERROR: initEndpoint failed (%d)\n", rc);
    return 1;
  }

  // Storage for latency statistics
  std::vector<LatencyStats> allStats;

  // Run selected mode
  switch (args.mode) {
  case 1:
    rc = runDeviceInitiated(args, allStats);
    break;
  case 2:
    rc = runHostTriggered(args, allStats);
    break;
  case 3:
    rc = runReverseOffload(args, false, allStats);
    break;
  case 4:
    rc = runReverseOffload(args, true, allStats);
    break;
  case 5:
    rc = runChained(args, allStats);
    break;
  case 6:
    rc = runStreamWaitMemcpy(args, allStats);
    break;
  default:
    fprintf(stderr, "ERROR: Invalid mode %d\n", args.mode);
    printUsage(argv[0]);
    return 1;
  }

  sdma_ep::shutdownEndpoint();

  // Write output file if requested (only rank 0)
  if (myRank == 0 && !args.outputFile.empty()) {
    if (args.outputFormat == "json") {
      writeJsonOutput(args.outputFile, args, allStats);
    } else if (args.outputFormat == "csv") {
      writeCsvOutput(args.outputFile, args, allStats);
    }
  }

  MPI_Finalize();
  return rc;
}
