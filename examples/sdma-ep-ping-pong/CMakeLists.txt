# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Example: SDMA endpoint ping-pong latency benchmark
#
# Requires two GPUs with XGMI/Infinity Fabric P2P access.
#
# Build against an installed rocm-xio:
#   cmake -S . -B build \
#     -DCMAKE_PREFIX_PATH=<install-prefix>
#   cmake --build build

cmake_minimum_required(VERSION 3.21)
project(sdma-ep-ping-pong LANGUAGES CXX HIP)

find_package(rocm-xio REQUIRED)
find_package(hsa-runtime64 CONFIG REQUIRED)
find_package(MPI REQUIRED)

add_executable(sdma-ep-ping-pong sdma-ep-ping-pong.hip)
target_link_libraries(sdma-ep-ping-pong PRIVATE
  rocm-xio::rocm-xio
  hsa-runtime64::hsa-runtime64
  MPI::MPI_CXX
  pthread)
if(NOT ROCM_XIO_SHARED_LIBS)
  target_link_options(sdma-ep-ping-pong PRIVATE -fgpu-rdc)
endif()
