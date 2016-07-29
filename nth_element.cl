// Template parameters: T, nDimensions

#define FLOOR_LOG2(X) ((unsigned int)(31 - clz(X | 1)))

// calculates the split position to have a complete tree on the left
uint partition_complete_kdtree(uint len) {
  if (len == 1) return 0;
  uint index = 1 << (FLOOR_LOG2(len));

  if ((index / 2) - 1 <= len - index)
    return index - 1;
  else
    return len - index / 2;
}

void swapDims(uint i, uint j, uint totalLen, __global T* buffer) {
  for (uint d = 0; d < nDimensions + 1; d++) {
    T temp = buffer[i + totalLen * d];
    buffer[i + totalLen * d] = buffer[j + totalLen * d];
    buffer[j + totalLen * d] = temp;
  }
}

// small partition size kernel. One thread per partition, uses bubblesort
// to find the parition value.
__kernel void nth_element_small(__global uint* groupStarts,
                                __global uint* groupLens, uint groupCount,
                                __global T* dimensions,
                                __global T* global_points_src,
                                __global T* global_points_dst, uint dimension,
                                uint totalLen, uint depth) {
  uint const tidx = get_global_id(0);
  uint const gridSize = get_global_size(0);

  if (groupCount == 0) return;

  // if (tidx == 0) printf("groupCount: %u\n", groupCount);
  // Grid stride loop over partitions
  for (uint gidx = tidx; gidx < groupCount; gidx += gridSize) {
    uint gindex = (1 << depth) - 1 + gidx;

    // printf("%u %u %u\n", gindex, depth, gidx);
    uint groupLen = groupLens[gindex];
    uint groupStart = groupStarts[gindex];
    uint group_N = partition_complete_kdtree(groupLen);

    uint nth_element_value = 0;
    uint buckets[3];

    // simple bubblesort
    for (uint i = 0; i < groupLen; i++) {
      for (uint j = 0; j < groupLen - 1 - i; j++) {
        if (global_points_src[groupStart + dimension * totalLen + j] >
            global_points_src[groupStart + dimension * totalLen + j + 1]) {
          swapDims(groupStart + j, groupStart + j + 1, totalLen,
                   global_points_src);
        }
      }
      nth_element_value =
          global_points_src[groupStart + dimension * totalLen + group_N];
    }

    // printf("%dth: %f\n", group_N, nth_element_value);

    // count values below, equal, above nth_element_value
    /*    buckets[0] = 0;
    buckets[1] = 0;
    buckets[2] = 0;

    for (uint row = 0; row < groupLen; row++) {
      T val = global_points_src[groupStart + dimension * totalLen + row];
      uint key = ((val > nth_element_value) ? 2 : 0);
      key = ((val == nth_element_value) ? 1 : key);
      buckets[key]++;
    }
    buckets[2] = buckets[1] + buckets[0];
    buckets[1] = buckets[0];
    buckets[0] = 0;

    // rearange values from points_src to the right partitions in points_dst
    for (uint row = 0; row < groupLen; row++) {
      T val = global_points_src[groupStart + dimension * totalLen + row];
      uint key = ((val > nth_element_value) ? 2 : 0);
      key = ((val == nth_element_value) ? 1 : key);

      for (uint d = 0; d < nDimensions + 1; d++) {
        global_points_dst[groupStart + d * totalLen + buckets[key]] =
            global_points_src[groupStart + d * totalLen + row];
      }
      buckets[key]++;
    }
    */
    uint leftChildIndex = (1 << (depth + 1)) - 1 + 2 * gidx;
    uint rightChildIndex = (1 << (depth + 1)) - 1 + 2 * gidx + 1;

    // write new groupStarts/groupLens for the next depth
    if (leftChildIndex < totalLen) {
      groupStarts[leftChildIndex] = groupStart;
      groupLens[leftChildIndex] = group_N;
      // printf("%u %u <- : %u:%u-%u\n", depth, gid, leftChildIndex,
      //       groupStarts[leftChildIndex], groupLens[leftChildIndex]);
    }
    if (rightChildIndex < totalLen) {
      groupStarts[rightChildIndex] = groupStart + group_N + 1;
      groupLens[rightChildIndex] = groupLen - group_N - 1;
      //        printf("%u %u  ->: %u:%u-%u\n", depth, gid, rightChildIndex,
      //       groupStarts[rightChildIndex], groupLens[rightChildIndex]);
    }
    for (uint d = 0; d < nDimensions + 1; d++) {
      dimensions[gindex + d * totalLen] =
          global_points_src[groupStart + d * totalLen + group_N];
    }
  }
}

// partition kernel for the earlier depth stages. One work group per partition
__kernel void nth_element(__global uint* groupStarts, __global uint* groupLens,
                          uint groupCount, __global T* dimensions,
                          __global T* global_points_src,
                          __global T* global_points_dst, __global T* global_A,
                          __global T* global_B, __global uint* global_hist,
                          uint dimension, uint totalLen, uint depth) {
// Radixsort with 4 bits/16 buckets
#define BUCKETCOUNT 16

  uint const tidx = get_local_id(0);
  uint gridSize = get_local_size(0);

  // grid stride loop over partitions
  for (uint gid = get_group_id(0); gid < groupCount; gid += get_num_groups(0)) {
    uint gindex = (1 << depth) - 1 + gid;

    if (gindex >= totalLen) continue;
    uint groupLen = groupLens[gindex];

    uint group_N = partition_complete_kdtree(groupLen);

    uint N = group_N;

    // calculate the right offsets in global arrays
    if (groupLen == 0) continue;
    __global T* points_src =
        global_points_src + dimension * totalLen + groupStarts[gindex];
    __global T* points_dst =
        global_points_dst + dimension * totalLen + groupStarts[gindex];
    __global T* A = global_A + groupStarts[gindex];
    __global T* B = global_B + groupStarts[gindex];
    __global uint* hist =
        global_hist + get_group_id(0) * BUCKETCOUNT * gridSize;

    __global T* src = points_src;
    __global T* dst = A;

    //    printf("%u %u %u  %u-%u\n", depth, gid, gindex, groupStarts[gindex],
    //       groupLens[gindex]);
    barrier(CLK_GLOBAL_MEM_FENCE);

    uint len = groupLen;
    uint local_histogram[BUCKETCOUNT];
    // Radix Sort, progress from highest to lowest bits, four at a time
    for (int bucket = 0; bucket < sizeof(T) * 8 / 4; bucket++) {
      if (len == 1) break;
      uint lowBit = sizeof(T) * 8 - 4 - bucket * 4;
      uint highBit = sizeof(T) * 8 - bucket * 4;
      // if (tidx == 0) printf("\n Bits %d-%d\n", lowBit, highBit);
      for (uint i = 0; i < BUCKETCOUNT; i++) {
        local_histogram[i] = 0;
      }

      uint mask = (1 << (highBit - lowBit)) - 1;

      // gridstride loop over elements, create thread local histograms
      for (uint row = tidx; row < len; row += gridSize) {
        uint key = (src[row] >> lowBit) & mask;
        local_histogram[key]++;
      }

      for (uint i = 0; i < BUCKETCOUNT; i++) {
        hist[tidx + i * gridSize] = local_histogram[i];
      }

      barrier(CLK_GLOBAL_MEM_FENCE);

      // Scan over interleaved local histograms
      // naively implemented as a sequential sum by thread 0
      // could be improved by parallel scan, OpenCL 2.0 bultin
      // kernels, or CUB workgroup kernels
      // Also finds the partition that contains the nth_element
      __local int selectedBucket;
      if (tidx == 0) {
        selectedBucket = -1;
        uint sum = 0;
        for (uint i = 0; i < BUCKETCOUNT * gridSize; i++) {
          uint t = hist[i];
          hist[i] = sum;
          if (sum > N && selectedBucket < 0) {
            selectedBucket = (i - 1) / gridSize;
          }
          sum += t;
        }
        if (selectedBucket == -1) selectedBucket = 15;
      }

      barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

      for (uint i = 0; i < BUCKETCOUNT; i++) {
        local_histogram[i] = hist[tidx + i * gridSize];
      }

      // copy the partition that contatins the nth_element to another buffer
      for (uint row = tidx; row < len; row += gridSize) {
        uint key = (src[row] >> lowBit) & mask;
        if (key == selectedBucket) {
          uint dstIndex =
              local_histogram[key] - hist[selectedBucket * gridSize];
          //        printf("Copy %u at %u to %u\n", src[row], row, dstIndex);
          dst[dstIndex] = src[row];
          local_histogram[key]++;
        }
      }

      barrier(CLK_GLOBAL_MEM_FENCE);

      // swap front/back buffer
      __global T* temp = src;
      src = dst;
      dst = temp;
      if (dst == points_src) dst = B;

      if (selectedBucket == 15) {
        len -= hist[selectedBucket * gridSize];
      } else {
        len = hist[(selectedBucket + 1) * gridSize] -
              hist[selectedBucket * gridSize];
      }

      N -= hist[selectedBucket * gridSize];
      //    if (tidx == 0)
      //  printf("Next Len: %u,\t N: %u, bucket: %d\n", len, N,
      //  selectedBucket);

      barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }

    uint nth_element_value = src[N];

    uint buckets[3];
    buckets[0] = 0;
    buckets[1] = 0;
    buckets[2] = 0;
    // buckets with below/equal/above nth_element_value
    for (uint row = tidx; row < groupLen; row += gridSize) {
      uint key = (points_src[row] > nth_element_value) ? 2 : 0;
      key = (points_src[row] == nth_element_value) ? 1 : key;
      buckets[key]++;
    }

    hist[tidx] = buckets[0];
    hist[tidx + gridSize] = buckets[1];
    hist[tidx + 2 * gridSize] = buckets[2];

    barrier(CLK_GLOBAL_MEM_FENCE);
    // scan over interleaved local hisograms by thread 0
    if (tidx == 0) {
      uint sum = 0;
      for (uint i = 0; i < 3 * gridSize; i++) {
        uint t = hist[i];
        // if (i % gridSize == 0) printf("\n");
        hist[i] = sum;
        //      printf("%u\n", sum);
        sum += t;
      }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    buckets[0] = hist[tidx];
    buckets[1] = hist[tidx + gridSize];
    buckets[2] = hist[tidx + 2 * gridSize];

    // Shuffle values into the right partitions
    for (uint row = tidx; row < groupLen; row += gridSize) {
      uint key = (points_src[row] > nth_element_value) ? 2 : 0;
      key = (points_src[row] == nth_element_value) ? 1 : key;
      for (uint d = 0; d < nDimensions + 1; d++) {
        global_points_dst[groupStarts[gindex] + d * totalLen + buckets[key]] =
            global_points_src[groupStarts[gindex] + d * totalLen + row];
      }
      buckets[key]++;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // set up groupStarts/groupLens for the next depth
    if (tidx == 0) {
      uint leftChildIndex = (1 << (depth + 1)) - 1 + 2 * gid;
      uint rightChildIndex = (1 << (depth + 1)) - 1 + 2 * gid + 1;

      if (leftChildIndex < totalLen) {
        groupStarts[leftChildIndex] = groupStarts[gindex];
        groupLens[leftChildIndex] = group_N;
      }
      if (rightChildIndex < totalLen) {
        groupStarts[rightChildIndex] = groupStarts[gindex] + group_N + 1;
        groupLens[rightChildIndex] = groupLens[gindex] - group_N - 1;
      }
      for (uint d = 0; d < nDimensions + 1; d++) {
        dimensions[gindex + d * totalLen] =
            global_points_dst[groupStarts[gindex] + d * totalLen + group_N];
      }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  }
}
