// Template parameters: T, numberOfDimensions

#define FLOOR_LOG2(X) ((unsigned int)(31 - clz(X | 1)))

uint partition_complete_kdtree(uint len) {
  if (len == 1) return 0;
  uint index = 1 << (FLOOR_LOG2(len));

  if ((index / 2) - 1 <= len - index)
    return index - 1;
  else
    return len - index / 2;
}

__kernel void nth_element_small(
    __global uint* groupStarts, __global uint* groupLens, uint groupCount,
    __global T* dimensions, __global T* global_points_src,
    __global T* global_points_dst, __global T* global_A, __global T* global_B,
    __global uint* global_hist, uint dimension, uint totalLen, uint depth) {
  uint const tidx = get_global_id(0);
  uint const gridSize = get_global_size(0);

  for (uint gidx = tidx; gidx < groupCount; gidx += gridSize) {
    uint gindex = (1 << depth) - 1 + gidx;
    uint groupLen = groupLens[gindex];
    uint groupStart = groupStarts[gindex];
    uint group_N = partition_complete_kdtree(groupLen);

    uint nth_element_value;
    if (groupLen == 1) {
      nth_element_value = global_points_src[groupStart];
    }
    if (groupLen == 2) {
      T temp[2];
      temp[0] =
          min(global_points_src[groupStart], global_points_src[groupStart + 1]);
      temp[1] =
          max(global_points_src[groupStart], global_points_src[groupStart + 1]);
      nth_element_value = temp[group_N];
    }
    if (groupLen > 2) {
      for (uint i = 0; i < groupLen; i++) {
        for (uint j = 0; j < groupLen - 1 - i; j++) {
          if (global_points_src[groupStart + j] >
              global_points_src[groupStart + j + 1]) {
            T temp = global_points_src[groupStart + j];
            global_points_src[groupStart + j] =
                global_points_src[groupStart + j + 1];
            global_points_src[groupStart + j + 1] = temp;
          }
        }
        nth_element_value = global_points_src[groupStart + group_N];
      }
    }

    uint buckets[3];

    buckets[0] = 0;
    buckets[1] = 0;
    buckets[2] = 0;

    for (uint row = 0; row < groupLen; row++) {
      uint key =
          (global_points_src[groupStart + row] > nth_element_value) ? 2 : 0;
      key =
          (global_points_src[groupStart + row] == nth_element_value) ? 1 : key;
      buckets[key]++;
    }
    buckets[2] += buckets[1];
    buckets[1] = buckets[0];
    buckets[0] = 0;

    for (uint row = 0; row < groupLen; row++) {
      uint key =
          (global_points_src[groupStart + row] > nth_element_value) ? 2 : 0;
      key =
          (global_points_src[groupStart + row] == nth_element_value) ? 1 : key;
      for (uint d = 0; d < numberOfDimensions + 1; d++) {
        global_points_dst[groupStart + d * totalLen + buckets[key]] =
            global_points_src[groupStart + d * totalLen + row];
      }
      buckets[key]++;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    uint leftChildIndex = (1 << (depth + 1)) - 1 + 2 * gidx;
    uint rightChildIndex = (1 << (depth + 1)) - 1 + 2 * gidx + 1;

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
    for (uint d = 0; d < numberOfDimensions + 1; d++) {
      dimensions[gindex + d * totalLen] =
          global_points_dst[groupStart + d * totalLen + group_N];
    }
  }
}

__kernel void nth_element(__global uint* groupStarts, __global uint* groupLens,
                          uint groupCount, __global T* dimensions,
                          __global T* global_points_src,
                          __global T* global_points_dst, __global T* global_A,
                          __global T* global_B, __global uint* global_hist,
                          uint dimension, uint totalLen, uint depth) {
#define BUCKETCOUNT 16

  uint const tidx = get_local_id(0);
  uint gridSize = get_local_size(0);

  for (uint gid = get_group_id(0); gid < groupCount; gid += get_num_groups(0)) {
    uint gindex = (1 << depth) - 1 + gid;

    if (gindex >= totalLen) continue;
    uint groupLen = groupLens[gindex];

    uint group_N = partition_complete_kdtree(groupLen);

    uint N = group_N;

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
    for (int bucket = 0; bucket < sizeof(T) * 8 / 4; bucket++) {
      if (len == 1) break;
      uint lowBit = 28 - bucket * 4;
      uint highBit = 32 - bucket * 4;
      // if (tidx == 0) printf("\n Bits %d-%d\n", lowBit, highBit);
      for (uint i = 0; i < BUCKETCOUNT; i++) {
        local_histogram[i] = 0;
      }

      uint mask = (1 << (highBit - lowBit)) - 1;

      for (uint row = tidx; row < len; row += gridSize) {
        uint key = (src[row] >> lowBit) & mask;
        local_histogram[key]++;
        //        printf("Row: %u  src[row]: %u   key(src[row]):  %u\n", row,
        //        src[row],
        //       key);
        //       barrier(CLK_GLOBAL_MEM_FENCE);
      }

      for (uint i = 0; i < BUCKETCOUNT; i++) {
        hist[tidx + i * gridSize] = local_histogram[i];
      }

      barrier(CLK_GLOBAL_MEM_FENCE);

      __local int selectedBucket;
      if (tidx == 0) {
        selectedBucket = -1;
        uint sum = 0;
        for (uint i = 0; i < BUCKETCOUNT * gridSize; i++) {
          uint t = hist[i];
          hist[i] = sum;
          if (sum > N && selectedBucket < 0) {
            selectedBucket = (i - 1) / gridSize;
            //            printf("%u>%u at %u = %d\n", sum, N, i,
            //            selectedBucket);
          }
          sum += t;
        }
        if (selectedBucket == -1) selectedBucket = 15;
        //        printf("bucket: %u \t  bounds: %u-%u\n", selectedBucket,
        //       hist[selectedBucket * gridSize],
        //       (selectedBucket == 15) ? len
        //                              : hist[(selectedBucket + 1) *
        //                              gridSize]);
      }

      barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

      for (uint i = 0; i < BUCKETCOUNT; i++) {
        local_histogram[i] = hist[tidx + i * gridSize];
      }

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

    for (uint row = tidx; row < groupLen; row += gridSize) {
      uint key = (points_src[row] > nth_element_value) ? 2 : 0;
      key = (points_src[row] == nth_element_value) ? 1 : key;
      buckets[key]++;
    }

    hist[tidx] = buckets[0];
    hist[tidx + gridSize] = buckets[1];
    hist[tidx + 2 * gridSize] = buckets[2];

    barrier(CLK_GLOBAL_MEM_FENCE);

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

    for (uint row = tidx; row < groupLen; row += gridSize) {
      uint key = (points_src[row] > nth_element_value) ? 2 : 0;
      key = (points_src[row] == nth_element_value) ? 1 : key;
      for (uint d = 0; d < numberOfDimensions + 1; d++) {
        global_points_dst[groupStarts[gindex] + d * totalLen + buckets[key]] =
            global_points_src[groupStarts[gindex] + d * totalLen + row];
      }
      buckets[key]++;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if (tidx == 0) {
      uint leftChildIndex = (1 << (depth + 1)) - 1 + 2 * gid;
      uint rightChildIndex = (1 << (depth + 1)) - 1 + 2 * gid + 1;

      if (leftChildIndex < totalLen) {
        groupStarts[leftChildIndex] = groupStarts[gindex];
        groupLens[leftChildIndex] = group_N;
        // printf("%u %u <- : %u:%u-%u\n", depth, gid, leftChildIndex,
        //       groupStarts[leftChildIndex], groupLens[leftChildIndex]);
      }
      if (rightChildIndex < totalLen) {
        groupStarts[rightChildIndex] = groupStarts[gindex] + group_N + 1;
        groupLens[rightChildIndex] = groupLens[gindex] - group_N - 1;
        //        printf("%u %u  ->: %u:%u-%u\n", depth, gid, rightChildIndex,
        //       groupStarts[rightChildIndex], groupLens[rightChildIndex]);
      }
      for (uint d = 0; d < numberOfDimensions + 1; d++) {
        dimensions[gindex + d * totalLen] =
            global_points_dst[groupStarts[gindex] + d * totalLen + group_N];
      }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  }
}
