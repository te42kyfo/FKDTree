// Template parameters: T, numberOfDimensions

__kernel void nth_element(__global T* data_src, __global T* data_dst,
                          __global T* A, __global T* B, __global uint* hist,
                          uint totalLen, uint N) {
  uint const tidx = get_global_id(0);
  uint gridSize = get_global_size(0);

// if (tidx == 0) printf("N=%u\n", N);

#define BUCKETCOUNT 16

  __global T* src = data_src;
  __global T* dst = A;
  uint len = totalLen;
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
      // printf("Row: %u  src[row]: %u   key(src[row]):  %u\n", row, src[row],
      //       key);
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
          // pprintf("%u>%u at %u = %d\n", sum, N, i, selectedBucket);
        }
        sum += t;
      }
      if (selectedBucket == -1) selectedBucket = 15;
      // printf(
      //    "Selected Buffer bounds: %u-%u\n", hist[selectedBucket * gridSize],
      //    (selectedBucket == 15) ? len : hist[(selectedBucket + 1) *
      //    gridSize]);
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    for (uint i = 0; i < BUCKETCOUNT; i++) {
      local_histogram[i] = hist[tidx + i * gridSize];
    }

    for (uint row = tidx; row < len; row += gridSize) {
      uint key = (src[row] >> lowBit) & mask;
      if (key == selectedBucket) {
        uint dstIndex = local_histogram[key] - hist[selectedBucket * gridSize];
        //        printf("Copy %u at %u to %u\n", src[row], row, dstIndex);
        dst[dstIndex] = src[row];
        local_histogram[key]++;
      }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    __global T* temp = src;
    src = dst;
    dst = temp;
    if (dst == data_src) dst = B;

    if (selectedBucket == 15) {
      len -= hist[selectedBucket * gridSize];
    } else {
      len = hist[(selectedBucket + 1) * gridSize] -
            hist[selectedBucket * gridSize];
    }

    N -= hist[selectedBucket * gridSize];
    //    if (tidx == 0)
    //  printf("Next Len: %u,\t N: %u, bucket: %d\n", len, N, selectedBucket);

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  }

  uint nth_element_value = src[N];

  uint buckets[3];
  buckets[0] = 0;
  buckets[1] = 0;
  buckets[2] = 0;

  for (uint row = tidx; row < totalLen; row += gridSize) {
    uint key = (data_src[row] > nth_element_value) ? 2 : 0;
    key = (data_src[row] == nth_element_value) ? 1 : key;
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

  for (uint row = tidx; row < totalLen; row += gridSize) {
    uint key = (data_src[row] > nth_element_value) ? 2 : 0;
    key = (data_src[row] == nth_element_value) ? 1 : key;
    for (uint d = 0; d < numberOfDimensions; d++) {
      data_dst[d * totalLen + buckets[key]] = data_src[d * totalLen + row];
    }
    buckets[key]++;
  }
}
