// Template Parameter: typename T, uint nDimensions, threadsPerQuery, uint
// maxIntermediates, maxResults, blockSize

bool isInBox(T* p, T* minP, T* maxP) {
  bool inBox = true;
  for (uint d = 0; d < nDimensions; d++) {
    inBox = inBox && (p[d] >= minP[d] && p[d] <= maxP[d]);
  }
  return inBox;
}

__kernel void searchInTheBox(__global T* dimensions, __global uint* results,
                             __global T* minPoints, __global T* maxPoints,
                             __global uint* global_A, __global uint* global_B,
                             uint nPoints, uint nQueries) {
  uint const groupId = get_group_id(0) * get_local_size(0) / threadsPerQuery +
                       get_local_id(0) / threadsPerQuery;
  uint const localGroupId = get_local_id(0) / threadsPerQuery;
  uint const groupLane = get_local_id(0) - localGroupId * threadsPerQuery;
  uint const groupCount =
      get_num_groups(0) * get_local_size(0) / threadsPerQuery;
  uint const queriesPerGroup = blockSize / threadsPerQuery;

  __local uint nextLens[blockSize / threadsPerQuery];
  __local uint resultCount[blockSize / threadsPerQuery];

  __global uint* queueSrc = global_A + maxIntermediates * groupId;
  __global uint* queueDst = global_B + maxIntermediates * groupId;

  for (uint loopQueryId = groupId;
       loopQueryId < ((nQueries - 1) / queriesPerGroup + 1) * queriesPerGroup;
       loopQueryId += groupCount) {
    uint queryId = loopQueryId;
    bool masked = false;
    if (loopQueryId >= nQueries) {
      queryId = 0;
      masked = true;
    }

    T minPoint[nDimensions];
    T maxPoint[nDimensions];
    for (uint d = 0; d < nDimensions; d++) {
      minPoint[d] = minPoints[d * nQueries + queryId];
      maxPoint[d] = maxPoints[d * nQueries + queryId];
      //      if (!masked && groupLane == 0)
      //  printf("%d %d: %f %f\n", groupId, queryId, minPoint[d], maxPoint[d]);
    }

    resultCount[localGroupId] = 0;
    nextLens[localGroupId] = 0;
    uint currentLen = 1;
    queueSrc[0] = 0;
    uint currentDim = 0;

    uint maximum_depth = ((unsigned int)(31 - clz(nPoints | 1)));

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (uint depth = 0; depth <= maximum_depth; depth++) {
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      for (uint p = groupLane; p < currentLen; p += threadsPerQuery) {
        T point[nDimensions];
        uint leftChildIndex = queueSrc[p] * 2 + 1;
        uint rightChildIndex = queueSrc[p] * 2 + 2;
        for (uint d = 0; d < nDimensions; d++) {
          point[d] = dimensions[d * nPoints + queueSrc[p]];
          //     if (groupLane == 0 && !masked)
          //  printf("%d %d @ %d: %f\n", groupId, queryId, queueSrc[p], point[d]);
        }
        if (isInBox(point, minPoint, maxPoint)) {
          if (!masked) {
            uint val = atomic_inc(&results[queryId]);
            //printf("%d %d: f: %u %u\n", groupId, queryId, val + 1, queueSrc[p]);
          }
        }
        if (leftChildIndex < nPoints &&
            minPoint[currentDim] <= point[currentDim]) {
          if (!masked) {
            uint oldIndex = atomic_inc(&nextLens[localGroupId]);
            if (oldIndex < maxIntermediates)
              queueDst[oldIndex] = leftChildIndex;
            // printf("%d %d l: %u %u %u\n", groupId, queryId, queueSrc[p],
            //       leftChildIndex, oldIndex);
          }
        }
        if (rightChildIndex < nPoints &&
            maxPoint[currentDim] >= point[currentDim]) {
          if (!masked) {
            uint oldIndex = atomic_inc(&nextLens[localGroupId]);
            if (oldIndex < maxIntermediates)
              queueDst[oldIndex] = rightChildIndex;
            // printf("%d %d r: %u %u %u\n", groupId, queryId, queueSrc[p],
            //       rightChildIndex, oldIndex);
          }
        }
      }

      __global uint* temp = queueSrc;
      queueSrc = queueDst;
      queueDst = temp;
      currentDim = (currentDim + 1) % nDimensions;
      barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
      currentLen = nextLens[localGroupId];
      barrier(CLK_LOCAL_MEM_FENCE);
      if (groupLane == 0) {
        nextLens[localGroupId] = 0;
        // printf("%u: %u %u\n\n", queryId, depth, currentLen);
      }
    }
    // if (groupLane == 0) printf("\n");
  }
}
