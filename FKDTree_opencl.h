#ifndef FKDTREE_FKDTREE_OPENCL_H_
#define FKDTREE_FKDTREE_OPENCL_H_

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "FKDPoint.h"
#include "FKDTree.h"
#include "ocl.hpp"

#include <sys/time.h>
namespace {
double dtime() {
  double tseconds = 0;
  struct timeval t;
  gettimeofday(&t, NULL);
  tseconds = (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
  return tseconds;
}
}

#define MAX_INTERMEDIATES 8192
#define MAX_RESULTS 1024

template <class T, unsigned int nDimensions>
class FKDTree_OpenCL : public FKDTree<T, nDimensions> {
 public:
  FKDTree_OpenCL(const std::vector<FKDPoint<T, nDimensions>>& points)
      : ocl(1), nPoints(points.size()), hdSync(false) {
    // unpacks data from point data structure to linear vectors
    h_ids.resize(nPoints);
    for (unsigned int d = 0; d < nDimensions; d++) {
      h_dimensions[d].resize(nPoints);
    }
    for (unsigned int i = 0; i < nPoints; i++) {
      for (unsigned int d = 0; d < nDimensions; d++) {
        h_dimensions[d][i] = points[i][d];
      }
      h_ids[i] = points[i].getId();
    }

    // create and initialize all device memory buffers
    std::vector<uint> groupStarts(nPoints);
    std::vector<uint> groupLens(nPoints);

    groupStarts[0] = 0;
    groupLens[0] = nPoints;

    d_groupStarts = ocl.createAndUpload(groupStarts);
    d_groupLens = ocl.createAndUpload(groupLens);

    cl_int error;
    d_points_src = clCreateBuffer(
        ocl.context, CL_MEM_READ_WRITE,
        (nDimensions + 1) * nPoints * sizeof(unsigned int), NULL, &error);
    checkOclErrors(error);
    for (uint d = 0; d < nDimensions; d++) {
      checkOclErrors(clEnqueueWriteBuffer(
          ocl.queue, d_points_src, true, d * nPoints * sizeof(T),
          nPoints * sizeof(T), h_dimensions[d].data(), 0, NULL, NULL));
    }
    checkOclErrors(clEnqueueWriteBuffer(
        ocl.queue, d_points_src, true, nDimensions * nPoints * sizeof(T),
        nPoints * sizeof(T), h_ids.data(), 0, NULL, NULL));

    d_points_dst = clCreateBuffer(
        ocl.context, CL_MEM_READ_WRITE,
        (nDimensions + 1) * nPoints * sizeof(unsigned int), NULL, &error);
    d_dimensions = clCreateBuffer(
        ocl.context, CL_MEM_READ_WRITE,
        (nDimensions + 1) * nPoints * sizeof(unsigned int), NULL, &error);
    d_A = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                         nPoints * sizeof(unsigned int), NULL, &error);
    d_B = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                         nPoints * sizeof(unsigned int), NULL, &error);
    d_temp = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                            blockCount * blockSize * 16 * sizeof(unsigned int),
                            NULL, &error);
    checkOclErrors(error);
    // indicates that host and device buffers are currently identical
    hdSync = true;

    nth_element_kernel = ocl.buildKernel(
        "nth_element.cl", "nth_element",
        std::string("-D T=uint -D nDimensions=" + std::to_string(nDimensions)));
    nth_element_kernel_small = ocl.buildKernel(
        "nth_element.cl", "nth_element_small",
        std::string("-D T=uint -D nDimensions=" + std::to_string(nDimensions)));
    searchInTheBox_kernel_2 = ocl.buildKernel(
        "searchInTheBox2.cl", "searchInTheBox",
        std::string("-D blockSize=" + std::to_string(blockSize) +
                    " -D T=float -D threadsPerQuery=8 -D maxResults=" +
                    std::to_string(MAX_RESULTS) + " -D maxIntermediates=" +
                    std::to_string(MAX_RESULTS) + " -D nDimensions=" +
                    std::to_string(nDimensions)));
  }

  ~FKDTree_OpenCL() {
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_points_src);
    clReleaseMemObject(d_dimensions);
    clReleaseMemObject(d_points_dst);
    clReleaseMemObject(d_temp);
    clReleaseMemObject(d_groupStarts);
    clReleaseMemObject(d_groupLens);
  }

  std::vector<unsigned int> search_in_the_box(
      const FKDPoint<T, nDimensions>& minPoint,
      const FKDPoint<T, nDimensions>& maxPoint) {
    return std::vector<unsigned int>();
  }

  std::vector<unsigned int> search_in_the_box_multiple(
      const std::vector<FKDPoint<T, nDimensions>>& minPoints,
      const std::vector<FKDPoint<T, nDimensions>>& maxPoints) {
    /*    auto gpuData = ocl.download<float>(d_dimensions);
    for (uint d = 0; d < nDimensions; d++) {
      for (uint i = 0; i < nPoints; i++) {
        std::cout <<  gpuData[d*nPoints + i] << " ";
      }
      std::cout << "\n";
      }*/

    std::vector<uint> h_results(minPoints.size());

    std::vector<T> h_minPoints(minPoints.size() * nDimensions);
    std::vector<T> h_maxPoints(minPoints.size() * nDimensions);

    for (unsigned int i = 0; i < minPoints.size(); i++) {
      for (unsigned int d = 0; d < nDimensions; d++) {
        h_minPoints[d * minPoints.size() + i] = minPoints[i][d];
        h_maxPoints[d * minPoints.size() + i] = maxPoints[i][d];
      }
    }
    cl_mem d_minPoints = ocl.createAndUpload(h_minPoints);
    cl_mem d_maxPoints = ocl.createAndUpload(h_maxPoints);

    cl_int error;
    cl_mem d_A = clCreateBuffer(
        ocl.context, CL_MEM_READ_WRITE,
        MAX_INTERMEDIATES * blockSize * blockCount * sizeof(unsigned int), NULL,
        &error);
    cl_mem d_B = clCreateBuffer(
        ocl.context, CL_MEM_READ_WRITE,
        MAX_INTERMEDIATES * blockSize * blockCount * sizeof(unsigned int), NULL,
        &error);

    cl_mem d_results = ocl.createAndUpload<uint>(h_results);

    ocl.execute(searchInTheBox_kernel_2, 1, {blockSize * blockCount},
                {blockSize}, d_dimensions, d_results, d_minPoints, d_maxPoints,
                d_A, d_B, nPoints, (uint)minPoints.size());

    checkOclErrors(clEnqueueReadBuffer(ocl.queue, d_results, true, 0,
                                       minPoints.size() * sizeof(uint),
                                       h_results.data(), 0, NULL, NULL));

    clReleaseMemObject(d_minPoints);
    clReleaseMemObject(d_maxPoints);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    return h_results;
  }

  std::vector<T> const& getDimensionVector(const int dimension) {
    if (!hdSync) {
      downloadAllData();
    }
    return h_dimensions[dimension];
  }

  std::vector<unsigned int> const& getIdVector() {
    if (!hdSync) {
      downloadAllData();
    }
    return h_ids;
  }

  void build() {
    double lastDepth = dtime();
    uint maximum_depth = ((unsigned int)(31 - __builtin_clz(nPoints | 1)));

    // auto gpudata = ocl.download<float>(d_points_src);
    // for (auto f : gpudata) std::cout << "gpu data: " << f << "\n";

    // Two different kernels for the different stages
    for (uint depth = 0; depth <= maximum_depth; depth++) {
      bool smallKernelUsed = false;
      if (nPoints / (1 << depth) > 200) {
        ocl.execute(nth_element_kernel, 1, {blockSize * blockCount},
                    {blockSize}, d_groupStarts, d_groupLens, (1 << depth),
                    d_dimensions, d_points_src, d_points_dst, d_A, d_B, d_temp,
                    depth % nDimensions, nPoints, depth);
        smallKernelUsed = false;
        std::swap(d_points_src, d_points_dst);
      } else {
        ocl.execute(nth_element_kernel_small, 1, {blockSize * blockCount},
                    {blockSize}, d_groupStarts, d_groupLens,
                    std::min(((uint)1 << depth), nPoints - (1 << depth) + 1),
                    d_dimensions, d_points_src, d_points_dst,
                    depth % nDimensions, nPoints, depth);
        smallKernelUsed = true;
      }

      ocl.finish();
      double thisDepth = dtime();
      if (smallKernelUsed)
        std::cout << "s";
      else
        std::cout << " ";
      std::cout << std::setw(3) << depth << " " << std::setw(4) << " "
                << (thisDepth - lastDepth) * 1000.0 << "\n";
      lastDepth = thisDepth;
    }
    // indicate that host and device memory is now out of sync
    hdSync = false;
  }

  OCL ocl;

 private:
  void downloadAllData() {
    for (unsigned int d = 0; d < nDimensions; d++) {
      checkOclErrors(clEnqueueReadBuffer(
          ocl.queue, d_dimensions, true, d * nPoints * sizeof(T),
          nPoints * sizeof(T), h_dimensions[d].data(), 0, NULL, NULL));
    }
    checkOclErrors(clEnqueueReadBuffer(
        ocl.queue, d_dimensions, true, nDimensions * nPoints * sizeof(T),
        nPoints * sizeof(T), h_ids.data(), 0, NULL, NULL));

    hdSync = true;
  }

  unsigned int nPoints;

  cl_kernel nth_element_kernel;
  cl_kernel nth_element_kernel_small;
  cl_kernel searchInTheBox_kernel_2;

  cl_mem d_groupStarts;
  cl_mem d_groupLens;

  cl_mem d_points_src;
  cl_mem d_points_dst;
  cl_mem d_dimensions;
  cl_mem d_A;
  cl_mem d_B;
  cl_mem d_temp;

  std::array<std::vector<T>, nDimensions> h_dimensions;
  std::vector<unsigned int> h_ids;
  bool hdSync;

  static const uint blockSize = 8;
  static const uint blockCount = 8;
};

#endif /* FKDTREE_FKDTREE_OPENCL_H_ */
