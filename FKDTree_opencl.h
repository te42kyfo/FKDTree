#ifndef FKDTREE_FKDTREE_OPENCL_H_
#define FKDTREE_FKDTREE_OPENCL_H_

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

template <class T, unsigned int numberOfDimensions>
class FKDTree_OpenCL : public FKDTree<T, numberOfDimensions> {
 public:
  FKDTree_OpenCL(const std::vector<FKDPoint<T, numberOfDimensions>>& points)
      : ocl(1), numberOfPoints(points.size()), hdSync(false) {
    h_ids.resize(numberOfPoints);
    for (unsigned int d = 0; d < numberOfDimensions; d++) {
      h_dimensions[d].resize(numberOfPoints);
    }
    for (unsigned int i = 0; i < numberOfPoints; i++) {
      for (unsigned int d = 0; d < numberOfDimensions; d++) {
        h_dimensions[d][i] = points[i][d];
      }
      h_ids[i] = points[i].getId();
    }

    std::vector<uint> groupStarts(numberOfPoints);
    std::vector<uint> groupLens(numberOfPoints);

    groupStarts[0] = 0;
    groupLens[0] = numberOfPoints;

    d_groupStarts = ocl.createAndUpload(groupStarts);
    d_groupLens = ocl.createAndUpload(groupLens);

    cl_int error;
    d_points_src = clCreateBuffer(
        ocl.context, CL_MEM_READ_WRITE,
        (numberOfDimensions + 1) * numberOfPoints * sizeof(unsigned int), NULL,
        &error);
    checkOclErrors(error);
    for (uint d = 0; d < numberOfDimensions; d++) {
      checkOclErrors(clEnqueueWriteBuffer(
          ocl.queue, d_points_src, true, d * numberOfPoints * sizeof(T),
          numberOfPoints * sizeof(T), h_dimensions[d].data(), 0, NULL, NULL));
    }
    checkOclErrors(clEnqueueWriteBuffer(
        ocl.queue, d_points_src, true,
        numberOfDimensions * numberOfPoints * sizeof(T),
        numberOfPoints * sizeof(T), h_ids.data(), 0, NULL, NULL));

    d_points_dst = clCreateBuffer(
        ocl.context, CL_MEM_READ_WRITE,
        (numberOfDimensions + 1) * numberOfPoints * sizeof(unsigned int), NULL,
        &error);
    d_dimensions = clCreateBuffer(
        ocl.context, CL_MEM_READ_WRITE,
        (numberOfDimensions + 1) * numberOfPoints * sizeof(unsigned int), NULL,
        &error);
    d_A = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                         numberOfPoints * sizeof(unsigned int), NULL, &error);
    d_B = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                         numberOfPoints * sizeof(unsigned int), NULL, &error);
    d_temp = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                            blockCount * blockSize * 16 * sizeof(unsigned int),
                            NULL, &error);
    checkOclErrors(error);
    hdSync = true;

    nth_element_kernel =
        ocl.buildKernel("nth_element.cl", "nth_element",
                        std::string("-D T=uint -D numberOfDimensions=" +
                                    std::to_string(numberOfDimensions)));
    nth_element_kernel_small =
        ocl.buildKernel("nth_element.cl", "nth_element_small",
                        std::string("-D T=uint -D numberOfDimensions=" +
                                    std::to_string(numberOfDimensions)));
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
      const FKDPoint<T, numberOfDimensions>& minPoint,
      const FKDPoint<T, numberOfDimensions>& maxPoint) const {
    return std::vector<unsigned int>();
  }

  std::vector<std::vector<unsigned int>> search_in_the_box_multiple(
      const std::vector<FKDPoint<T, numberOfDimensions>>& minPoints,
      const std::vector<FKDPoint<T, numberOfDimensions>>& maxPoints) const {
    return std::vector<std::vector<unsigned int>>();
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
    uint maximum_depth =
        ((unsigned int)(31 - __builtin_clz(numberOfPoints | 1)));
    for (uint depth = 0; depth < maximum_depth; depth++) {
      if (depth < 12) {
        ocl.execute(nth_element_kernel, 1, {blockSize * blockCount},
                    {blockSize}, d_groupStarts, d_groupLens, (1 << depth),
                    d_dimensions, d_points_src, d_points_dst, d_A, d_B, d_temp,
                    depth % numberOfDimensions, numberOfPoints, depth);
      } else {
        ocl.execute(nth_element_kernel_small, 1, {blockSize * blockCount},
                    {blockSize}, d_groupStarts, d_groupLens, (1 << depth),
                    d_dimensions, d_points_src, d_points_dst, d_A, d_B, d_temp,
                    depth % numberOfDimensions, numberOfPoints, depth);
      }
      std::swap(d_points_src, d_points_dst);
      ocl.finish();
      double thisDepth = dtime();
      std::cout << std::setw(3) << depth << " " << std::setw(4) << blockSize
                << " " << (thisDepth - lastDepth) * 1000.0 << "\n";
      lastDepth = thisDepth;
    }
    hdSync = false;
  }

  OCL ocl;

 private:
  void downloadAllData() {
    for (unsigned int d = 0; d < numberOfDimensions; d++) {
      checkOclErrors(clEnqueueReadBuffer(
          ocl.queue, d_dimensions, true, d * numberOfPoints * sizeof(T),
          numberOfPoints * sizeof(T), h_dimensions[d].data(), 0, NULL, NULL));
    }
    checkOclErrors(clEnqueueReadBuffer(
        ocl.queue, d_dimensions, true,
        numberOfDimensions * numberOfPoints * sizeof(T),
        numberOfPoints * sizeof(T), h_ids.data(), 0, NULL, NULL));

    hdSync = true;
  }

  unsigned int numberOfPoints;

  cl_kernel nth_element_kernel;
  cl_kernel nth_element_kernel_small;

  cl_mem d_groupStarts;
  cl_mem d_groupLens;

  cl_mem d_points_src;
  cl_mem d_points_dst;
  cl_mem d_dimensions;
  cl_mem d_A;
  cl_mem d_B;
  cl_mem d_temp;

  std::array<std::vector<T>, numberOfDimensions> h_dimensions;
  std::vector<unsigned int> h_ids;
  bool hdSync;

  static const uint blockSize = 64;
  static const uint blockCount = 64;
};

#endif /* FKDTREE_FKDTREE_OPENCL_H_ */
