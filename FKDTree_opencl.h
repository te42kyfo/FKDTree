#ifndef FKDTREE_FKDTREE_OPENCL_H_
#define FKDTREE_FKDTREE_OPENCL_H_

#include <vector>
#include "FKDPoint.h"
#include "FKDTree.h"
#include "ocl.hpp"

template <class T, unsigned int numberOfDimensions>
class FKDTree_OpenCL : public FKDTree<T, numberOfDimensions> {
 public:
  FKDTree_OpenCL(const std::vector<FKDPoint<T, numberOfDimensions>>& points)
      : ocl(1), theNumberOfPoints(points.size()), hdSync(false) {
    h_ids.resize(theNumberOfPoints);
    for (unsigned int d = 0; d < numberOfDimensions; d++) {
      h_dimensions[d].resize(theNumberOfPoints);
    }
    for (unsigned int i = 0; i < theNumberOfPoints; i++) {
      for (unsigned int d = 0; d < numberOfDimensions; d++) {
        h_dimensions[d][i] = points[i][d];
      }
      h_ids[i] = points[i].getId();
    }

    cl_int error;
    d_dimensions = clCreateBuffer(
        ocl.context, CL_MEM_READ_WRITE,
        numberOfDimensions * theNumberOfPoints * sizeof(T), NULL, &error);
    d_A = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                         theNumberOfPoints * sizeof(T), NULL, &error);
    d_B = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                         theNumberOfPoints * sizeof(T), NULL, &error);
    checkOclErrors(error);
    for (unsigned int d = 0; d < numberOfDimensions; d++) {
      checkOclErrors(clEnqueueWriteBuffer(
          ocl.queue, d_dimensions, true, d * theNumberOfPoints * sizeof(T),
          theNumberOfPoints * sizeof(T), h_dimensions[d].data(), 0, NULL,
          NULL));
    }

    d_ids = ocl.createAndUpload(h_ids);
    hdSync = true;

    nth_element_kernel =
        ocl.buildKernel("nth_element.cl", "nth_element", "-D T=unsigned int");
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
    nth_element(d_dimensions, d_dimensions_backbuffer, theNumberOfPoints,
                theNumberOfPoints / 2);
  }

  void nth_element(cl_mem data, cl_mem A, cl_mem B, unsigned int len,
                   unsigned int N) {
    unsigned int blockSize = 64;
    ocl.execute(nth_element_kernel, 1, {blockSize}, {blockSize}, data, A, B,
                len, N);
  }

  OCL ocl;

 private:
  void downloadAllData() {
    for (unsigned int d = 0; d < numberOfDimensions; d++) {
      checkOclErrors(clEnqueueReadBuffer(
          ocl.queue, d_dimensions, true, d * theNumberOfPoints * sizeof(T),
          theNumberOfPoints * sizeof(T), h_dimensions[d].data(), 0, NULL,
          NULL));
    }
    checkOclErrors(clEnqueueReadBuffer(ocl.queue, d_ids, true, 0,
                                       theNumberOfPoints * sizeof(T),
                                       h_ids.data(), 0, NULL, NULL));
    hdSync = true;
  }

  unsigned int theNumberOfPoints;
  cl_mem d_dimensions;
  cl_mem d_A;
  cl_mem d_B;
  cl_mem d_ids;
  cl_kernel nth_element_kernel;

  std::array<std::vector<T>, numberOfDimensions> h_dimensions;
  std::vector<unsigned int> h_ids;
  bool hdSync;
};

#endif /* FKDTREE_FKDTREE_OPENCL_H_ */
