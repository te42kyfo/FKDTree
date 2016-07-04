#ifndef FKDTREE_FKDTREE_OPENCL_H_
#define FKDTREE_FKDTREE_OPENCL_H_

#include <vector>
#include "FKDPoint.h"
#include "FKDTree.h"
#include "ocl.hpp"

template <class TYPE, unsigned int numberOfDimensions>
class FKDTree_OpenCL : public FKDTree<TYPE, numberOfDimensions> {
 public:
  FKDTree_OpenCL(const std::vector<FKDPoint<TYPE, numberOfDimensions>>& points)
      : ocl(0) {}

  std::vector<unsigned int> search_in_the_box(
      const FKDPoint<TYPE, numberOfDimensions>& minPoint,
      const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const {
    return std::vector<unsigned int>();
  }

  std::vector<std::vector<unsigned int>> search_in_the_box_multiple(
      const std::vector<FKDPoint<TYPE, numberOfDimensions>>& minPoints,
      const std::vector<FKDPoint<TYPE, numberOfDimensions>>& maxPoints) const {
    return std::vector<std::vector<unsigned int>>();
  }

  std::vector<TYPE> const& getDimensionVector(const int dimension) const {
    return h_dimensions_vec[dimension];
  }
  std::vector<unsigned int> const& getIdVector() const { return h_id_vec; }

  void build() {}

 private:
  
  OCL ocl;
  cl_mem d_dimensions;
  cl_mem d_ids;
  cl_mem d_results;
  cl_kernel search_in_the_box_kernel;
  std::array<std::vector<TYPE>, numberOfDimensions> h_dimensions_vec;
  std::vector<unsigned int> h_id_vec;
};

#endif /* FKDTREE_FKDTREE_OPENCL_H_ */
