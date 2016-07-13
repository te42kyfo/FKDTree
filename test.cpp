#include <algorithm>
//#include <boost/compute/algorithm/copy.hpp>
//#include <boost/compute/algorithm/nth_element.hpp>
//#include <boost/compute/container/vector.hpp>
//#include <boost/compute/functional/math.hpp>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include "ocl.hpp"

#include <sys/time.h>
static double dtime() {
  double tseconds = 0;
  struct timeval t;
  gettimeofday(&t, NULL);
  tseconds = (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
  return tseconds;
}

using namespace std;
// namespace compute = boost::compute;

bool test_correct_build(vector<uint>& dimensions, unsigned int index,
                        int dimension, uint len, uint dim) {
  unsigned int leftSonIndexInArray = 2 * index + 1;
  unsigned int rightSonIndexInArray = leftSonIndexInArray + 1;
  if (rightSonIndexInArray >= len && leftSonIndexInArray >= len) {
    return true;
  } else {
    if (leftSonIndexInArray < len) {
      if (dimensions[dimension * len + index] >=
          dimensions[dimension * len + leftSonIndexInArray]) {
        test_correct_build(dimensions, leftSonIndexInArray,
                           (dimension + 1) % dim, len, dim);
      } else
        return false;
    }

    if (rightSonIndexInArray < len) {
      if (dimensions[dimension * len + index] <=
          dimensions[dimension * len + rightSonIndexInArray]) {
        test_correct_build(dimensions, rightSonIndexInArray,
                           (dimension + 1) % dim, len, dim);
      } else
        return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  OCL ocl(0);
  cl_kernel nth_element_kernel = ocl.buildKernel(
      "nth_element.cl", "nth_element", "-D T=uint -D numberOfDimensions=3");
  unsigned int blockCount = 100;
  std::cout << "# BlockCount: " << blockCount << "\n";
  std::cout << "# Size: " << (1u << 18) << "\n";
  std::cout << "depth blocksize time\n";
  for (uint blockSize = 1; blockSize <= 256; blockSize *= 2) {
    for (cl_uint len = (1u << 18); len < (1u << 19); len *= 2) {
      cl_int error;
      vector<unsigned int> host_data(len * 4);

      random_device rd;
      default_random_engine eng(rd());
      uniform_int_distribution<unsigned int> dis;

      for (unsigned int i = 0; i < len * 3; i++) {
        host_data[i] = dis(eng);
      }
      for (unsigned int i = 0; i < len; i++) {
        host_data[3 * len + i] = i;
      }
      if (len < 32) {
        for (uint i = 0; i < len * 4; i++) {
          cout << setw(4) << host_data[i] << " ";
          if (i % len == len - 1) cout << "\n";
        }
      }

      vector<uint> groupStarts(len);
      vector<uint> groupLens(len);

      groupStarts[0] = 0;
      groupLens[0] = len;

      cl_mem d_groupStarts = ocl.createAndUpload(groupStarts);
      cl_mem d_groupLens = ocl.createAndUpload(groupLens);

      cl_mem d_points_src = ocl.createAndUpload(host_data);
      cl_mem d_points_dst =
          clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                         host_data.size() * sizeof(unsigned int), NULL, &error);
      cl_mem d_dimensions =
          clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                         host_data.size() * sizeof(unsigned int), NULL, &error);
      cl_mem d_A = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                                  len * sizeof(unsigned int), NULL, &error);
      cl_mem d_B = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                                  len * sizeof(unsigned int), NULL, &error);
      cl_mem d_temp = clCreateBuffer(
          ocl.context, CL_MEM_READ_WRITE,
          blockCount * blockSize * 16 * sizeof(unsigned int), NULL, &error);

      double t1 = dtime();
      double lastDepth = t1;
      uint maximum_depth = ((unsigned int)(31 - __builtin_clz(len | 1)));
      for (uint depth = 0; depth < maximum_depth; depth++) {
        ocl.execute(nth_element_kernel, 1, {blockSize * blockCount},
                    {blockSize}, d_groupStarts, d_groupLens, (1 << depth),
                    d_dimensions, d_points_src, d_points_dst, d_A, d_B, d_temp,
                    depth % 3, len, depth);
        swap(d_points_src, d_points_dst);
        ocl.finish();
        double thisDepth = dtime();
        cout << setw(3) << depth << " " << setw(4) << blockSize << " "
             << (thisDepth - lastDepth) * 1000.0 << "\n";
        lastDepth = thisDepth;
      }

      double t2 = dtime();
      auto results = ocl.download<unsigned int>(d_dimensions);
      clReleaseMemObject(d_A);
      clReleaseMemObject(d_B);
      clReleaseMemObject(d_points_src);
      clReleaseMemObject(d_dimensions);
      clReleaseMemObject(d_points_dst);
      clReleaseMemObject(d_temp);
      clReleaseMemObject(d_groupStarts);
      clReleaseMemObject(d_groupLens);
      //      std::cout << len << " " << t2 - t1 << "\n";

      if (len < 32) {
        for (uint i = 0; i < len * 4; i++) {
          cout << setw(4) << results[i] << " ";
          if (i % len == len - 1) cout << "\n";
        }
      }
      /*      if (test_correct_build(results, 0, 0, len, 3)) {
        cout << "Correct Build\n";
      } else {
        cout << "Wrong Build\n";
      }
      cout << "\n";*/
    }
  }
  return 0;
}
