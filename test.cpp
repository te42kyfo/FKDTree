#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include "FKDTree_cpu.h"
#include "FKDTree_opencl.h"
#include "dtime.hpp"

using namespace std;

int main(int argc, char** argv) {

  for (cl_uint len = (1u << 14); len < (1u << 19); len = len * 1.5 + 1) {
    uint boxCount = len;
    vector<FKDPoint<float, 3>> host_data(len);
    vector<FKDPoint<float, 3>> minPoints(boxCount);
    vector<FKDPoint<float, 3>> maxPoints(boxCount);

    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<float> dis(0.0f, 10.1f);

    for (unsigned int i = 0; i < len; i++) {
      host_data[i] = {dis(eng), dis(eng), dis(eng), i};
      // cout << "p: " << host_data[i][0] << " " << host_data[i][1] << " "
      //     << host_data[i][2] << "\n";
    }
    for (uint i = 0; i < boxCount; i++) {
      minPoints[i] = {host_data[i][0] - 0.51f, host_data[i][1] - 0.51f,
                      host_data[i][2] - 0.51f, i};
      maxPoints[i] = {host_data[i][0] + 0.51f, host_data[i][1] + 0.51f,
                      host_data[i][2] + 0.51f, i};
    }

    FKDTree_OpenCL<float, 3> clKdtree(host_data);
    FKDTree_CPU<float, 3> kdtree(host_data);

    double t1 = dtime();
    clKdtree.build();
    double t2 = dtime();
    kdtree.build();
    double t3 = dtime();

    cout << setprecision(3);
    cout << "Points: " << len << "  OpenCL build:" << t2 - t1
         << "    CPU build: " << t3 - t2 << " ";

    /*bool identical = true;
    for (uint d = 0; d < 3; d++) {
      auto dataCL = clKdtree.getDimensionVector(d);
      auto dataCPU = kdtree.getDimensionVector(d);
      for (uint i = 0; i < len; i++) {
        if (dataCL[i] != dataCPU[i]) {
          std::cout << i << " " << d << ": " << dataCL[i] << " - " << dataCPU[i]
                    << "\n";
          identical = false;
        }
      }
      }
    std::cout << len << " " << t2 - t1 << " " << t3 - t2;
    if (identical)
      std::cout << "\t identical\n";
    else
      cout << "\t Mismatch\n";
    */

    auto results = clKdtree.search_in_the_box_multiple(minPoints, maxPoints);
    results = clKdtree.search_in_the_box_multiple_old(minPoints, maxPoints);

    double branchless_start = dtime();
#pragma omp parallel for
    for (uint i = 0; i < len; i++) {
      std::vector<unsigned int> foundPoints =
          kdtree.search_in_the_box_branchless(minPoints[i], maxPoints[i]);
    }
    double branchless_end = dtime();
    std::cout << "   CPU branchless: " << (branchless_end - branchless_start)
              << " ";

    double BFS_start = dtime();
#pragma omp parallel for
    for (uint i = 0; i < len; i++) {
      std::vector<unsigned int> foundPoints =
          kdtree.search_in_the_box_BFS(minPoints[i], maxPoints[i]);
    }
    double BFS_end = dtime();
    std::cout << "   CPU BFS: " << (BFS_end - BFS_start) << " ";

    cout << "\n";
  }

  return 0;
}
