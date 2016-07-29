#include <algorithm>

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include "FKDTree_cpu.h"
#include "FKDTree_opencl.h"

using namespace std;

int main(int argc, char** argv) {
  const uint boxCount = 1024;
  for (cl_uint len = (1u << 0); len < (1u << 24); len *= 1.9 + 1) {
    vector<FKDPoint<float, 3>> host_data(len);
    vector<FKDPoint<float, 3>> minPoints(boxCount);
    vector<FKDPoint<float, 3>> maxPoints(boxCount);

    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<float> dis;

    for (unsigned int i = 0; i < len; i++) {
      host_data[i] = {i, i * 2, i * 3, i};
      // cout << "p: " << host_data[i][0] << " " << host_data[i][1] << " "
      //     << host_data[i][2] << "\n";
    }
    for (uint i = 0; i < boxCount; i++) {
      float x1 = dis(eng);
      float x2 = dis(eng);
      float y1 = dis(eng);
      float y2 = dis(eng);
      float z1 = dis(eng);
      float z2 = dis(eng);
      minPoints[i] = {min(x1, x2), min(y1, y2), min(z1, z2), i};
      maxPoints[i] = {max(x1, x2), max(y1, y2), max(z1, z2), i};
    }

    FKDTree_OpenCL<float, 3> clKdtree(host_data);
    FKDTree_CPU<float, 3> kdtree(host_data);

    double t1 = dtime();
    clKdtree.build();
    double t2 = dtime();
    kdtree.build();
    double t3 = dtime();

    bool identical = true;
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

    //    auto results = clKdtree.search_in_the_box_multiple(minPoints,
    //    maxPoints);

    cout << "\n";
  }

  return 0;
}
