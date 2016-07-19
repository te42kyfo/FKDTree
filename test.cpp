#include <algorithm>

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include "FKDTree_opencl.h"

using namespace std;

int main(int argc, char** argv) {
  for (cl_uint len = (1u << 18); len < (1u << 19); len *= 2) {
    vector<FKDPoint<float, 3>> host_data(len);

    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<float> dis;

    for (unsigned int i = 0; i < len; i++) {
      host_data[i] = {dis(eng), dis(eng), dis(eng), i};
    }

    FKDTree_OpenCL<float, 3> clKdtree(host_data);

    double t1 = dtime();
    clKdtree.build();
    double t2 = dtime();

    std::cout << len << " " << t2 - t1 << "\n";

    if (clKdtree.test_correct_build()) {
      cout << "Correct Build\n";
    } else {
      cout << "Wrong Build\n";
    }
    cout << "\n";
  }

  return 0;
}
