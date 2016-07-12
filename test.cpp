#include <algorithm>
//#include <boost/compute/algorithm/copy.hpp>
//#include <boost/compute/algorithm/nth_element.hpp>
//#include <boost/compute/container/vector.hpp>
//#include <boost/compute/functional/math.hpp>
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

int main(int argc, char** argv) {
  /*auto devices = compute::system::devices();
  for (auto& d : devices) {
    cout << d.name() << "\n";
    compute::context context(d);
    compute::command_queue queue(context, d);

    for (unsigned int N = 8; N < (1u << 4); N *= 2) {
      vector<unsigned int> host_data(N);

      for (unsigned int i = 0; i < N; i++) {
        host_data[i] = (i * 12312) % 100;
      }

      compute::vector<unsigned int> device_vector(host_data.size(), context);

      double t0 = dtime();

      nth_element(begin(host_data), begin(host_data) + N / 2, end(host_data));

      double t1 = dtime();
      compute::copy(begin(host_data), end(host_data), device_vector.begin(),
                    queue);

      double t2 = dtime();
      compute::nth_element(begin(device_vector), begin(device_vector) + N / 2,
                           end(device_vector), queue);
      double t3 = dtime();
      compute::copy(device_vector.begin(), device_vector.end(),
                    begin(host_data), queue);

      cout << setprecision(3);
      cout << N << ": " << t1 - t0 << " " << t3 - t2 << "\n";
    }
  }
*/
  OCL ocl(1);
  cl_kernel nth_element_kernel = ocl.buildKernel(
      "nth_element.cl", "nth_element", "-D T=uint -D numberOfDimensions=3");
  unsigned int blockSize = 512;
  for (cl_uint len = 8; len < (1u << 20); len *= 2) {
    cl_int error;
    vector<unsigned int> host_data(len * 4);

    random_device rd;
    default_random_engine eng(rd());
    uniform_int_distribution<unsigned int> dis(0, 1000);

    for (unsigned int i = 0; i < len * 3; i++) {
      host_data[i] = dis(eng);
      if (len < 32) cout << host_data[i] << " ";
    }
    for (unsigned int i = 0; i < len; i++) {
      host_data[3 * len + i] = i;
    }
    cout << "\n";

    vector<uint> groupStarts = {0, len / 2};
    vector<uint> groupLens = {len / 2, len / 2};

    cl_mem d_dimensions_src = ocl.createAndUpload(host_data);
    cl_mem d_dimensions_dst =
        clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                       host_data.size() * sizeof(unsigned int), NULL, &error);
    cl_mem d_A = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                                len * sizeof(unsigned int), NULL, &error);
    cl_mem d_B = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE,
                                len * sizeof(unsigned int), NULL, &error);
    cl_mem d_temp = clCreateBuffer(
        ocl.context, CL_MEM_READ_WRITE,
        groupStarts.size() * blockSize * 16 * sizeof(unsigned int), NULL,
        &error);

    double t1 = dtime();

    groupStarts = {0};
    groupLens = {len};
    cl_mem d_groupStarts = ocl.createAndUpload(groupStarts);
    cl_mem d_groupLens = ocl.createAndUpload(groupLens);
    ocl.execute(nth_element_kernel, 1, {groupStarts.size() * blockSize},
                {blockSize}, d_groupStarts, d_groupLens, 1, d_dimensions_src,
                d_dimensions_dst, d_A, d_B, d_temp, 0, len);

    swap(d_dimensions_src, d_dimensions_dst);
    groupStarts = {0, len / 2};
    groupLens = {len / 2, len - len / 2};
    d_groupStarts = ocl.createAndUpload(groupStarts);
    d_groupLens = ocl.createAndUpload(groupLens);
    ocl.execute(nth_element_kernel, 1, {groupStarts.size() * blockSize},
                {blockSize}, d_groupStarts, d_groupLens, 2, d_dimensions_src,
                d_dimensions_dst, d_A, d_B, d_temp, 1, len);

    ocl.finish();
    double t2 = dtime();
    auto results = ocl.download<unsigned int>(d_dimensions_dst);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_dimensions_src);
    clReleaseMemObject(d_dimensions_dst);
    clReleaseMemObject(d_temp);
    std::cout << len << " " << t2 - t1 << "\n";

    for (uint i = 0; i < len * 4; i++) {
      if (len < 32) cout << results[i] << " ";
    }
    if (len < 32) cout << "\n";

  }
  return 0;
}
