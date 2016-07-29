#include <string.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <thread>
#include "tbb/tbb.h"
#ifdef __USE_OPENCL__
#include <CL/cl.h>
#include <fstream>
#include "FKDTree_opencl.h"
#include "cl_helper.h"
#endif

#ifdef __USE_CUDA__
#include "cuda.h"
#include "cuda_runtime.h"
#endif
#include <stdlib.h>
#include <sys/time.h>
#include "FKDPoint.h"
#include "FKDTree.h"
#include "FKDTree_cpu.h"
#include "KDTreeLinkerAlgoT.h"

#ifdef __USE_CUDA__
void CUDAKernelWrapper(unsigned int nPoints, float* h_dim, unsigned int* h_ids,
                       unsigned int* h_results);
#endif

#ifndef __USE_CUDA__
typedef struct float4 {
  float x;
  float y;
  float z;
  float w;
} float4;
#endif

static void show_usage(std::string name) {
  std::cerr << "\nUsage: " << name << " <option(s)>"
            << " Options:\n"
            << "\t-h,--help\t\tShow this help message\n"
            << "\t-n <number of points>\tSpecify the number of points to use "
               "for the kdtree [default 100000]\n"
            << "\t-t \tRun the validity tests [default disabled]\n"
            << "\t-i \tNumber of iterations to run [default 1]\n"
            << "\t-s \tRun the sequential algo\n"
            << "\t-c \tRun the vanilla cmssw algo\n"
            << "\t-f \tRun FKDtree algo\n"
            << "\t-a \tRun all the algos\n"
            << "\t-r \tRun FKDtree recursive search algo\n"

            << "\t-b \tRun branchless FKDtree algo\n"
            << "\t-bfs \tRun branchless FKDtree BFS algo\n"

            << "\t-p <number of threads>\tSpecify the number of tbb parallel "
               "threads to use [default 1]\n"
#ifdef __USE_OPENCL__
            << "\t-ocl \tRun OpenCL search algo\n"
#endif
#ifdef __USE_CUDA__
            << "\t-cuda \tRunr CUDA search algo\n"
#endif
            << std::endl;
}
int main(int argc, char* argv[]) {
  if (argc < 2) {
    show_usage(argv[0]);
    return 1;
  }

  unsigned int numberOfIterations = 1;
  int nPoints = 100000;
  unsigned int numberOfThreads = 1;
  bool runTheTests = false;
  bool runSequential = false;
  bool runFKDTree = false;
  bool runOldKDTree = false;
  bool runOpenCL = false;
  bool runCuda = false;
  bool runBranchless = false;
  bool runRecursive = false;
  bool runBFS = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-h") || (arg == "--help")) {
      show_usage(argv[0]);
      return 0;
    }

    else if (arg == "-n") {
      if (i + 1 < argc)  // Make sure we aren't at the end of argv!
      {
        i++;
        std::istringstream ss(argv[i]);
        if (!(ss >> nPoints)) {
          std::cerr << "Invalid number " << argv[i] << '\n';
          exit(1);
        }
      }
    } else if (arg == "-i") {
      if (i + 1 < argc)  // Make sure we aren't at the end of argv!
      {
        i++;
        std::istringstream ss(argv[i]);
        if (!(ss >> numberOfIterations)) {
          std::cerr << "Invalid number " << argv[i] << '\n';
          exit(1);
        }
      }
    } else if (arg == "-t") {
      runTheTests = true;
    } else if (arg == "-s") {
      runSequential = true;
    } else if (arg == "-f") {
      runFKDTree = true;
    } else if (arg == "-c") {
      runOldKDTree = true;
    } else if (arg == "-a") {
      runOldKDTree = true;
      runFKDTree = true;
      runSequential = true;
      runRecursive = true;
      runBFS = true;
      runBranchless = true;

    } else if (arg == "-b") {
      runBranchless = true;
      runFKDTree = true;

    } else if (arg == "-bfs") {
      runFKDTree = true;
      runBFS = true;

    } else if (arg == "-r") {
      runRecursive = true;
      runFKDTree = true;

    } else if (arg == "-ocl") {
      runOpenCL = true;
    } else if (arg == "-p") {
      if (i + 1 < argc)  // Make sure we aren't at the end of argv!
      {
        i++;
        std::istringstream ss(argv[i]);
        if (!(ss >> numberOfThreads)) {
          std::cerr << "Invalid number of threads " << argv[i] << '\n';

          exit(1);
        }
      }
    } else if (arg == "-cuda") {
      runFKDTree = true;
      runCuda = true;
    }
  }
  tbb::task_scheduler_init init(numberOfThreads);

  std::vector<FKDPoint<float, 3>> points;
  std::vector<FKDPoint<float, 3>> minPoints;
  std::vector<FKDPoint<float, 3>> maxPoints;

  float range_x = 0.51;
  float range_y = 0.51;
  float range_z = 0.51;

  //	FKDPoint<float, 3> minPoint(0, 1, 8);
  //	FKDPoint<float, 3> maxPoint(0.4, 1.2, 8.3);

  std::default_random_engine rd_gen;
  std::uniform_real_distribution<float> udis(0.0f, 10.1f);

  for (int i = 0; i < nPoints; ++i) {
    float x = udis(rd_gen);
    float y = udis(rd_gen);
    float z = udis(rd_gen);
    points.push_back(make_FKDPoint(x, y, z, i));
    FKDPoint<float, 3> m(x - range_x, y - range_y, z - range_z, 0);
    minPoints.push_back(m);
    FKDPoint<float, 3> M(x + range_x, y + range_y, z + range_z, 0);
    maxPoints.push_back(M);
  }
  // needed by the vanilla algo
  float4* cmssw_points = new float4[nPoints];
  for (int j = 0; j < nPoints; j++) {
    cmssw_points[j].x = points[j][0];
    cmssw_points[j].y = points[j][1];
    cmssw_points[j].z = points[j][2];
    cmssw_points[j].w = j;  // Use this to save the index
  }

  std::cout << "Cloud of points generated.\n" << std::endl;

  std::atomic<unsigned int> pointsFound(0);

  std::chrono::steady_clock::time_point start_building =
      std::chrono::steady_clock::now();
  FKDTree_CPU<float, 3> kdtree(points);

  kdtree.build();
  std::chrono::steady_clock::time_point end_building =
      std::chrono::steady_clock::now();
  std::cout << "building FKDTree with " << nPoints << " points took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end_building - start_building)
                   .count()
            << "ms" << std::endl;
  if (runTheTests) {
    if (kdtree.test_correct_build())
      std::cout << "FKDTree built correctly" << std::endl;
    else
      std::cout << "FKDTree wrong" << std::endl;
  }

  if (runOpenCL) {
#ifdef __USE_OPENCL__

    std::unique_ptr<FKDTree<float, 3>> clKdtree(
        static_cast<FKDTree<float, 3>*>(new FKDTree_OpenCL<float, 3>(points)));

    clKdtree->build();

    if (runTheTests) {
      if (clKdtree->test_correct_build())
        std::cout << "OpenCL FKDTree built correctly" << std::endl;
      else
        std::cout << "OpenCL FKDTree wrong" << std::endl;

      tbb::tick_count start_searching = tbb::tick_count::now();
      auto partial_results =
          clKdtree->search_in_the_box_multiple(minPoints, maxPoints);
      tbb::tick_count end_searching = tbb::tick_count::now();
      pointsFound =
          std::accumulate(partial_results.begin(), partial_results.end(), 0);
      std::cout << "searching points using OpenCL FKDTree took "
                << (end_searching - start_searching).seconds() * 1e3 << "ms\n"
                << " found points: " << pointsFound
                << "\n******************************\n"
                << std::endl;
    } else {
      tbb::tick_count start_searching = tbb::tick_count::now();
      for (unsigned int iteration = 0; iteration < numberOfIterations;
           ++iteration) {
        auto result =
            clKdtree->search_in_the_box_multiple(minPoints, maxPoints);
      }
      tbb::tick_count end_searching = tbb::tick_count::now();
      std::cout << "searching points using OpenCL FKDTree took "
                << (end_searching - start_searching).seconds() * 1e3 << "ms\n"
                << std::endl;
    }
#endif
  }

  if (runCuda) {
#ifdef __USE_CUDA__
    const size_t maxResultSize = 512;

    unsigned int* host_ids;
    float* host_dimensions;
    unsigned int* host_results;

    // host allocations
    host_ids = (unsigned int*)malloc(nPoints * sizeof(unsigned int));
    host_dimensions = (float*)malloc(3 * nPoints * sizeof(float));
    host_results = (unsigned int*)malloc((nPoints + nPoints * maxResultSize) *
                                         sizeof(unsigned int));

    // initialise ids
    memcpy(host_ids, kdtree.getIdVector().data(),
           nPoints * sizeof(unsigned int));

    // initialise dimensions
    for (int dim = 0; dim < 3; dim++) {
      memcpy(&host_dimensions[nPoints * dim],
             kdtree.getDimensionVector(dim).data(), nPoints * sizeof(float));
    }

    // Device vectors
    float* d_dim = 0;
    unsigned int* d_ids;
    unsigned int* d_results;

    // Allocate device memory
    cudaMalloc(&d_dim, 3 * nPoints * sizeof(float));
    cudaMalloc(&d_ids, nPoints * sizeof(unsigned int));
    cudaMalloc(&d_results,
               (nPoints + nPoints * maxResultSize) * sizeof(unsigned int));

    // Copy host vectors to device
    cudaMemcpy(d_dim, host_dimensions, 3 * nPoints * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, host_ids, nPoints * sizeof(unsigned int),
               cudaMemcpyHostToDevice);
    // cudaMemcpy( d_results, host_results, (nPoints + nPoints *
    // maxResultSize)* sizeof(unsigned int), cudaMemcpyHostToDevice);

    tbb::tick_count start_searching_CUDA = tbb::tick_count::now();

    CUDAKernelWrapper(nPoints, d_dim, d_ids, d_results);
    cudaStreamSynchronize(0);
    tbb::tick_count end_searching_CUDA = tbb::tick_count::now();

    // Back to host
    cudaMemcpy(host_results, d_results,
               (nPoints + nPoints * maxResultSize) * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);

    // Release device memory
    cudaFree(d_dim);
    cudaFree(d_ids);
    cudaFree(d_results);

    if (runTheTests) {
      int totalNumberOfPointsFound = 0;
      for (int p = 0; p < nPoints; p++) {
        unsigned int length = host_results[p];
        totalNumberOfPointsFound += length;
        int firstIndex = nPoints + maxResultSize * p;
      }

      std::cout << "GPU using CUDA found " << totalNumberOfPointsFound
                << " points." << std::endl;
    }

    std::cout << "searching points using CUDA took "
              << (end_searching_CUDA - start_searching_CUDA).seconds() * 1e3
              << "ms\n"
              << std::endl;

    free(host_ids);
    free(host_dimensions);
    free(host_results);
#endif
  }

  if (runFKDTree) {
    if (runTheTests) {
      std::vector<unsigned int> partial_results(nPoints);

      tbb::tick_count start_searching = tbb::tick_count::now();
      //		for (int i = 0; i < nPoints; ++i)
      //			pointsFound+=kdtree.search_in_the_box(minPoints[i],
      // maxPoints[i]).size();

      tbb::parallel_for(0, nPoints, 1, [&](unsigned int i) {

        auto foundPoints = kdtree.search_in_the_box(minPoints[i], maxPoints[i]);
        if (!test_correct_search(points, foundPoints, minPoints[i],
                                 maxPoints[i]))
          exit(1);
        partial_results[i] = foundPoints.size();

      });
      tbb::tick_count end_searching = tbb::tick_count::now();
      pointsFound =
          std::accumulate(partial_results.begin(), partial_results.end(), 0);
      std::cout << "searching points using FKDTree took "
                << (end_searching - start_searching).seconds() * 1e3 << "ms\n"
                << " found points: " << pointsFound
                << "\n******************************\n"
                << std::endl;
    } else {
      tbb::tick_count start_searching = tbb::tick_count::now();
      for (unsigned int iteration = 0; iteration < numberOfIterations;
           ++iteration) {
        tbb::parallel_for(
            0, nPoints, 1, [&](unsigned int i)
            //				for (int i = 0; i < nPoints; ++i)
            { kdtree.search_in_the_box(minPoints[i], maxPoints[i]); });
        //				}
      }
      tbb::tick_count end_searching = tbb::tick_count::now();
      std::cout << "searching points using FKDTree took "
                << (end_searching - start_searching).seconds() * 1e3 << "ms\n"
                << std::endl;
    }
    if (runBranchless) {
      if (runTheTests) {
        std::vector<unsigned int> partial_results(nPoints);

        tbb::tick_count start_searching = tbb::tick_count::now();
        //		for (int i = 0; i < nPoints; ++i)
        //			pointsFound+=kdtree.search_in_the_box(minPoints[i],
        // maxPoints[i]).size();

        tbb::parallel_for(0, nPoints, 1, [&](unsigned int i) {

          std::vector<unsigned int> foundPoints =
              kdtree.search_in_the_box_branchless(minPoints[i], maxPoints[i]);
          if (!test_correct_search(points, foundPoints, minPoints[i],
                                   maxPoints[i]))
            exit(1);
          partial_results[i] = foundPoints.size();

        });
        tbb::tick_count end_searching = tbb::tick_count::now();
        pointsFound =
            std::accumulate(partial_results.begin(), partial_results.end(), 0);
        std::cout << "searching points using branchless FKDTree took "
                  << (end_searching - start_searching).seconds() * 1e3 << "ms\n"
                  << " found points: " << pointsFound
                  << "\n******************************\n"
                  << std::endl;
      } else {
        tbb::tick_count start_searching = tbb::tick_count::now();
        for (unsigned int iteration = 0; iteration < numberOfIterations;
             ++iteration) {
          tbb::parallel_for(
              0, nPoints, 1, [&](unsigned int i)
              //				for (int i = 0; i < nPoints;
              //++i)
              {

                std::vector<unsigned int> foundPoints =
                    kdtree.search_in_the_box_branchless(minPoints[i],
                                                        maxPoints[i]);
              });
          //				}
        }
        tbb::tick_count end_searching = tbb::tick_count::now();
        std::cout << "searching points using branchless FKDTree took "
                  << (end_searching - start_searching).seconds() * 1e3 << "ms\n"
                  << std::endl;
      }
    }
    if (runBFS) {
      if (runTheTests) {
        std::vector<unsigned int> partial_results(nPoints);

        tbb::tick_count start_searching = tbb::tick_count::now();
        //		for (int i = 0; i < nPoints; ++i)
        //			pointsFound+=kdtree.search_in_the_box(minPoints[i],
        // maxPoints[i]).size();

        tbb::parallel_for(0, nPoints, 1, [&](unsigned int i) {

          std::vector<unsigned int> foundPoints =
              kdtree.search_in_the_box_BFS(minPoints[i], maxPoints[i]);
          if (!test_correct_search(points, foundPoints, minPoints[i],
                                   maxPoints[i]))
            exit(1);
          partial_results[i] = foundPoints.size();

        });
        tbb::tick_count end_searching = tbb::tick_count::now();
        pointsFound =
            std::accumulate(partial_results.begin(), partial_results.end(), 0);
        std::cout << "searching points using BFS FKDTree took "
                  << (end_searching - start_searching).seconds() * 1e3 << "ms\n"
                  << " found points: " << pointsFound
                  << "\n******************************\n"
                  << std::endl;
      } else {
        tbb::tick_count start_searching = tbb::tick_count::now();
        for (unsigned int iteration = 0; iteration < numberOfIterations;
             ++iteration) {
          tbb::parallel_for(
              0, nPoints, 1, [&](unsigned int i)
              //				for (int i = 0; i < nPoints;
              //++i)
              {

                std::vector<unsigned int> foundPoints =
                    kdtree.search_in_the_box_BFS(minPoints[i], maxPoints[i]);
              });
          //				}
        }
        tbb::tick_count end_searching = tbb::tick_count::now();
        std::cout << "searching points using BFS FKDTree took "
                  << (end_searching - start_searching).seconds() * 1e3 << "ms\n"
                  << std::endl;
      }
    }
    if (runRecursive) {
      if (runTheTests) {
        std::vector<unsigned int> partial_results(nPoints);

        tbb::tick_count start_searching = tbb::tick_count::now();
        //		for (int i = 0; i < nPoints; ++i)
        //			pointsFound+=kdtree.search_in_the_box(minPoints[i],
        // maxPoints[i]).size();

        tbb::parallel_for(0, nPoints, 1, [&](unsigned int i) {
          std::vector<unsigned int> foundPoints;
          kdtree.search_in_the_box_recursive(minPoints[i], maxPoints[i],
                                             foundPoints);
          if (!test_correct_search(points, foundPoints, minPoints[i],
                                   maxPoints[i]))
            exit(1);
          partial_results[i] = foundPoints.size();

        });
        tbb::tick_count end_searching = tbb::tick_count::now();
        pointsFound =
            std::accumulate(partial_results.begin(), partial_results.end(), 0);
        std::cout << "searching points using recursive FKDTree took "
                  << (end_searching - start_searching).seconds() * 1e3 << "ms\n"
                  << " found points: " << pointsFound
                  << "\n******************************\n"
                  << std::endl;
      } else {
        tbb::tick_count start_searching = tbb::tick_count::now();
        for (unsigned int iteration = 0; iteration < numberOfIterations;
             ++iteration) {
          tbb::parallel_for(
              0, nPoints, 1, [&](unsigned int i)
              //				for (int i = 0; i < nPoints;
              //++i)
              {
                std::vector<unsigned int> foundPoints;

                kdtree.search_in_the_box_recursive(minPoints[i], maxPoints[i],
                                                   foundPoints);
              });
          //				}
        }
        tbb::tick_count end_searching = tbb::tick_count::now();
        std::cout << "searching points using recursive FKDTree took "
                  << (end_searching - start_searching).seconds() * 1e3 << "ms\n"
                  << std::endl;
      }
    }
  }

  //	int pointsFoundNaive = 0;
  //
  if (runSequential) {
    std::chrono::steady_clock::time_point start_sequential =
        std::chrono::steady_clock::now();
    unsigned int pointsFound = 0;
    for (int i = 0; i < nPoints; ++i) {
      for (auto p : points) {
        bool inTheBox = true;

        for (int d = 0; d < 3; ++d) {
          inTheBox &= (p[d] <= maxPoints[i][d] && p[d] >= minPoints[i][d]);
        }
        pointsFound += inTheBox;
      }
    }

    std::chrono::steady_clock::time_point end_sequential =
        std::chrono::steady_clock::now();
    std::cout << "Sequential search algorithm took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     end_sequential - start_sequential)
                     .count()
              << "ms\n"
              << " found points: " << pointsFound
              << "\n******************************\n"
              << std::endl;
  }

  if (runOldKDTree) {
    std::chrono::steady_clock::time_point start_building =
        std::chrono::steady_clock::now();

    KDTreeLinkerAlgo<unsigned, 3> vanilla_tree;
    std::vector<KDTreeNodeInfoT<unsigned, 3>> vanilla_nodes;
    std::vector<KDTreeNodeInfoT<unsigned, 3>> vanilla_founds;

    std::array<float, 3> minpos{{0.0f, 0.0f, 0.0f}}, maxpos{{0.0f, 0.0f, 0.0f}};

    vanilla_tree.clear();
    vanilla_founds.clear();
    for (int i = 0; i < nPoints; ++i) {
      float4 pos = cmssw_points[i];
      vanilla_nodes.emplace_back(i, (float)pos.x, (float)pos.y, (float)pos.z);
      if (i == 0) {
        minpos[0] = pos.x;
        minpos[1] = pos.y;
        minpos[2] = pos.z;
        maxpos[0] = pos.x;
        maxpos[1] = pos.y;
        maxpos[2] = pos.z;
      } else {
        minpos[0] = std::min((float)pos.x, minpos[0]);
        minpos[1] = std::min((float)pos.y, minpos[1]);
        minpos[2] = std::min((float)pos.z, minpos[2]);
        maxpos[0] = std::max((float)pos.x, maxpos[0]);
        maxpos[1] = std::max((float)pos.y, maxpos[1]);
        maxpos[2] = std::max((float)pos.z, maxpos[2]);
      }
    }

    KDTreeCube cluster_bounds = KDTreeCube(minpos[0], maxpos[0], minpos[1],
                                           maxpos[1], minpos[2], maxpos[2]);

    vanilla_tree.build(vanilla_nodes, cluster_bounds);
    std::chrono::steady_clock::time_point end_building =
        std::chrono::steady_clock::now();
    unsigned int pointsFound = 0;
    std::chrono::steady_clock::time_point start_searching =
        std::chrono::steady_clock::now();
    for (unsigned int iteration = 0; iteration < numberOfIterations;
         ++iteration) {
      for (int i = 0; i < nPoints; ++i) {
        KDTreeCube kd_searchcube(minPoints[i][0], maxPoints[i][0],
                                 minPoints[i][1], maxPoints[i][1],
                                 minPoints[i][2], maxPoints[i][2]);
        vanilla_tree.search(kd_searchcube, vanilla_founds);
        pointsFound += vanilla_founds.size();
        vanilla_founds.clear();
      }
    }
    std::chrono::steady_clock::time_point end_searching =
        std::chrono::steady_clock::now();

    std::cout << "building Vanilla CMSSW KDTree with " << nPoints
              << " points took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     end_building - start_building)
                     .count()
              << "ms" << std::endl;
    std::cout << "searching points using Vanilla CMSSW KDTree took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     end_searching - start_searching)
                     .count()
              << "ms" << std::endl;
    std::cout << pointsFound << " points found using Vanilla CMSSW "
                                "KDTree\n******************************\n"
              << std::endl;
  }
  delete[] cmssw_points;

  return 0;
}
