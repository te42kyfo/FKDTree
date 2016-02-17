#include "FKDTree.h"
#include "KDPoint.h"
#include <chrono>
#include "KDTreeLinkerAlgoT.h"
#include <sstream>
#include <unistd.h>
#include <thread>
#include "tbb/tbb.h"
typedef struct float4
{
	float x;
	float y;
	float z;
	float w;
} float4;


static void show_usage(std::string name)
{
	std::cerr << "\nUsage: " << name << " <option(s)>" << " Options:\n"
			<< "\t-h,--help\t\tShow this help message\n"
			<< "\t-n <number of points>\tSpecify the number of points to use for the kdtree\n"
			<< "\t-t \tRun the validity tests\n"
			<< "\t-s \tRun the sequential algo\n"
			<< "\t-c \tRun the vanilla cmssw algo\n"
			<< "\t-f \tRun FKDtree algo\n"
			<< "\t-a \tRun all the algos\n"
			<< "\t-p <number of threads>\tSpecify the number of parallel threads to use\n"
			<< std::endl;

}
int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		show_usage(argv[0]);
		return 1;
	}

	int nPoints=100000;
	int numberOfThreads = 1;
	bool runTheTests = false;
	bool runSequential = false;
	bool runFKDTree = false;
	bool runOldKDTree = false;
	for (int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];
		if ((arg == "-h") || (arg == "--help"))
		{
			show_usage(argv[0]);
			return 0;
		}

		else if (arg == "-n")
		{
			if (i + 1 < argc) // Make sure we aren't at the end of argv!
			{
				i++;
				std::istringstream ss(argv[i]);
				if (!(ss >> nPoints))
				{
					std::cerr << "Invalid number " << argv[i] << '\n';
					exit(1);

				}
			}
		}
		else if (arg == "-t")
		{
			runTheTests = true;
		}
		else if (arg == "-s")
		{
			runSequential = true;
		}
		else if (arg == "-f")
		{
			runFKDTree = true;
		}
		else if (arg == "-c")
		{
			runOldKDTree = true;
		}
		else if (arg == "-a")
		{
			runOldKDTree = true;
			runFKDTree = true;
			runSequential = true;
		}
		else if (arg == "-p")
		{
			if (i + 1 < argc) // Make sure we aren't at the end of argv!
			{
				i++;
				std::istringstream ss(argv[i]);
				if (!(ss >> numberOfThreads))
				{
					std::cerr << "Invalid number of threads " << argv[i] << '\n';

					exit(1);

				}
			}
		}
	}
    tbb::task_scheduler_init init(numberOfThreads);

	std::vector<KDPoint<float, 3> > points;
	std::vector<KDPoint<float, 3> > minPoints;
	std::vector<KDPoint<float, 3> > maxPoints;

	float range_x = 1;
	float range_y = 2;
	float range_z = 1;

	KDPoint<float, 3> minPoint(0, 1, 8);
	KDPoint<float, 3> maxPoint(0.4, 1.2, 8.3);
	for (int i = 0; i < nPoints; ++i)
	{
		float x = static_cast<float>(rand())
				/ (static_cast<float>(RAND_MAX / 10.1));
		;
		float y = static_cast<float>(rand())
				/ (static_cast<float>(RAND_MAX / 10.1));
		;
		float z = static_cast<float>(rand())
				/ (static_cast<float>(RAND_MAX / 10.1));
		KDPoint<float, 3> Point(x, y, z);
		Point.setId(i);

		points.push_back(Point);
		KDPoint<float, 3> m(x - range_x, y - range_y, z - range_z);
		minPoints.push_back(m);
		KDPoint<float, 3> M(x + range_x, y + range_y, z + range_z);
		maxPoints.push_back(M);

	}
//needed by the vanilla algo
	float4* cmssw_points;
	cmssw_points = new float4[nPoints];
	for (int j = 0; j < nPoints; j++)
	{
		cmssw_points[j].x = points[j][0];
		cmssw_points[j].y = points[j][1];
		cmssw_points[j].z = points[j][2];
		cmssw_points[j].w = j; //Use this to save the index

	}

	std::cout << "Cloud of points generated.\n" << std::endl;
	std::this_thread::sleep_for(std::chrono::seconds(1));

	if (runFKDTree)
	{
		long int pointsFound=0;
		std::cout << "FKDTree run will start in 1 second.\n" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));

		std::chrono::steady_clock::time_point start_building =
				std::chrono::steady_clock::now();
		FKDTree<float, 3> kdtree(nPoints, points);

		kdtree.build();
		std::chrono::steady_clock::time_point end_building =
				std::chrono::steady_clock::now();
		std::cout << "building FKDTree with " << nPoints << " points took "
				<< std::chrono::duration_cast < std::chrono::milliseconds
				> (end_building - start_building).count() << "ms" << std::endl;
		if (runTheTests)
		{
			if (kdtree.test_correct_build())
				std::cout << "FKDTree built correctly" << std::endl;
			else
				std::cerr << "FKDTree wrong" << std::endl;
		}

	    tbb::tick_count start_searching =
	    		tbb::tick_count::now();
//		for (int i = 0; i < nPoints; ++i)
//			pointsFound+=kdtree.search_in_the_box(minPoints[i], maxPoints[i]).size();
	    tbb::parallel_for(0, nPoints, 1, [=](int i) {

	    		kdtree.search_in_the_box(minPoints[i], maxPoints[i]);
	        		});
		tbb::tick_count end_searching =
				tbb::tick_count::now();

		std::cout << "searching points using FKDTree took "
				<< (end_searching - start_searching).seconds()*1e3<< "ms\n"
				<< " found points: " << pointsFound<< "\n******************************\n"
						<< std::endl;
	}
//	int pointsFoundNaive = 0;
//
	if (runSequential)
	{
		std::cout << "Sequential run will start in 1 second.\n" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
		std::chrono::steady_clock::time_point start_sequential =
				std::chrono::steady_clock::now();
		long int pointsFound = 0;
		for (int i = 0; i < nPoints; ++i)
		{
			for (auto p : points)
			{

				bool inTheBox = true;

				for (int d = 0; d < 3; ++d)
				{

					inTheBox &= (p[d] <= maxPoints[i][d]
							&& p[d] >= minPoints[i][d]);

				}
				pointsFound += inTheBox;
			}

		}

		std::chrono::steady_clock::time_point end_sequential =
				std::chrono::steady_clock::now();
		std::cout << "Sequential search algorithm took "
				<< std::chrono::duration_cast < std::chrono::milliseconds
				> (end_sequential - start_sequential).count() << "ms\n"
				<< " found points: " << pointsFound<< "\n******************************\n" << std::endl;
	}

	if (runOldKDTree)
	{


		std::cout << "Vanilla CMSSW KDTree run will start in 1 second.\n"
				<< std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
		std::chrono::steady_clock::time_point start_building =
				std::chrono::steady_clock::now();

		KDTreeLinkerAlgo<unsigned, 3> vanilla_tree;
		std::vector<KDTreeNodeInfoT<unsigned, 3> > vanilla_nodes;
		std::vector<KDTreeNodeInfoT<unsigned, 3> > vanilla_founds;

		std::array<float, 3> minpos
		{
		{ 0.0f, 0.0f, 0.0f } }, maxpos
		{
		{ 0.0f, 0.0f, 0.0f } };

		vanilla_tree.clear();
		vanilla_founds.clear();
		for (unsigned i = 0; i < nPoints; ++i)
		{
			float4 pos = cmssw_points[i];
			vanilla_nodes.emplace_back(i, (float) pos.x, (float) pos.y,
					(float) pos.z);
			if (i == 0)
			{
				minpos[0] = pos.x;
				minpos[1] = pos.y;
				minpos[2] = pos.z;
				maxpos[0] = pos.x;
				maxpos[1] = pos.y;
				maxpos[2] = pos.z;
			}
			else
			{
				minpos[0] = std::min((float) pos.x, minpos[0]);
				minpos[1] = std::min((float) pos.y, minpos[1]);
				minpos[2] = std::min((float) pos.z, minpos[2]);
				maxpos[0] = std::max((float) pos.x, maxpos[0]);
				maxpos[1] = std::max((float) pos.y, maxpos[1]);
				maxpos[2] = std::max((float) pos.z, maxpos[2]);
			}
		}

		KDTreeCube cluster_bounds = KDTreeCube(minpos[0], maxpos[0], minpos[1],
				maxpos[1], minpos[2], maxpos[2]);

		vanilla_tree.build(vanilla_nodes, cluster_bounds);
		std::chrono::steady_clock::time_point end_building =
				std::chrono::steady_clock::now();
		long int pointsFound = 0;
		std::chrono::steady_clock::time_point start_searching =
				std::chrono::steady_clock::now();
		for (int i = 0; i < nPoints; ++i)
		{
			KDTreeCube kd_searchcube(minPoints[i][0], maxPoints[i][0],
					minPoints[i][1], maxPoints[i][1], minPoints[i][2],
					maxPoints[i][2]);
			vanilla_tree.search(kd_searchcube, vanilla_founds);
			pointsFound += vanilla_founds.size();
			vanilla_founds.clear();
		}
		std::chrono::steady_clock::time_point end_searching =
				std::chrono::steady_clock::now();

		std::cout << "building Vanilla CMSSW KDTree with " << nPoints << " points took "
				<< std::chrono::duration_cast < std::chrono::milliseconds
				> (end_building - start_building).count() << "ms" << std::endl;
		std::cout << "searching points using Vanilla CMSSW KDTree took "
				<< std::chrono::duration_cast < std::chrono::milliseconds
				> (end_searching - start_searching).count() << "ms"
						<< std::endl;
		std::cout << pointsFound << " points found using Vanilla CMSSW KDTree\n******************************\n"<< std::endl;

		delete[] cmssw_points;
	}
	return 0;
}
