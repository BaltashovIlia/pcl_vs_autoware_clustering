#include "gpu_euclidean_clustering.h"

#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/gpu/octree/octree.hpp>
#include <pcl/gpu/segmentation/gpu_extract_clusters.h>

#include <benchmark/benchmark.h>

constexpr size_t kClusterSizeMin = 8;

constexpr size_t kClusterSizeMax = 10000000;

constexpr double kClusteringDistance = 0.8;

static void
BM_PclEuclideanClustering(benchmark::State& t_state, const std::string& t_file)
{
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::PCDReader reader;
    reader.read(t_file, *cloud);

    for (auto _: t_state)
    {
        // This code gets timed
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> euclideanClusterExtraction;
        euclideanClusterExtraction.setClusterTolerance(kClusteringDistance);
        euclideanClusterExtraction.setMinClusterSize(kClusterSizeMin);
        euclideanClusterExtraction.setMaxClusterSize(kClusterSizeMax);
        euclideanClusterExtraction.setInputCloud(cloud);

        std::vector<pcl::PointIndices> clusterIndices;
        euclideanClusterExtraction.extract(clusterIndices);
    }
}

static void
BM_PclEuclideanClusteringGpu(benchmark::State& t_state, const std::string& t_file)
{
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::PCDReader reader;
    reader.read(t_file, *cloud);

    for (auto _: t_state)
    {
        // This code gets timed
        pcl::gpu::Octree::PointCloud cloudDevice;
        cloudDevice.upload(cloud->points);
        pcl::gpu::Octree::Ptr octreeDevice = pcl::make_shared<pcl::gpu::Octree>();
        octreeDevice->setCloud(cloudDevice);
        octreeDevice->build();

        pcl::gpu::EuclideanClusterExtraction<pcl::PointXYZ> euclideanClusterExtraction;
        euclideanClusterExtraction.setClusterTolerance(kClusteringDistance);
        euclideanClusterExtraction.setMinClusterSize(kClusterSizeMin);
        euclideanClusterExtraction.setMaxClusterSize(kClusterSizeMax);
        euclideanClusterExtraction.setSearchMethod(octreeDevice);
        euclideanClusterExtraction.setHostCloud(cloud);

        std::vector<pcl::PointIndices> clusterIndices;
        euclideanClusterExtraction.extract(clusterIndices);
    }
}

static void
BM_AutowareEuclideanClustering(benchmark::State& t_state, const std::string& t_file)
{
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::PCDReader reader;
    reader.read(t_file, *cloud);

    for (auto _: t_state)
    {
        // This code gets timed
        // Convert input point cloud to vectors of x, y, and z
        size_t size = cloud->points.size();

        std::vector<float> tmpX(size);
        std::vector<float> tmpY(size);
        std::vector<float> tmpZ(size);

        for (size_t i = 0; i < size; i++)
        {
            pcl::PointXYZ tmpPoint = cloud->at(i);

            tmpX[i] = tmpPoint.x;
            tmpY[i] = tmpPoint.y;
            tmpZ[i] = tmpPoint.z;
        }
        // Кластеризация на GPU
        GpuEuclideanCluster gpuEuclideanCluster;
        gpuEuclideanCluster.setInputPoints(tmpX.data(), tmpY.data(), tmpZ.data(), static_cast<int>(size));
        gpuEuclideanCluster.setThreshold(kClusteringDistance);
        gpuEuclideanCluster.setMinClusterPts(kClusterSizeMin);
        gpuEuclideanCluster.setMaxClusterPts(kClusterSizeMax);

        std::vector<GpuEuclideanCluster::GClusterIndex> clusterIndices;

        gpuEuclideanCluster.extractClusters();
        clusterIndices = gpuEuclideanCluster.getOutput();

        // pcl returns a sorted vector, so let's sort here too
        std::sort(clusterIndices.begin(), clusterIndices.end(), [](const auto& t_cluster1, const auto& t_cluster2)
        {
            return t_cluster1.points_in_cluster.size() > t_cluster2.points_in_cluster.size();
        });
    }
}

int main(int t_argc, char** t_argv)
{
    if (t_argc < 2)
    {
        std::cerr << "No test files given." << std::endl;
        return 1;
    }

    std::string testFilePath = t_argv[1];

    benchmark::RegisterBenchmark("BM_PclEuclideanClustering", &BM_PclEuclideanClustering, testFilePath)->Unit(benchmark::kMillisecond)->MinTime(10);
    benchmark::RegisterBenchmark("BM_PclEuclideanClusteringGpu", &BM_PclEuclideanClusteringGpu, testFilePath)
        ->Unit(benchmark::kMillisecond)->MinTime(10);
    benchmark::RegisterBenchmark("BM_AutowareEuclideanClustering", &BM_AutowareEuclideanClustering, testFilePath)
        ->Unit(benchmark::kMillisecond)->MinTime(10);

    benchmark::Initialize(&t_argc, t_argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
