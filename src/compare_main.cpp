#include "gpu_euclidean_clustering.h"

#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>

constexpr size_t kClusterSizeMin = 8;

constexpr size_t kClusterSizeMax = 10000000;

constexpr double kClusteringDistance = 0.8;

std::vector<pcl::PointIndices> pclEuclideanClustering(const std::string& t_file)
{
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::PCDReader reader;
    reader.read(t_file, *cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> euclideanClusterExtraction;
    euclideanClusterExtraction.setClusterTolerance(kClusteringDistance);
    euclideanClusterExtraction.setMinClusterSize(kClusterSizeMin);
    euclideanClusterExtraction.setMaxClusterSize(kClusterSizeMax);
    euclideanClusterExtraction.setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusterIndices;
    euclideanClusterExtraction.extract(clusterIndices);

    // sort indices by size and then by first element
    for (auto cluster: clusterIndices)
    {
        std::sort(cluster.indices.begin(), cluster.indices.end());
    }

    std::sort(clusterIndices.begin(), clusterIndices.end(), [](const auto& t_cluster1, const auto& t_cluster2)
    {
        return t_cluster1.indices[0] < t_cluster2.indices[0];
    });


    return clusterIndices;
}

std::vector<pcl::PointIndices> autowareEuclideanClustering(const std::string& t_file)
{
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::PCDReader reader;
    reader.read(t_file, *cloud);

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

    gpuEuclideanCluster.setThreshold(kClusteringDistance);
    gpuEuclideanCluster.setMinClusterPts(kClusterSizeMin);
    gpuEuclideanCluster.setMaxClusterPts(kClusterSizeMax);

    std::vector<GpuEuclideanCluster::GClusterIndex> gpuClusterIndices;

    // This code gets timed
    gpuEuclideanCluster.setInputPoints(tmpX.data(), tmpY.data(), tmpZ.data(), static_cast<int>(size));
    gpuEuclideanCluster.extractClusters();
    gpuClusterIndices = gpuEuclideanCluster.getOutput();

    // convert to std::vector<pcl::PointIndices>
    std::vector<pcl::PointIndices> clusterIndices;
    clusterIndices.reserve(gpuClusterIndices.size());

    for (const auto& ind: gpuClusterIndices)
    {
        pcl::PointIndices pInd;
        pInd.indices = ind.points_in_cluster;
        clusterIndices.push_back(pInd);
    }

    // sort indices by size and then by first element
    for (auto cluster: clusterIndices)
    {
        std::sort(cluster.indices.begin(), cluster.indices.end());
    }

    std::sort(clusterIndices.begin(), clusterIndices.end(), [](const auto& t_cluster1, const auto& t_cluster2)
    {
        return t_cluster1.indices[0] < t_cluster2.indices[0];
    });

    return clusterIndices;
}

bool clustersIsSame(const std::vector<pcl::PointIndices>& t_pointIndices1, const std::vector<pcl::PointIndices>& t_pointIndices2)
{
    if (t_pointIndices1.size() != t_pointIndices2.size())
    {
        return false;
    }

    for (int clusterInd = 0; clusterInd < t_pointIndices1.size(); ++clusterInd)
    {
        if (t_pointIndices1[clusterInd].indices.size() != t_pointIndices2[clusterInd].indices.size())
        {
            return false;
        }

        for (int i = 0; i < t_pointIndices1[clusterInd].indices.size(); ++i)
        {
            if (t_pointIndices1[clusterInd].indices[i] != t_pointIndices2[clusterInd].indices[i])
            {
                return false;
            }
        }
    }

    return true;
}

int main(int t_argc, char** t_argv)
{
    if (t_argc < 2)
    {
        std::cerr << "No test files given." << std::endl;
        return 1;
    }

    std::string testFilePath = t_argv[1];

    auto pclCpuInd = pclEuclideanClustering(testFilePath);
    auto autowareInd = autowareEuclideanClustering(testFilePath);

    std::cout << "Is pcl_cpu result same as autoware = " << std::boolalpha << clustersIsSame(pclCpuInd, autowareInd) << std::endl;

    return 0;
}