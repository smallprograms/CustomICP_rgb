#ifndef CUSTOMICP_H
#define CUSTOMICP_H
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <boost/thread/thread.hpp>
#include "CustomCorrespondenceEstimation.h"
#include "oflow_pcl.h"
#include "BilateralFilter.h"
#include <pcl/filters/fast_bilateral.h>
#include "SobelFilter.h"
#include <unsupported/Eigen/SparseExtra>

class CustomICP
{
public:
    CustomICP();
    void setInputSource(pcl::PointCloud<pcl::PointXYZRGB>::Ptr src);
    void setInputTarget(pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt);
    void align(pcl::PointCloud<pcl::PointXYZRGB>& cloud);
    Eigen::Matrix4f getFinalTransformation();
    pcl::Correspondences getCorrespondences();
    pcl::PointCloud<pcl::PointXYZRGB> getSourceFiltered();
    pcl::PointCloud<pcl::PointXYZRGB> getTargetFiltered();
    double getFitnessScore();
    void setPrevTransf(Eigen::Matrix4f prevT);
    void setOflowStop(bool val);
    bool foundOflowTransf();
    void randomICP(Eigen::Vector3f maxYawPitchRoll, Eigen::Vector3f maxDist, float maxCorDist, float maxFit, int maxIter, float& bestFit, int& numCorresp);

private:
    //use our custom correspondences estimator
    CustomCorrespondenceEstimation<pcl::PointXYZRGB,pcl::PointXYZRGB,float>* customCorresp;
    SobelFilter<pcl::PointXYZRGB> sobFilter;
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr src;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt;
    pcl::PointCloud<pcl::PointXYZRGB>  srcNonDense;
    pcl::PointCloud<pcl::PointXYZRGB>  tgtNonDense;
    Eigen::Matrix4f oflowTransf;
    Eigen::Matrix4f prevTransf;
    Eigen::Matrix4f finalTransf;
    double fitness;
    bool oflowFound;
    bool stopIfOflowFails;
    pcl::Correspondences correspondences;
};

#endif // CUSTOMICP_H
