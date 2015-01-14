#ifndef CUSTOMICP_H
#define CUSTOMICP_H
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <boost/thread/thread.hpp>
#include "oflow_pcl.h"
#include "BilateralFilter.h"
#include <pcl/filters/fast_bilateral.h>
#include "SobelFilter.h"
#include "EdgeFilter.h"
#include <unsupported/Eigen/SparseExtra>

class CustomICP
{
public:
    CustomICP();
    void setInputSource(pcl::PointCloud<pcl::PointXYZRGB>::Ptr src);
    void setInputTarget(pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt);
    void setPrevTransf(Eigen::Matrix4f prevT);
    void setOflowStop(bool val);

    void align(pcl::PointCloud<pcl::PointXYZRGB>& cloud, Eigen::Matrix4f guess, float max_dist);
    void align(pcl::PointCloud<pcl::PointXYZRGB>& cloud, Eigen::Matrix4f guess, bool loop);

    Eigen::Matrix4f getFinalTransformation();
    pcl::Correspondences getCorrespondences();
    pcl::PointCloud<pcl::PointXYZRGB> getSourceFiltered();
    pcl::PointCloud<pcl::PointXYZRGB> getTargetFiltered();

    bool foundOflowTransf();
    double getFitnessScore();
    float getPhotoConsistency();
    float getPhotoConsistency(Eigen::Matrix4f ctransf);
    float getPhotoConsistency(pcl::PointCloud<pcl::PointXYZRGB>& cloudA,pcl::PointCloud<pcl::PointXYZRGB>& cloudB,Eigen::Matrix4f ctransf);
    float photoConsistency(pcl::PointCloud<pcl::PointXYZRGB> &cloudSrc, pcl::PointCloud<pcl::PointXYZRGB> &cloudTgt,Eigen::Matrix4f transf);

private:

    SobelFilter<pcl::PointXYZRGB> sobFilter;
    EdgeFilter edgeFilter;
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
    pcl::VoxelGrid<pcl::PointXYZRGB> voxelFilter;

};

#endif // CUSTOMICP_H
