#ifndef UTILS_H
#define UTILS_H
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "opencv2/opencv.hpp"

void cloudToMat(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, cv::Mat& outMat);
void setCloudAsNaN(pcl::PointCloud<pcl::PointXYZRGB> &cloud);
void filterMask(const pcl::PointCloud<pcl::PointXYZRGB>& cloudIn, cv::Mat mask,pcl::PointCloud<pcl::PointXYZRGB> &cloudOut);
#endif
