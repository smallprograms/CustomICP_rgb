#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>


void cloudToMat(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, cv::Mat& outMat);
Eigen::Matrix4f getOflow3Dtransf(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudA, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudB, float maxCorrespDist);


