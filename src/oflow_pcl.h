#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>


Eigen::Matrix4f getOflow3Dtransf(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudA, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudB, float maxCorrespDist);


