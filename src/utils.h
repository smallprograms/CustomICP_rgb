#ifndef UTILS_H
#define UTILS_H
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <pcl/kdtree/kdtree_flann.h>

void cloudToMat(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, cv::Mat& outMat);
void setCloudAsNaN(pcl::PointCloud<pcl::PointXYZRGB> &cloud);
void removeFarPoints(pcl::PointCloud<pcl::PointXYZRGB> &cloud, float z);
void filterMask(const pcl::PointCloud<pcl::PointXYZRGB>& cloudIn, cv::Mat mask,
                pcl::PointCloud<pcl::PointXYZRGB> &cloudOut);
void writeTransformationQuaternion(Eigen::Matrix4f transf, std::string fileName);
void writeNumber(float number, std::string fileName);
void writeNumber(int number, std::string fileName);
void writeTwoNumbers(int fromIndex, int toIndex , std::string fileName);
void writeEdge(const int fromIndex,const int toIndex,Eigen::Matrix4f& relativePose,
               Eigen::Matrix<double,6,6>& informationMatrix, std::string fileNamePrefix);
Eigen::Matrix4f quaternionToMatrix(float tx, float ty, float tz,
                                   float qx, float qy, float qz, float qw );

void reorthogonalizeMatrix(Eigen::Matrix4d& transf);
void writeTransformationMatrix(Eigen::Matrix4f transf, std::string fileName);
float matrixDistance(Eigen::Matrix4f transf1,Eigen::Matrix4f transf2);
Eigen::Matrix4f loadTransformationMatrix(std::string fileName);
int endswith(const char* haystack, const char* needle);
void mergeClouds(pcl::PointCloud<pcl::PointXYZRGB>& globalCloud, const pcl::PointCloud<pcl::PointXYZRGB>& newCloud);

#endif
