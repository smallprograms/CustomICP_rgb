#ifndef EDGEFILTER_H
#define EDGEFILTER_H
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include "opencv2/opencv.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/flann/flann_base.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <map>
#include "utils.h"

class EdgeFilter
{
public:

    EdgeFilter();
    void setSourceCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source);
    void setTargetCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target);
    void applyFilter(pcl::PointCloud<pcl::PointXYZRGB> &sourceFiltered,
                     pcl::PointCloud<pcl::PointXYZRGB> &targetFiltered,float max_dist);

private:

    cv::Mat getSobelBorders(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,int sobelThreshold=50);
    cv::Mat getCannyBorders(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);
    cv::Mat getBorders(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,int sobelThreshold=50);

    void visitNeighBoor(std::vector<cv::Point2i> neighBoor, const cv::Mat& bordersImage,
                                    cv::Mat& visitedImage,cv::Mat& localImage, int& count );

    std::vector<cv::Point2i> generateNeighBoor(const cv::Mat& bordersImage,
                                                           const cv::Mat& visitedImage, int row, int col);

    bool hasSomeNeighBoor(const cv::Mat& img,int row, int col, int dist);
    /** try to remove isolated blobs from image, to get only long borders */
    void removeNoise(cv::Mat& bordersImage);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr sourceCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetCloud;
    size_t win_width;
    size_t win_height;
};

#endif // EDGEFILTER_H
