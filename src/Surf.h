#ifndef SURF_H
#define SURF_H
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

class Surf
{
public:
    Surf();
    void visualDistance( const int indexA, const int indexB,
                         const pcl::PointCloud<pcl::PointXYZRGB>& cloudA,const pcl::PointCloud<pcl::PointXYZRGB>& cloudB,
                         float& featureDist, int& pixelDist );

    Eigen::Matrix4f getSurfTransform( const int indexA, const int indexB,
                                    const pcl::PointCloud<pcl::PointXYZRGB>& cloudA,
                                      const pcl::PointCloud<pcl::PointXYZRGB>& cloudB
                                    );
    void saveCloudDescriptors(int cloudIndex, const pcl::PointCloud<pcl::PointXYZRGB>& cloudA);

private:
    std::map<int,std::vector<cv::KeyPoint> > cloudKeyPoints;
    std::map<int,cv::Mat>  cloudDescriptors;
};

#endif // SURF_H
