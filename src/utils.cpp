#include "utils.h"

void cloudToMat(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, cv::Mat& outMat){

    for(int w=0; w < cloud->width; w++) {
        for( int h=0; h < cloud->height; h++) {

                outMat.at<cv::Vec3b>(h,w)[0] = (*cloud)(w,h).b;
                outMat.at<cv::Vec3b>(h,w)[1] = (*cloud)(w,h).g;
                outMat.at<cv::Vec3b>(h,w)[2] = (*cloud)(w,h).r;
         }
    }

}

void setCloudAsNaN(pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
    for( size_t m=0; m < cloud.width;m++ ) {
        for( size_t n=0; n < cloud.height; n++) {
            cloud(m,n).x=cloud(m,n).y=cloud(m,n).z = std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void removeFarPoints(pcl::PointCloud<pcl::PointXYZRGB> &cloud, float z)
{
    for( size_t m=0; m < cloud.width;m++ ) {
        for( size_t n=0; n < cloud.height; n++) {
            if( cloud(m,n).z > z ) {
                cloud(m,n).x=cloud(m,n).y=cloud(m,n).z = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void filterMask(const pcl::PointCloud<pcl::PointXYZRGB>& cloudIn, cv::Mat mask,pcl::PointCloud<pcl::PointXYZRGB> &cloudOut) {

    for(size_t m=0; m < cloudIn.width; m++) {
        for(size_t n=0; n < cloudIn.height; n++) {


            if( cloudIn.at(m,n).z > 0.1 ) {

                if( mask.at<uchar>(n,m) > 0 ) {

                    cloudOut.at(m,n) = cloudIn.at(m,n);
                }

            }
        }
    }
}
