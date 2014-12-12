/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef SOBEL_FILTER_H
#define SOBEL_FILTER_H

#include <pcl/filters/filter.h>
#include <pcl/search/pcl_search.h>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/flann/flann_base.hpp"
#include "oflow_pcl.h"

#define MIN3(x,y,z)  ((y) <= (z) ? \
    ((x) <= (y) ? (x) : (y)) \
    : \
    ((x) <= (z) ? (x) : (z)))

#define MAX3(x,y,z)  ((y) >= (z) ? \
    ((x) >= (y) ? (x) : (y)) \
    : \
    ((x) >= (z) ? (x) : (z)))


inline float flann_knn(cv::Mat& m_destinations, cv::Mat& m_object, std::vector<int>& ptpairs, std::vector<float>& dists) {
    using namespace cv;
    // find nearest neighbors using FLANN
    cv::Mat m_indices(m_object.rows, 1, CV_32S);
    cv::Mat m_dists(m_object.rows, 1, CV_32F);

    Mat dest_32f; m_destinations.convertTo(dest_32f,CV_32FC2);
    Mat obj_32f; m_object.convertTo(obj_32f,CV_32FC2);

    assert(dest_32f.type() == CV_32F);

    cv::flann::Index flann_index(dest_32f, cv::flann::KDTreeIndexParams(2));  // using 2 randomized kdtrees
    flann_index.knnSearch(obj_32f, m_indices, m_dists, 1, cv::flann::SearchParams(64) );

    int* indices_ptr = m_indices.ptr<int>(0);
    //float* dists_ptr = m_dists.ptr<float>(0);
    for (int i=0;i<m_indices.rows;++i) {
        //target index:i source index:indices.at(i)
        ptpairs.push_back(indices_ptr[i]);
    }

    dists.resize(m_dists.rows);
    m_dists.copyTo(Mat(dists));

    return cv::sum(m_dists)[0]/m_dists.rows;
}
/** Input: source matrix, destination matrix, Output: rotation matrix and translation vector */
inline void findTransformSVD(cv::Mat& _m, cv::Mat& _d, cv::Mat& outR, cv::Scalar& outT) {
    using namespace cv;
    Mat m; _m.convertTo(m,CV_32F);
    Mat d; _d.convertTo(d,CV_32F);

    Mat mMean;
    Mat dMean;
    //get centroids
    reduce(m, mMean, 0,CV_REDUCE_AVG);
    reduce(d, dMean, 0,CV_REDUCE_AVG);
    //substract centroids from each point set
    for(int k=0;k<m.rows;k++) {
        m.row(k).at<float>(0) -=  mMean.row(0).at<float>(0);
        m.row(k).at<float>(1) -=  mMean.row(0).at<float>(1);
    }

    for(int k=0;k<d.rows;k++) {
        d.row(k).at<float>(0) -=  dMean.row(0).at<float>(0);
        d.row(k).at<float>(1) -=  dMean.row(0).at<float>(1);
    }
    //calculate H matrix
    Mat H = Mat::zeros(2,2,CV_32FC1);
    for(int i=0;i<m.rows;i++) {
        Mat mci = m.row(i);
        Mat dci = d.row(i);
        H = H + mci.t()  * dci;
    }
    //calculate SVD over H to find rotation
    cv::SVD svd(H);

    //obtain rotation
    Mat R = svd.vt.t() * svd.u.t();
    //check if R is orthogonal?
    double det_R = cv::determinant(R);
    if(abs(det_R + 1.0) < 0.0001) {
        float _tmp[4] = {1,0,0,cv::determinant(svd.vt*svd.u)};
        R = svd.u * Mat(2,2,CV_32FC1,_tmp) * svd.vt;
    }

    float* _R = R.ptr<float>(0);
    //translation is d - m*rot
    Scalar T(dMean.row(0).at<float>(0) - (mMean.row(0).at<float>(0)*_R[0] + mMean.row(0).at<float>(1)*_R[1]),
            dMean.row(0).at<float>(1) - (mMean.row(0).at<float>(0)*_R[2] + mMean.row(0).at<float>(1)*_R[3]));


    outR = R;
    outT = T;

}

/** \brief A bilateral filter implementation for point cloud data. Uses the intensity data channel.
    * \note For more information please see
    * <b>C. Tomasi and R. Manduchi. Bilateral Filtering for Gray and Color Images.
    * In Proceedings of the IEEE International Conference on Computer Vision,
    * 1998.</b>
    * \author Luca Penasa
    * \ingroup filters
    */
template<typename PointT>
class SobelFilter : public pcl::Filter<PointT>
{
    using pcl::Filter<PointT>::input_;
    using pcl::Filter<PointT>::indices_;
    typedef typename pcl::Filter<PointT>::PointCloud PointCloud;
    typedef typename pcl::search::Search<PointT>::Ptr KdTreePtr;

public:
    /** \brief Constructor.
    * Sets sigma_s_ to 0 and sigma_r_ to MAXDBL
    */
    SobelFilter () : sigma_s_ (100000),
        sigma_r_ (100000000),
        tree_ (),
        win_width(3),
        win_height(3)
    {
    }


    /** \brief Filter the input data and store the results into output
    * \param[out] output the resultant point cloud message
    */
    void
    applyFilter (PointCloud &output);
    cv::Mat getBorders(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);
    cv::Mat getPoints(cv::Mat bordersImage);
    /** try to remove isolated blobs from image, we want only long borders */
    void removeNoise(cv::Mat& bordersImage);
    void setCloudAsNaN(pcl::PointCloud<pcl::PointXYZRGB> &cloud);
    void setSourceCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source);
    void setTargetCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target);
    void applyFilter(pcl::PointCloud<pcl::PointXYZRGB> &sourceFiltered,pcl::PointCloud<pcl::PointXYZRGB> &targetFiltered);
    cv::Mat runICP(cv::Mat src, cv::Mat tgt, int maxIter,std::vector<int>& pairs, std::vector<float>& dists);

    /** \brief Compute the intensity average for a single point
    * \param[in] pid the point index to compute the weight for
    * \param[in] indices the set of nearest neighor indices
    * \param[in] distances the set of nearest neighbor distances
    * \return the intensity average at a given point index
    */
    pcl::PointXYZRGBA
    computePointWeight (const int pid, const std::vector<int> &indices, const std::vector<float> &distances);

    /** \brief Set the half size of the Gaussian bilateral filter window.
    * \param[in] sigma_s the half size of the Gaussian bilateral filter window to use
    */
    inline void
    setHalfSize (const double sigma_s)
    {
        sigma_s_ = sigma_s;
    }

    /** \brief Get the half size of the Gaussian bilateral filter window as set by the user. */
    double
    getHalfSize ()
    {
        return (sigma_s_);
    }

    /** \brief Set the standard deviation parameter
    * \param[in] sigma_r the new standard deviation parameter
    */
    void
    setStdDev (const double sigma_r)
    {
        sigma_r_ = sigma_r;
    }

    /** \brief Get the value of the current standard deviation parameter of the bilateral filter. */
    double
    getStdDev ()
    {
        return (sigma_r_);
    }

    /** \brief Provide a pointer to the search object.
    * \param[in] tree a pointer to the spatial search object.
    */
    void
    setSearchMethod (const KdTreePtr &tree)
    {
        tree_ = tree;
    }

    void
    setWindowSize(size_t width, size_t height) {
        win_width = width;
        win_height = height;
    }

private:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr sourceCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetCloud;
    /** \brief The bilateral filter Gaussian distance kernel.
    * \param[in] x the spatial distance (distance or intensity)
    * \param[in] sigma standard deviation
    */
    inline double
    kernel (double x, double sigma)
    {
        return (exp (- (x*x)/(2*sigma*sigma)));
    }

    cv::Mat getImageBorders(pcl::PointCloud<pcl::PointXYZRGB>::Ptr);

    /** \brief The half size of the Gaussian bilateral filter window (e.g., spatial extents in Euclidean). */
    double sigma_s_;
    /** \brief The standard deviation of the bilateral filter (e.g., standard deviation in intensity). */
    double sigma_r_;

    /** \brief A pointer to the spatial search object. */
    KdTreePtr tree_;

    size_t win_width;
    size_t win_height;
};

template <typename PointT> cv::Mat
SobelFilter<PointT>::getImageBorders(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {

    cv::Mat imgColor(480,640,CV_8UC3);
    cloudToMat(cloud->makeShared(),imgColor);

    cv::Mat imgGray(480,640,CV_8UC1);
    cv::cvtColor(imgColor,imgGray,CV_BGR2GRAY);

    using namespace cv;
    GaussianBlur( imgGray, imgGray, Size(3,3), 0, 0, BORDER_DEFAULT );
    Mat grad_x, grad_y;
    Mat grad;
    Mat abs_grad_x, abs_grad_y;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    Sobel( imgGray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_CONSTANT );
    convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( imgGray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_CONSTANT );
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    threshold(grad,grad,75,255,THRESH_BINARY);




    return grad;

}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
SobelFilter<PointT>::applyFilter (PointCloud &output)
{
    //std::cout << "Aplying sobel filter\n";

    // Check if sigma_s has been given by the user
    //    if (sigma_s_ == 0)
    //    {
    //        PCL_ERROR ("[pcl::SobelFilter::applyFilter] Need a sigma_s value given before continuing.\n");
    //        return;
    //    }
    // In case a search method has not been given, initialize it using some defaults
    /*
    if (!tree_)
    {

        // For organized datasets, use an OrganizedDataIndex
        if (input_->isOrganized ())
            tree_.reset (new pcl::search::OrganizedNeighbor<PointT> ());
        // For unorganized data, use a FLANN kdtree
        else
            tree_.reset (new pcl::search::KdTree<PointT> (false));
    }
    tree_->setInputCloud (input_);
*/

    for( size_t m=0; m < input_->width;m++ ) {
        for( size_t n=0; n < input_->height; n++) {
            output(m,n).x=output(m,n).y=output(m,n).z = std::numeric_limits<float>::quiet_NaN();
        }
    }

    int numPoints=0;
    cv::Mat imgBorder = cv::Mat::zeros(480,640,CV_8UC1);
    return;
    cv::Mat rgbBorder;;// = getImageBorders(input_);

    for(size_t m=win_width/2+1; m < (input_->width-win_width/2-1); m++) {
        for(size_t n=win_height/2+1; n < (input_->height-win_height/2-1); n++) {


            size_t j_min = n-win_width/2;
            size_t j_max = n + win_width/2;
            size_t i_min = m-win_height/2;
            size_t i_max = m + win_height/2;

            if( pcl::isFinite(input_->at(i_min,j_min)) && pcl::isFinite(input_->at(i_max,j_min)) && pcl::isFinite(input_->at(m,j_min)) &&
                    pcl::isFinite(input_->at(m,j_max))
                    && pcl::isFinite(input_->at(i_min,j_max)) && pcl::isFinite(input_->at(i_max,j_max)) && pcl::isFinite(input_->at(i_max,n))
                    && pcl::isFinite(input_->at(i_min,n)) /*
                            && input_->at(i_min,j_min).z != 0 && input_->at(i_max,j_min).z != 0 && input_->at(m,j_min).z != 0
                            && input_->at(m,j_max).z != 0 && input_->at(i_min,j_max).z !=0 && input_->at(i_max,j_max).z != 0 &&
                            input_->at(i_max,n).z != 0 && input_->at(i_min,n).z != 0 */) {


                //horizontal diff mask
                float distH = std::abs(input_->at(i_min,j_min).z - input_->at(i_min,j_max).z);
                distH += 2*std::abs(input_->at(m,j_min).z - input_->at(m,j_max).z);
                distH += std::abs(input_->at(i_max,j_min).z - input_->at(i_max,j_max).z);

                //std::cout <<input_->at(i_min,j_min).z << "\n";

                //vertical diff mask
                float distV = std::abs(input_->at(i_min,j_min).z - input_->at(i_max,j_min).z);
                distV += 2*std::abs(input_->at(i_min,n).z - input_->at(i_max,n).z);
                distV += std::abs(input_->at(i_min,j_max).z - input_->at(i_max,j_max).z);

                float distG = std::sqrt( distH*distH + distV*distV );

                //const float thresh = 0.1;
                const float thresh = 0.2;

                if( rgbBorder.at<uchar>(n,m) > 150 ) {
                    output(m,n) = input_->at(m,n);
                    imgBorder.at<uchar>(n,m) = 255;
                    /** INCLUDE MORE POINTS*
                     output(m+1,n) = input_->at(m+1,n);
                     output(m-1,n) = input_->at(m-1,n);
                     output(m,n+1) = input_->at(m,n+1);
                     output(m,n-1) = input_->at(m,n-1);
                     output(m-1,n-1) = input_->at(m-1,n-1);
                     output(m+1,n-1) = input_->at(m+1,n-1);
                     output(m+1,n+1) = input_->at(m+1,n+1);
                     output(m-1,n+1) = input_->at(m-1,n+1);
                    /**/
                    numPoints++;


                }

            }
        }
    }
    //    static std::string id("a");
    //    std::string name=std::string("sobel_img")+id+std::string(".jpg");
    //    imwrite(name.c_str(),imgBorder);
    //    if( id=="a") id="b";
    //    else if( id=="b") id="a";

    //std::cout << "ssssssssssssssssssnum points passed SobeL less points: " << numPoints << "\n dsafasdf\n";

    //reject far lines!
    //    const float MAX_Z_DIST=2.5;
    //    for(size_t m=win_width/2+1; m < (input_->width-win_width/2-1); m++) {
    //        for(size_t n=win_height/2+1; n < (input_->height-win_height/2-1); n++) {

    //            if( output(m,n).z != std::numeric_limits<float>::quiet_NaN() ) {

    //                if( output(m,n).z > MAX_Z_DIST )  output(m,n).x=output(m,n).y=output(m,n).z = std::numeric_limits<float>::quiet_NaN();
    //            }
    //        }
    //    }

}
template <typename PointT>
cv::Mat SobelFilter<PointT>::getBorders(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& bordersImage)
{
    int numPoints=0;
    cv::Mat imgBorder = cv::Mat::zeros(480,640,CV_8UC1);
    cv::Mat rgbBorder = getImageBorders(bordersImage);

    for(size_t m=win_width/2+1; m < (bordersImage->width-win_width/2-1); m++) {
        for(size_t n=win_height/2+1; n < (bordersImage->height-win_height/2-1); n++) {


            size_t j_min = n-win_width/2;
            size_t j_max = n + win_width/2;
            size_t i_min = m-win_height/2;
            size_t i_max = m + win_height/2;

            if( pcl::isFinite(bordersImage->at(i_min,j_min)) && pcl::isFinite(bordersImage->at(i_max,j_min)) && pcl::isFinite(bordersImage->at(m,j_min)) &&
                    pcl::isFinite(bordersImage->at(m,j_max))
                    && pcl::isFinite(bordersImage->at(i_min,j_max)) && pcl::isFinite(bordersImage->at(i_max,j_max)) && pcl::isFinite(bordersImage->at(i_max,n))
                    && pcl::isFinite(bordersImage->at(i_min,n)) /*
                            && cloud->at(i_min,j_min).z != 0 && cloud->at(i_max,j_min).z != 0 && cloud->at(m,j_min).z != 0
                            && cloud->at(m,j_max).z != 0 && cloud->at(i_min,j_max).z !=0 && cloud->at(i_max,j_max).z != 0 &&
                            cloud->at(i_max,n).z != 0 && cloud->at(i_min,n).z != 0 */) {


                //horizontal diff mask
                float distH = std::abs(bordersImage->at(i_min,j_min).z - bordersImage->at(i_min,j_max).z);
                distH += 2*std::abs(bordersImage->at(m,j_min).z - bordersImage->at(m,j_max).z);
                distH += std::abs(bordersImage->at(i_max,j_min).z - bordersImage->at(i_max,j_max).z);

                //std::cout <<cloud->at(i_min,j_min).z << "\n";

                //vertical diff mask
                float distV = std::abs(bordersImage->at(i_min,j_min).z - bordersImage->at(i_max,j_min).z);
                distV += 2*std::abs(bordersImage->at(i_min,n).z - bordersImage->at(i_max,n).z);
                distV += std::abs(bordersImage->at(i_min,j_max).z - bordersImage->at(i_max,j_max).z);

                float distG = std::sqrt( distH*distH + distV*distV );

                //const float thresh = 0.1;
                const float thresh = 0.2;

                if( rgbBorder.at<uchar>(n,m) > 150 ) {

                    imgBorder.at<uchar>(n,m) = 255;

                }

            }
        }
    }

//    static std::string id("a");
//    std::string name=std::string("sobel_img")+id+std::string(".jpg");
//    imwrite(name.c_str(),imgBorder);
//    if( id=="a") id="b";
//    else if( id=="b") id="a";

    return imgBorder;
}

template <typename PointT>
cv::Mat SobelFilter<PointT>::getPoints(cv::Mat bordersImage)
{

    cv::Point2i point;


    int numPoints=0;
    for(int y=0; y < bordersImage.rows; y++) {
        for(int x=0; x < bordersImage.cols; x++ ) {
            if( bordersImage.at<uchar>(y,x) > 0 ) {
                numPoints++;
            }
        }
    }
    int num=0;
    cv::Mat pointMat(numPoints,2,CV_32SC1);
    for(int y=0; y < bordersImage.rows; y++) {
        for(int x=0; x < bordersImage.cols; x++ ) {
            if( bordersImage.at<uchar>(y,x) > 0 ) {
                pointMat.at<int32_t>(num,0) = x;
                pointMat.at<int32_t>(num,1) = y;
                num++;
            }
        }
    }

    return pointMat;
}

inline bool hasSomeNeighBoor(const cv::Mat& img,int row, int col, int dist) {
    if( img.at<uchar>(row+dist,col) > 150 ) return true;
    if( img.at<uchar>(row-dist,col) > 150 ) return true;
    if( img.at<uchar>(row,col+dist) > 150 ) return true;
    if( img.at<uchar>(row,col-dist) > 150 ) return true;
    if( img.at<uchar>(row+dist,col+dist) > 150 ) return true;
    if( img.at<uchar>(row+dist,col-dist) > 150 ) return true;
    if( img.at<uchar>(row-dist,col+dist) > 150 ) return true;
    if( img.at<uchar>(row-dist,col-dist) > 150 ) return true;

    return false;
}

inline std::vector<cv::Point2i> generateNeighBoor(const cv::Mat& bordersImage,const cv::Mat& visitedImage, int row, int col) {

    //point.x=row, point.y=column
    std::vector<cv::Point2i> toVisit;
    const int dist = 1;
    if( bordersImage.at<uchar>(row+dist,col) > 150 && visitedImage.at<uchar>(row+dist,col) == 0 ) toVisit.push_back(cv::Point2i(row+dist,col));
    if( bordersImage.at<uchar>(row-dist,col) > 150 && visitedImage.at<uchar>(row-dist,col) == 0 ) toVisit.push_back(cv::Point2i(row-dist,col));
    if( bordersImage.at<uchar>(row,col+dist) > 150 && visitedImage.at<uchar>(row,col+dist) == 0 ) toVisit.push_back(cv::Point2i(row,col+dist));
    if( bordersImage.at<uchar>(row,col-dist) > 150 && visitedImage.at<uchar>(row,col-dist) == 0 ) toVisit.push_back(cv::Point2i(row,col-dist));
    if( bordersImage.at<uchar>(row+dist,col+dist) > 150 && visitedImage.at<uchar>(row+dist,col+dist) == 0 ) toVisit.push_back(cv::Point2i(row+dist,col+dist));
    if( bordersImage.at<uchar>(row+dist,col-dist) > 150 && visitedImage.at<uchar>(row+dist,col-dist) == 0 ) toVisit.push_back(cv::Point2i(row+dist,col-dist));
    if( bordersImage.at<uchar>(row-dist,col+dist) > 150 && visitedImage.at<uchar>(row-dist,col+dist) == 0 ) toVisit.push_back(cv::Point2i(row-dist,col+dist));
    if( bordersImage.at<uchar>(row-dist,col-dist) > 150 && visitedImage.at<uchar>(row-dist,col-dist) == 0 ) toVisit.push_back(cv::Point2i(row-dist,col-dist));

    return toVisit;

}


inline void visitNeighBoor(std::vector<cv::Point2i> neighBoor, const cv::Mat& bordersImage,cv::Mat& visitedImage,cv::Mat& localImage, int& count ) {
    for( int k=0; k<neighBoor.size(); k++ ) {
        visitedImage.at<uchar>(neighBoor.at(k).x,neighBoor.at(k).y) = 255;
        localImage.at<uchar>(neighBoor.at(k).x,neighBoor.at(k).y) = 255;
        count++;
        std::vector<cv::Point2i> toVisit = generateNeighBoor(bordersImage,visitedImage,neighBoor.at(k).x,neighBoor.at(k).y);
        visitNeighBoor(toVisit,bordersImage,visitedImage,localImage,count);
    }
}

template <typename PointT>
void SobelFilter<PointT>::removeNoise(cv::Mat& bordersImage) {

    cv::Mat visitedImage=cv::Mat::zeros(bordersImage.rows,bordersImage.cols,bordersImage.type());
    cv::Mat finalImage=cv::Mat::zeros(bordersImage.rows,bordersImage.cols,bordersImage.type());


    for(int row=0; row < bordersImage.rows; row++) {
        for(int col=0; col < bordersImage.cols; col++) {

            if( bordersImage.at<uchar>(row,col) > 150 && visitedImage.at<uchar>(row,col) == 0 ) {

                int count=0;
                cv::Mat localImage=cv::Mat::zeros(bordersImage.rows,bordersImage.cols,bordersImage.type());

                visitedImage.at<uchar>(row,col) = 255;
                localImage.at<uchar>(row,col) = 255;
                count++;
                std::vector<cv::Point2i> toVisit = generateNeighBoor(bordersImage,visitedImage,row,col);
                visitNeighBoor(toVisit,bordersImage,visitedImage,localImage,count);

                if( count > 100 ) {
                    cv::bitwise_or(localImage,finalImage,finalImage);
                }
            }
        }
    }

    bordersImage = finalImage;
}

template <typename PointT>
void SobelFilter<PointT>::setCloudAsNaN(pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
    for( size_t m=0; m < cloud.width;m++ ) {
        for( size_t n=0; n < cloud.height; n++) {
            cloud(m,n).x=cloud(m,n).y=cloud(m,n).z = std::numeric_limits<float>::quiet_NaN();
        }
    }
}

template <typename PointT> void
SobelFilter<PointT>::setSourceCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source)
{
    sourceCloud = source;
}

template <typename PointT> void
SobelFilter<PointT>::setTargetCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target)
{
    targetCloud = target;
}

template <typename PointT> cv::Mat
SobelFilter<PointT>::runICP(cv::Mat src, cv::Mat tgt, int maxIter,std::vector<int>& pairs, std::vector<float>& dists) {


    cv::Mat R = cv::Mat::eye(2,2,CV_32F);
    cv::Scalar T(0,0,0,0);
    cv::Mat srcMoved = src.clone();
    float prevDist=0;
    for(int k=0; k < maxIter; k++) {

        pairs.clear();
        dists.clear();

        //apply ICP to find best R,t between source and target points
        int dist = flann_knn(srcMoved,tgt,pairs,dists); //get closest points!
        if( dist == prevDist ) {
            std::cout << "breakin at " << k << " iteration\n";
            break;
        }
        prevDist = dist;
        //fill corresp with closest points from src to tgt
        //target point i matches with pairs[i] point
        cv::Mat corresp(tgt.size(),tgt.type());
        for(int i=0;i<tgt.rows;i++) {
            cv::Point p = srcMoved.at<cv::Point>(pairs[i],0);
            corresp.at<cv::Point>(i,0) = p;

        }

        cv::Mat Rlocal;
        cv::Scalar Tlocal;
        //get R,t
        findTransformSVD(corresp,tgt,Rlocal,Tlocal);

        R = Rlocal*R;
        T = T + Tlocal;
        //apply R,t to srcMoved
        for(int i=0;i<srcMoved.rows;i++) {
            //rotate point
            srcMoved.at<cv::Point>(i,0).x = Rlocal.at<float>(0,0)*srcMoved.at<cv::Point>(i,0).x + Rlocal.at<float>(0,1)*srcMoved.at<cv::Point>(i,0).y;
            srcMoved.at<cv::Point>(i,0).y = Rlocal.at<float>(1,0)*srcMoved.at<cv::Point>(i,0).x + Rlocal.at<float>(1,1)*srcMoved.at<cv::Point>(i,0).y;
            //translate point
            srcMoved.at<cv::Point>(i,0).x += Tlocal(0);
            srcMoved.at<cv::Point>(i,0).y += Tlocal(1);
        }
    }


    cv::Mat Affine(2,3,CV_32F);
    //put rotation and translation inside Affine matrix
    Affine.at<float>(0,0) = R.at<float>(0,0);
    Affine.at<float>(0,1) = R.at<float>(0,1);
    Affine.at<float>(1,0) = R.at<float>(1,0);
    Affine.at<float>(1,1) = R.at<float>(1,1);
    Affine.at<float>(0,2) = T(0);
    Affine.at<float>(1,2) = T(1);
    return Affine;
}

template <typename PointT> void
SobelFilter<PointT>::applyFilter(pcl::PointCloud<pcl::PointXYZRGB> &sourceFiltered, pcl::PointCloud<pcl::PointXYZRGB>&targetFiltered)
{
    setCloudAsNaN(sourceFiltered);
    setCloudAsNaN(targetFiltered);
    //apply sobel filter to RGB image and get only points with DEPTH not null at depthmap
    cv::Mat sourceBorders = getBorders(sourceCloud);
    cv::Mat targetBorders = getBorders(targetCloud);
    cv::Mat transf = cv::estimateRigidTransform(sourceBorders,targetBorders,false);

    if( transf.rows == 0 ) {

        for(int m=0; m < sourceBorders.rows; m++) {
            for(int n=0; n < sourceBorders.cols; n++) {

                if( sourceBorders.at<uchar>(m,n) > 150 ) {
                    sourceFiltered(n,m) = sourceCloud->at(n,m);
                }
                if( targetBorders.at<uchar>(m,n) > 150 ) {
                    targetFiltered(n,m) = targetCloud->at(n,m);
                }

            }
        }

    } else {


        cv::Mat movedSource;
        cv::Mat intersectionTarget;
        cv::Mat intersectionSource;
        //apply rotation and translation ot image
        cv::warpAffine(sourceBorders, movedSource, transf, cv::Size(sourceBorders.cols,sourceBorders.rows));
        cv::bitwise_and(movedSource,targetBorders,intersectionTarget);

        static bool writeImage=true;
        if( writeImage )
            cv::imwrite("before_filter.jpg",intersectionTarget);

        removeNoise(intersectionTarget);


        if( writeImage ){
            cv::imwrite("after_filter.jpg",intersectionTarget);
            writeImage = false;
        }
        cv::Mat AffineInv;
        cv::invertAffineTransform(transf,AffineInv);
        cv::warpAffine(intersectionTarget, intersectionSource, AffineInv, cv::Size(sourceBorders.cols,sourceBorders.rows));

        for(int m=0; m < intersectionSource.rows; m++) {
            for(int n=0; n < intersectionSource.cols; n++) {

                if( intersectionSource.at<uchar>(m,n) > 150 ) {
                    sourceFiltered(n,m) = sourceCloud->at(n,m);
                }
                if( intersectionTarget.at<uchar>(m,n) > 150 ) {
                    targetFiltered(n,m) = targetCloud->at(n,m);
                }

            }
        }


//        static std::string id("a");
//        std::string name=std::string("rotated")+id+std::string(".jpg");
//        //    imwrite(name.c_str(),outMat);
//        std::string name2=std::string("finalSrc")+id+std::string(".jpg");
//        std::string name3=std::string("finalTgt")+id+std::string(".jpg");
//        std::string name4=std::string("sobelSrc")+id+std::string(".jpg");
//        std::string name5=std::string("sobelTgt")+id+std::string(".jpg");
//        std::string name6=std::string("rotated22")+id+std::string(".jpg");
//        imwrite(name.c_str(),movedSource);
//        imwrite(name2.c_str(),intersectionSource);
//        imwrite(name3.c_str(),intersectionTarget);
//        imwrite(name4.c_str(),sourceBorders);
//        imwrite(name5.c_str(),targetBorders);
//        if( id=="a") id="b";
//        else if( id=="b") id="a";

    }
}
#endif // PCL_FILTERS_BILATERAL_H_
