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
#include "opencv2/opencv.hpp"
#include "oflow_pcl.h"

#define MIN3(x,y,z)  ((y) <= (z) ? \
    ((x) <= (y) ? (x) : (y)) \
    : \
    ((x) <= (z) ? (x) : (z)))

#define MAX3(x,y,z)  ((y) >= (z) ? \
    ((x) >= (y) ? (x) : (y)) \
    : \
    ((x) >= (z) ? (x) : (z)))



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

    /** \brief The bilateral filter Gaussian distance kernel.
    * \param[in] x the spatial distance (distance or intensity)
    * \param[in] sigma standard deviation
    */
    inline double
    kernel (double x, double sigma)
    {
        return (exp (- (x*x)/(2*sigma*sigma)));
    }

    cv::Mat getImageBorders();

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
SobelFilter<PointT>::getImageBorders() {

    cv::Mat imgColor(480,640,CV_8UC3);
    cloudToMat(input_->makeShared(),imgColor);

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
     imwrite("sobel_color.jpg",grad);

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
    cv::Mat rgbBorder = getImageBorders();

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
    imwrite("SOBEL_IMAGE.jpg",imgBorder);
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
#endif // PCL_FILTERS_BILATERAL_H_
