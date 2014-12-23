#include "EdgeFilter.h"

EdgeFilter::EdgeFilter() : win_width(3),
win_height(3)
{
}

void EdgeFilter::setSourceCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source)
{
    sourceCloud = source;
}

void EdgeFilter::setTargetCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target)
{
    targetCloud = target;
}

void EdgeFilter::applyFilter(pcl::PointCloud<pcl::PointXYZRGB> &sourceFiltered,
                             pcl::PointCloud<pcl::PointXYZRGB>&targetFiltered, int sobelThreshold)
{
    setCloudAsNaN(sourceFiltered);
    setCloudAsNaN(targetFiltered);
    //apply sobel filter to RGB image and get only points with DEPTH not null at depthmap
    cv::Mat sourceBorders = getBorders(sourceCloud,sobelThreshold);
    cv::Mat targetBorders = getBorders(targetCloud,sobelThreshold);
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

    removeFarPoints(sourceFiltered,2);
    removeFarPoints(targetFiltered,2);

}


cv::Mat EdgeFilter::getSobelBorders(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,int sobelThreshold) {
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

//    Mat txgrad;
//    threshold(grad_x,txgrad,75,255,THRESH_BINARY);
//    pcl::PointCloud<pcl::PointXYZRGB> gradxCloud(640,480);
//    setCloudAsNaN(gradxCloud);
//    filterMask(*cloud,txgrad,gradxCloud);

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( imgGray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_CONSTANT );
    convertScaleAbs( grad_y, abs_grad_y );

//    Mat tygrad;
//    threshold(grad_y,tygrad,75,255,THRESH_BINARY);
//    pcl::PointCloud<pcl::PointXYZRGB> gradyCloud(640,480);
//    setCloudAsNaN(gradyCloud);
//    filterMask(*cloud,tygrad,gradyCloud);

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    threshold(grad,grad,sobelThreshold,255,THRESH_BINARY);

//    pcl::PointCloud<pcl::PointXYZRGB> gradCloud(640,480);
//    setCloudAsNaN(gradCloud);
//    filterMask(*cloud,grad,gradCloud);


//    static std::string id("a");
//    std::string namex=std::string("gradx")+id+std::string(".pcd");
//    std::string namey=std::string("grady")+id+std::string(".pcd");
//    std::string name=std::string("grad")+id+std::string(".pcd");
//    if( id=="a") id="b";

//    pcl::io::savePCDFileASCII(namex,gradxCloud);
//    pcl::io::savePCDFileASCII(namey,gradyCloud);
//    pcl::io::savePCDFileASCII(name,gradCloud);


    return grad;
}

cv::Mat EdgeFilter::getCannyBorders(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) {
    using namespace cv;
    cv::Mat imgColor(480,640,CV_8UC3);
    cloudToMat(cloud->makeShared(),imgColor);

    cv::Mat imgGray(480,640,CV_8UC1);
    cv::cvtColor(imgColor,imgGray,CV_BGR2GRAY);
    /// Global variables
    Mat detected_edges;

    int lowThreshold=75;
    int ratio = 3;
    int kernel_size = 3;

    /// Reduce noise with a kernel 3x3
    blur( imgGray, detected_edges, Size(3,3) );

     /// Canny detector
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    threshold(detected_edges,detected_edges,75,255,THRESH_BINARY);

    return detected_edges;
}

cv::Mat EdgeFilter::getBorders(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& bordersImage,int sobelThreshold)
{
    cv::Mat imgBorder = cv::Mat::zeros(480,640,CV_8UC1);
    cv::Mat rgbBorder = getSobelBorders(bordersImage, sobelThreshold);

    for(size_t m=win_width/2+1; m < (bordersImage->width-win_width/2-1); m++) {
        for(size_t n=win_height/2+1; n < (bordersImage->height-win_height/2-1); n++) {


            size_t j_min = n-win_width/2;
            size_t j_max = n + win_width/2;
            size_t i_min = m-win_height/2;
            size_t i_max = m + win_height/2;

            if( bordersImage->at(i_min,j_min).z > 0.1 && bordersImage->at(i_max,j_min).z > 0.1 && bordersImage->at(m,j_min).z > 0.1 &&
                    bordersImage->at(m,j_max).z > 0.1
                    && bordersImage->at(i_min,j_max).z > 0.1 && bordersImage->at(i_max,j_max).z > 0.1 && bordersImage->at(i_max,n).z > 0.1
                    && bordersImage->at(i_min,n).z > 0.1 && bordersImage->at(m,n).z > 0.1 ) {

                if( pcl::isFinite(bordersImage->at(i_min,j_min) ) == false ) std::cout << "NOT FINITE PASSED\n";




                if( rgbBorder.at<uchar>(n,m) > 150 ) {

                    float zMean = bordersImage->at(i_min,j_min).z + bordersImage->at(i_max,j_min).z + bordersImage->at(m,j_min).z;
                    zMean += bordersImage->at(m,j_max).z + bordersImage->at(i_min,j_max).z + bordersImage->at(i_max,j_max).z;
                    zMean += bordersImage->at(i_max,n).z + bordersImage->at(i_min,n).z;

                    if( bordersImage->at(i_min,j_min).z < zMean ) imgBorder.at<uchar>(j_min,i_min) = 255;
                    if( bordersImage->at(i_max,j_min).z < zMean ) imgBorder.at<uchar>(j_min,i_max) = 255;
                    if( bordersImage->at(m,j_min).z < zMean ) imgBorder.at<uchar>(j_min,m) = 255;
                    if( bordersImage->at(m,j_max).z < zMean ) imgBorder.at<uchar>(j_max,m) = 255;
                    if( bordersImage->at(i_max,j_max).z < zMean ) imgBorder.at<uchar>(j_max,i_max) = 255;
                    if( bordersImage->at(i_max,n).z < zMean ) imgBorder.at<uchar>(n,i_max) = 255;
                    if( bordersImage->at(i_min,n).z < zMean ) imgBorder.at<uchar>(n,i_min) = 255;
                    if( bordersImage->at(i_min,j_max).z < zMean ) imgBorder.at<uchar>(j_max,i_min) = 255;

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

void EdgeFilter::removeNoise(cv::Mat& bordersImage) {

    cv::Mat visitedImage=cv::Mat::zeros(bordersImage.rows,bordersImage.cols,bordersImage.type());
    cv::Mat finalImage=cv::Mat::zeros(bordersImage.rows,bordersImage.cols,bordersImage.type());

    //let pass blobs that have more than MIN_NEIGHBOORHOOD conected pixels!
    int MIN_NEIGHBOORHOOD = 1000;
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

                if( count > MIN_NEIGHBOORHOOD ) {
                    cv::bitwise_or(localImage,finalImage,finalImage);
                }
            }
        }
    }

    bordersImage = finalImage;
}
bool EdgeFilter::hasSomeNeighBoor(const cv::Mat& img,int row, int col, int dist) {
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

std::vector<cv::Point2i> EdgeFilter::generateNeighBoor(const cv::Mat& bordersImage,
                                                       const cv::Mat& visitedImage, int row, int col) {

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


void EdgeFilter::visitNeighBoor(std::vector<cv::Point2i> neighBoor, const cv::Mat& bordersImage,
                                cv::Mat& visitedImage,cv::Mat& localImage, int& count ) {
    for( int k=0; k<neighBoor.size(); k++ ) {
        visitedImage.at<uchar>(neighBoor.at(k).x,neighBoor.at(k).y) = 255;
        localImage.at<uchar>(neighBoor.at(k).x,neighBoor.at(k).y) = 255;
        count++;
        std::vector<cv::Point2i> toVisit = generateNeighBoor(bordersImage,visitedImage,neighBoor.at(k).x,neighBoor.at(k).y);
        visitNeighBoor(toVisit,bordersImage,visitedImage,localImage,count);
    }
}
