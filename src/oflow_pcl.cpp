#include <iostream>
#include <vector>
#include <cmath>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/point_types.h>
#include "opencv2/opencv.hpp"
#include "oflow_pcl.h"
#include "utils.h"



bool pointExists(const pcl::PointCloud<pcl::PointXYZ>& cornersCloudA, const pcl::PointXYZ& point) {
    for(int p=0; p < cornersCloudA.size(); p++) {
        if( cornersCloudA[p].x == point.x && cornersCloudA[p].y == point.y && cornersCloudA[p].z == point.z) {
            return true;
        }
    }
    return false;
}

/** let points closests to median at first positions **/
void sortCloud( pcl::PointCloud<pcl::PointXYZ>& cornersCloudA,  pcl::PointCloud<pcl::PointXYZ>& cornersCloudB,
                pcl::PointXYZ medianDir) {

    pcl::PointXYZ direc;
    float prevDist;
    bool notOrdered;

    do {
        notOrdered=false;
        for(int p=0; p < cornersCloudA.size(); p++) {

            //current direction (vector between two matching points)
            direc.x = cornersCloudB[p].x - cornersCloudA[p].x;
            direc.y = cornersCloudB[p].y - cornersCloudA[p].y;
            direc.z = cornersCloudB[p].z - cornersCloudA[p].z;

            //distance to median direction
            float dist = (direc.x - medianDir.x)*(direc.x - medianDir.x);
            dist = dist + (direc.y - medianDir.y)*(direc.y - medianDir.y);
            dist = dist + (direc.z - medianDir.z)*(direc.z - medianDir.z);

            //swap if not ordered
            if(p != 0 && dist < prevDist) {
                direc = cornersCloudA[p-1]; //direc just used as temp
                cornersCloudA[p-1] = cornersCloudA[p];
                cornersCloudA[p]   = direc;
                direc = cornersCloudB[p-1]; //direc just used as temp
                cornersCloudB[p-1] = cornersCloudB[p];
                cornersCloudB[p]   = direc;
                notOrdered = true;
            }
            prevDist = dist;
        }

    } while(notOrdered);

}
/** let closest vectors at first positions **/
void sortCloud( pcl::PointCloud<pcl::PointXYZ>& cornersCloudA,  pcl::PointCloud<pcl::PointXYZ>& cornersCloudB) {

    pcl::PointXYZ direc;
    float prevDist;
    bool notOrdered;

    do {
        notOrdered=false;
        for(int p=0; p < cornersCloudA.size(); p++) {

            //current direction (vector between two matching points)
            direc.x = cornersCloudB[p].x - cornersCloudA[p].x;
            direc.y = cornersCloudB[p].y - cornersCloudA[p].y;
            direc.z = cornersCloudB[p].z - cornersCloudA[p].z;

            //distance between matching points
            float dist = direc.x*direc.x + direc.y*direc.y + direc.z*direc.z;

            //swap if not ordered
            if(p != 0 && dist < prevDist) {
                direc = cornersCloudA[p-1]; //direc just used as temp
                cornersCloudA[p-1] = cornersCloudA[p];
                cornersCloudA[p]   = direc;
                direc = cornersCloudB[p-1]; //direc just used as temp
                cornersCloudB[p-1] = cornersCloudB[p];
                cornersCloudB[p]   = direc;
                notOrdered = true;
            }
            prevDist = dist;
        }

    } while(notOrdered);

}


Eigen::Matrix4f getOflow3Dtransf(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudA, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudB, float maxCorrespDist) {

    cv::Mat imgAcolor(480,640,CV_8UC3);
    cloudToMat(cloudA,imgAcolor);
    cv::Mat imgBcolor(480,640,CV_8UC3);
    cloudToMat(cloudB,imgBcolor);
    cv::Mat imgA(480,640,CV_8UC1);
    cv::Mat imgB(480,640,CV_8UC1);
    cv::Mat imgC = cv::Mat::zeros(480,640,CV_8UC1);
    cv::cvtColor(imgAcolor,imgA,CV_BGR2GRAY);
    cv::cvtColor(imgBcolor,imgB,CV_BGR2GRAY);


    int win_size = 25;
    int maxCorners = 40;
    double qualityLevel = 0.01;
    double minDistance = 5.0;

    std::vector<cv::Point2f> cornersA;
    cornersA.reserve(maxCorners);
    std::vector<cv::Point2f> cornersB;
    cornersB.reserve(maxCorners);

    goodFeaturesToTrack( imgA,cornersA,maxCorners,qualityLevel,minDistance,cv::Mat());
    goodFeaturesToTrack( imgB,cornersB,maxCorners,qualityLevel,minDistance,cv::Mat());

    cornerSubPix( imgA, cornersA, cv::Size( win_size, win_size ), cv::Size( -1, -1 ),
            cv::TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );

    cornerSubPix( imgB, cornersB, cv::Size( win_size, win_size ), cv::Size( -1, -1 ),
            cv::TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );

    // Call Lucas Kanade algorithm

    std::vector<uchar> features_found;
    features_found.reserve(maxCorners);
    std::vector<float> feature_errors;
    feature_errors.reserve(maxCorners);

    calcOpticalFlowPyrLK( imgA, imgB, cornersA, cornersB, features_found, feature_errors ,
            cv::Size( win_size, win_size ), 3,
            cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0 );


    //loop features found in both images and get one cloud of points per image
    pcl::PointCloud<pcl::PointXYZ> cornersCloudA;
    pcl::PointCloud<pcl::PointXYZ> cornersCloudB;
    std::vector<float> dirX;
    std::vector<float> dirY;
    std::vector<float> dirZ;
    cornersCloudA.reserve(30);
    cornersCloudB.reserve(30);
    dirX.reserve(30);
    dirY.reserve(30);
    dirZ.reserve(30);
    Eigen::Matrix4f transfMat = Eigen::Matrix4f::Identity();

    for( int i=0; i < features_found.size(); i++ ){

        if( feature_errors[i] < 60) {

            cv::Point p0( round( cornersA[i].x ), round( cornersA[i].y ) );
            cv::Point p1( round( cornersB[i].x ), round( cornersB[i].y ) );
            if( p0.x > 639 ) p0.x = 639;
            if( p0.y > 479 ) p0.y = 479;
            if( p0.x < 0 )   p0.x = 0;
            if( p0.y < 0 )   p0.y = 0;
            if( p1.x > 639 ) p1.x = 639;
            if( p1.y > 479 ) p1.y = 479;
            if( p1.x < 0 )   p1.x = 0;
            if( p1.y < 0 )   p1.y = 0;
            //std::cout << p0.x << " " << p0.y << " p1: " << p1.x << " " << p1.y << "\n";
            if( pcl::isFinite((*cloudA)(p0.x,p0.y)) && pcl::isFinite((*cloudB)(p1.x,p1.y)) ) {

                line( imgC, p0, p1, CV_RGB(255,255,255), 2 );
                line( imgAcolor,p0,p0,CV_RGB((i*30)%255,(i*45+77)%255,(i*75+17)%255), 2);
                line( imgBcolor,p1,p1,CV_RGB((i*30)%255,(i*45+77)%255,(i*75+17)%255), 2);
                pcl::PointXYZ pointA;
                pcl::PointXYZ pointB;
                pcl::PointXYZ dir;
                pointA.x = (*cloudA)(p0.x,p0.y).x;
                pointA.y = (*cloudA)(p0.x,p0.y).y;
                pointA.z = (*cloudA)(p0.x,p0.y).z;
                pointB.x = (*cloudB)(p1.x,p1.y).x;
                pointB.y = (*cloudB)(p1.x,p1.y).y;
                pointB.z = (*cloudB)(p1.x,p1.y).z;
                if( pointExists(cornersCloudA,pointA) == false && pointExists(cornersCloudB,pointB) == false ) {


                    dir.x = pointB.x - pointA.x;
                    dir.y = pointB.y - pointA.y;
                    dir.z = pointB.z - pointA.z;

                    float dist = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
                    dist = std::sqrt(dist);
                    if( dist > maxCorrespDist /** 20 sep: 0.05*/ ) {
                        //std::cout << "skiping correspondence in oflow, with dist:" << dist << "\n";
                        continue;
                    }
                    cornersCloudA.push_back( pointA );
                    cornersCloudB.push_back( pointB );

                    dirX.push_back(dir.x);
                    dirY.push_back(dir.y);
                    dirZ.push_back(dir.z);

                }

            } else {
                std::cout << "not finite\n\n";
            }
        }
    }

    /** //Save optical flow images
    static int number=0;
    number++;
    std::stringstream str1;
    str1 << number;
    std::string name = "oflow" + str1.str() + ".jpg";
    imwrite(name.c_str(),imgAcolor);
    ++number;
    std::stringstream str2;
    str2 << number;
    name = "oflow" + str2.str() + ".jpg";
    imwrite(name.c_str(),imgBcolor);
    ++number;
    std::stringstream str3;
    str3 << number;
    name = "oflow" + str3.str() + ".jpg";
    imwrite(name.c_str(),imgC);
    /**/



    sortCloud(cornersCloudA,cornersCloudB);

    pcl::PointCloud<pcl::PointXYZ> finalCornersA;
    pcl::PointCloud<pcl::PointXYZ> finalCornersB;
    pcl::Correspondences corrVec;
    int maxUsedCorners = cornersCloudA.size();

    if( maxUsedCorners > 4 ) {
        maxUsedCorners = 4;
    }

    for(int j=0; j < cornersCloudA.size(); j++) {

        float dist = cornersCloudA[j].x - cornersCloudB[j].x;
        dist = dist*dist;
        dist = dist + (cornersCloudA[j].y - cornersCloudB[j].y)*(cornersCloudA[j].y - cornersCloudB[j].y);
        dist = dist + (cornersCloudA[j].z - cornersCloudB[j].z)*(cornersCloudA[j].z - cornersCloudB[j].z);
        finalCornersA.push_back( cornersCloudA[j] );
        finalCornersB.push_back( cornersCloudB[j] );
        corrVec.push_back(pcl::Correspondence(j,j,dist));
    }
    /**/


    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ,pcl::PointXYZ,float_t> tEst;

    if( cornersCloudA.size() > 2) {


        tEst.estimateRigidTransformation(cornersCloudA,cornersCloudB,corrVec,transfMat);

    }

    return transfMat;
}

