#include "Surf.h"

Surf::Surf()
{
}

/** uses SURF for comparison of two clouds, saves the feature distance in featureDist parameter and detected points pixel distance in pixelDist paramter **/
void Surf::visualDistance( const int indexA, const int indexB,const pcl::PointCloud<pcl::PointXYZRGB>& cloudA,const pcl::PointCloud<pcl::PointXYZRGB>& cloudB, float& featureDist, int& pixelDist ) {

    using namespace cv;
    //calculate descriptors if we dont have them
    if( cloudKeyPoints.find(indexA) == cloudKeyPoints.end() ) {
        saveCloudDescriptors(indexA,cloudA);
    }
    if( cloudKeyPoints.find(indexB) == cloudKeyPoints.end() ) {
        saveCloudDescriptors(indexB,cloudB);
    }

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    keypoints_1 = cloudKeyPoints[indexA];

    keypoints_2 = cloudKeyPoints[indexB];

    Mat descriptors_1, descriptors_2;
    descriptors_1 = cloudDescriptors[indexA];
    descriptors_2 = cloudDescriptors[indexB];

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;

    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;
    double mean_dist = 0;
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
        mean_dist += dist;

    }

    mean_dist = mean_dist/descriptors_1.rows;

    featureDist = mean_dist;

    //calculation of distance in pixels between best matches
    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        if( matches[i].distance <= max(2*min_dist, 0.02) )
        { good_matches.push_back( matches[i]); }
    }

    int img_dist=0;
    for( int m =0; m < good_matches.size(); m++) {
        int d=0;
        int m1 = good_matches.at(m).queryIdx;
        int m2 = good_matches.at(m).trainIdx;

        d += (keypoints_1.at(m1).pt.x - keypoints_2.at(m2).pt.x)*(keypoints_1.at(m1).pt.x - keypoints_2.at(m2).pt.x);
        d += (keypoints_1.at(m1).pt.y - keypoints_2.at(m2).pt.y)*(keypoints_1.at(m1).pt.y - keypoints_2.at(m2).pt.y);
        img_dist += std::sqrt(d);
    }

    img_dist = img_dist/good_matches.size();

    pixelDist = img_dist;
}

Eigen::Matrix4f Surf::getSurfTransform(const int indexA, const int indexB, const pcl::PointCloud<pcl::PointXYZRGB> &cloudA, const pcl::PointCloud<pcl::PointXYZRGB> &cloudB)
{
    using namespace cv;
    //calculate descriptors if we dont have them
    if( cloudKeyPoints.find(indexA) == cloudKeyPoints.end() ) {
        saveCloudDescriptors(indexA,cloudA);
    }
    if( cloudKeyPoints.find(indexB) == cloudKeyPoints.end() ) {
        saveCloudDescriptors(indexB,cloudB);
    }

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    keypoints_1 = cloudKeyPoints[indexA];

    keypoints_2 = cloudKeyPoints[indexB];

    Mat descriptors_1, descriptors_2;
    descriptors_1 = cloudDescriptors[indexA];
    descriptors_2 = cloudDescriptors[indexB];

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;

    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;


    }

    //calculation of distance in pixels between best matches
    std::vector< DMatch > good_matches;
    float factor = 1.0;
    for(int k=0; k<12;k++) {

        for( int i = 0; i < descriptors_1.rows; i++ )
        {
            if( matches[i].distance <= max(factor*min_dist, 0.02) )
            { good_matches.push_back( matches[i]); }
        }
        factor = factor + 0.5;
        if( good_matches.size() > 20 ) {
            break;
        } else {
            good_matches.clear();
        }
    }

    pcl::PointCloud<pcl::PointXYZ> cornersCloudA;
    pcl::PointCloud<pcl::PointXYZ> cornersCloudB;
    Eigen::Matrix4f transfMat = Eigen::Matrix4f::Identity();

    for( int m =0; m < good_matches.size(); m++) {
        int d=0;
        int m1 = good_matches.at(m).queryIdx;
        int m2 = good_matches.at(m).trainIdx;

        cv::Point p0( round( keypoints_1.at(m1).pt.x ), round( keypoints_1.at(m1).pt.y ) );
        cv::Point p1( round( keypoints_2.at(m2).pt.x ), round( keypoints_2.at(m2).pt.y ) );
        if( p0.x > 639 ) p0.x = 639;
        if( p0.y > 479 ) p0.y = 479;
        if( p0.x < 0 )   p0.x = 0;
        if( p0.y < 0 )   p0.y = 0;
        if( p1.x > 639 ) p1.x = 639;
        if( p1.y > 479 ) p1.y = 479;
        if( p1.x < 0 )   p1.x = 0;
        if( p1.y < 0 )   p1.y = 0;
        //std::cout << p0.x << " " << p0.y << " p1: " << p1.x << " " << p1.y << "\n";
        if( pcl::isFinite(cloudA(p0.x,p0.y)) && pcl::isFinite(cloudB(p1.x,p1.y)) ) {
            pcl::PointXYZ pointA;
            pcl::PointXYZ pointB;
            pointA.x = cloudA(p0.x,p0.y).x;
            pointA.y = cloudA(p0.x,p0.y).y;
            pointA.z = cloudA(p0.x,p0.y).z;
            pointB.x = cloudB(p1.x,p1.y).x;
            pointB.y = cloudB(p1.x,p1.y).y;
            pointB.z = cloudB(p1.x,p1.y).z;
            if( pointA.z > 0.1 && pointB.z > 0.1 ) {
                cornersCloudA.push_back( pointA );
                cornersCloudB.push_back( pointB );
            }

        }

    }

    pcl::Correspondences corrVec;
    for(int j=0; j < cornersCloudA.size() ; j++) {

        float dist = cornersCloudA[j].x - cornersCloudB[j].x;
        dist = dist*dist;
        dist = dist + (cornersCloudA[j].y - cornersCloudB[j].y)*(cornersCloudA[j].y - cornersCloudB[j].y);
        dist = dist + (cornersCloudA[j].z - cornersCloudB[j].z)*(cornersCloudA[j].z - cornersCloudB[j].z);
        corrVec.push_back(pcl::Correspondence(j,j,dist));
    }

    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ,pcl::PointXYZ,float_t> tEst;

    if(cornersCloudA.size() > 2) {
        tEst.estimateRigidTransformation(cornersCloudA,cornersCloudB,corrVec,transfMat);

    }

    return transfMat;

}

/** save keypoints and descriptors of each cloud in two std::maps */
void Surf::saveCloudDescriptors(int cloudIndex, const pcl::PointCloud<pcl::PointXYZRGB>& cloudA) {

    using namespace cv;
    cv::Mat imgAcolor(480,640,CV_8UC3);
    cloudToMat(cloudA.makeShared(),imgAcolor);

    cv::Mat img_1(480,640,CV_8UC1);
    cv::cvtColor(imgAcolor,img_1,CV_BGR2GRAY);

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    SurfFeatureDetector detector( minHessian );
    detector.upright = true;


    std::vector<KeyPoint> keypoints_1;
    detector.detect( img_1, keypoints_1 );

    cloudKeyPoints[cloudIndex] = keypoints_1;

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;
    Mat descriptors_1;
    extractor.compute( img_1, keypoints_1, descriptors_1 );

    cloudDescriptors[cloudIndex] = descriptors_1;
}
