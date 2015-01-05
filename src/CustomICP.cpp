#include "CustomICP.h"
#include "SobelFilter.h"
#include "time.h"
#include <stdlib.h>
#include <cmath>

CustomICP::CustomICP()
{
    //customCorresp = new CustomCorrespondenceEstimation<pcl::PointXYZRGB,pcl::PointXYZRGB,float>;
    //icp.setCorrespondenceEstimation(
    //boost::shared_ptr<pcl::registration::CorrespondenceEstimation<pcl::PointXYZRGB,pcl::PointXYZRGB,float> > (customCorresp));

    std::cout << "CAAAAA\n\n";
    icp.setMaxCorrespondenceDistance(0.1);
    icp.setMaximumIterations (100);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1e-4);
    icp.setUseReciprocalCorrespondences(false);
    prevTransf = Eigen::Matrix4f::Identity();
    finalTransf = Eigen::Matrix4f::Identity();
    fitness = 1;
    stopIfOflowFails = false;
    oflowFound = false;

}


void CustomICP::setInputSource( pcl::PointCloud<pcl::PointXYZRGB>::Ptr src ) {

    this->src = src;

}

void CustomICP::setInputTarget( pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt ) {
    this->tgt = tgt;

}

inline float photoConsistency(pcl::PointCloud<pcl::PointXYZRGB> &cloudSrc, pcl::PointCloud<pcl::PointXYZRGB> &cloudTgt,Eigen::Matrix4f transf) {
    pcl::PointCloud<pcl::PointXYZRGB> srcMoved(640,480);
    pcl::transformPointCloud(cloudSrc, srcMoved, transf);

    float fx = 525.0;  // focal length x
    float fy = 525.0;  // focal length y
    float cx = 319.5;  // optical center x
    float cy = 239.5;  // optical center y

    float factor = 5000; // for the 16-bit PNG files

    float diff = 0;
    int count=0;
    for(int i=0; i < srcMoved.height; i++) {
        for(int j=0; j < srcMoved.width; j++) {

             if( std::isnan(srcMoved(j,i).x) == false && srcMoved(j,i).z > 0.01) {

                float z = srcMoved(j,i).z;
                float x = srcMoved(j,i).x;
                float y = srcMoved(j,i).y;

                float rowf= (y*fy)/z+cy;
                float colf = (x*fx)/z+cx;
                //aprox to nearest int
                int row = rowf+0.5;
                int col = colf+0.5;

                if( col > 639 ) col = 639;
                if( row > 479 ) row = 479;
                if( col < 0 )   col = 0;
                if( row < 0 )   row = 0;

//                float srcGray =  0.299f * static_cast <float> (srcMoved(j,i).r) +
//                                 0.587f * static_cast <float> (srcMoved(j,i).g) +
//                                 0.114f * static_cast <float> (srcMoved(j,i).b);

//                float tgtGray = 0.299f * static_cast <float> (cloudTgt(col,row).r) +
//                                0.587f * static_cast <float> (cloudTgt(col,row).g) +
//                                0.114f * static_cast <float> (cloudTgt(col,row).b);

                float colorDiff = ( srcMoved(j,i).b - cloudTgt(col,row).b )* ( srcMoved(j,i).b - cloudTgt(col,row).b );
                colorDiff += ( srcMoved(j,i).g - cloudTgt(col,row).g )* ( srcMoved(j,i).g - cloudTgt(col,row).g );
                colorDiff += ( srcMoved(j,i).r - cloudTgt(col,row).r )* ( srcMoved(j,i).r - cloudTgt(col,row).r );

//                float colorDiff = (srcGray - tgtGray)*(srcGray - tgtGray);
                colorDiff = std::sqrt(colorDiff);
                diff += colorDiff;
                count++;
             }

        }
    }

    return diff/count;

}

void CustomICP::align( pcl::PointCloud<pcl::PointXYZRGB> &cloud, Eigen::Matrix4f guess,float max_dist )
{

    //oflowTransf = Eigen::Matrix4f::Identity();
    pcl::PointCloud<pcl::PointXYZRGB> sobTgt(640,480);
    pcl::PointCloud<pcl::PointXYZRGB> sobSrc(640,480);
    pcl::PointCloud<pcl::PointXYZRGB> sobSrcMoved(640,480);


    edgeFilter.setSourceCloud(src);
    edgeFilter.setTargetCloud(tgt);
    edgeFilter.applyFilter(sobSrc,sobTgt,max_dist);

//    static int saveSobel = true;
//    if( saveSobel ) {
//        char name1[100];
//        char name2[100];
//        sprintf(name1,"sobTgt%f.pcd",max_dist);
//        sprintf(name2,"sobSrc%f.pcd",max_dist);
//        pcl::io::savePCDFileASCII(name1,sobTgt);
//        pcl::io::savePCDFileASCII(name2,sobSrc);
//        if( max_dist >= 0.1 )
//            saveSobel=false;
//    }

    /** Generate clouds without NaN points to work with ICP **/
    tgtNonDense.clear();
    srcNonDense.clear();

    for(int k=0; k < sobTgt.size(); k++) {
        if( std::isnan(sobTgt.points[k].x) == false && sobTgt.points[k].z > 0.01) {
            tgtNonDense.push_back(sobTgt.at(k));
        }
    }

    for(int k=0; k < sobSrc.size(); k++) {
        if( std::isnan(sobSrc.points[k].x) == false && sobSrc.points[k].z > 0.01 ) {
            srcNonDense.push_back(sobSrc.at(k));
        }
    }
    //std::cout << "NON DENSE SIZE " << srcNonDense.size() << "\n";
    icp.setInputTarget(tgtNonDense.makeShared());
    icp.setInputSource(srcNonDense.makeShared());
    icp.align(cloud, guess);
    finalTransf = icp.getFinalTransformation();


}

void CustomICP::align(pcl::PointCloud<pcl::PointXYZRGB> &cloud, Eigen::Matrix4f guess)
{


    float max_dist=0.1;
    const int MAX_PHOTOCONS = 40;

    align(cloud,guess,max_dist);
    Eigen::Matrix4f currTransf = getFinalTransformation();

    float currPhotoCons = 1000;
    float currFitness = 1;
    if( currTransf != Eigen::Matrix4f::Identity() ) {
        currPhotoCons = getPhotoConsistency();
        currFitness = getFitnessScore();
    }

    std::cout << "Obtained " << currPhotoCons << "Fitness" << getFitnessScore() << " with max dist  " << max_dist << "\n";


//    max_dist=0.05;


//    while( currPhotoCons > MAX_PHOTOCONS && max_dist <= 0.25) {

//        if (getFinalTransformation() != Eigen::Matrix4f::Identity() && getPhotoConsistency() < 40 )
//            align(cloud,getFinalTransformation(),max_dist);
//        else
//            align(cloud,guess,max_dist);


//        if( getFinalTransformation() != Eigen::Matrix4f::Identity() ) {
//            float pCons = getPhotoConsistency();
//            std::cout << "Obtained " << pCons /*<< "Fitness" << getFitnessScore()*/ << "with max dist: " << max_dist << "\n";
//            if( pCons < (currPhotoCons-5) || (pCons < currPhotoCons && ((getFitnessScore()*0.05) < currFitness))  || (pCons < (currPhotoCons+5) && getFitnessScore() < (currFitness*0.05) )) {
//                currPhotoCons = pCons;
//                currFitness = getFitnessScore();
//                currTransf = getFinalTransformation();
//            }
//        } else {
//            std::cout << "failed with " << max_dist << "\n";
//        }

//        max_dist = max_dist + 0.1;
//        std::cout << max_dist << "\n";

//    }
//    std::cout << "2end\n";

    finalTransf = currTransf;
}

Eigen::Matrix4f CustomICP::getFinalTransformation() {

    return finalTransf;


}

pcl::Correspondences CustomICP::getCorrespondences()
{
    return correspondences;
}

pcl::PointCloud<pcl::PointXYZRGB> CustomICP::getSourceFiltered()
{
    return srcNonDense;
}

pcl::PointCloud<pcl::PointXYZRGB> CustomICP::getTargetFiltered()
{
    return tgtNonDense;
}

double CustomICP::getFitnessScore()
{

    return fitness;

}

void CustomICP::setPrevTransf(Eigen::Matrix4f prevT)
{
    prevTransf = prevT;
}

void CustomICP::setOflowStop(bool val)
{
    stopIfOflowFails = val;
}

bool CustomICP::foundOflowTransf()
{
    return oflowFound;
}


float CustomICP::getPhotoConsistency()
{
    return photoConsistency(*src,*tgt,finalTransf);
}

float CustomICP::getPhotoConsistency(Eigen::Matrix4f ctransf)
{
    return photoConsistency(*src,*tgt,ctransf);
}

float CustomICP::getPhotoConsistency(pcl::PointCloud<pcl::PointXYZRGB> &cloudA, pcl::PointCloud<pcl::PointXYZRGB> &cloudB, Eigen::Matrix4f ctransf)
{
    return photoConsistency(cloudA,cloudB,ctransf);
}


