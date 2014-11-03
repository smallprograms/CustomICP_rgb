#include "CustomICP.h"
#include "time.h"
#include <stdlib.h>

CustomICP::CustomICP()
{
    //customCorresp = new CustomCorrespondenceEstimation<pcl::PointXYZRGB,pcl::PointXYZRGB,float>;
    //icp.setCorrespondenceEstimation(
    //boost::shared_ptr<pcl::registration::CorrespondenceEstimation<pcl::PointXYZRGB,pcl::PointXYZRGB,float> > (customCorresp));

    //icp.setMaxCorrespondenceDistance(0.003); //22 sept
    std::cout << "\tsetMaxCorrespondenceDistance(0.5) \n NO CUSTOM sobel01\n";
    icp.setMaxCorrespondenceDistance(0.1);
    icp.setMaximumIterations (25);
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

void CustomICP::align( pcl::PointCloud<pcl::PointXYZRGB> &cloud )
{
    /**/
    //for max 3D distance of projected optical flow correspondences
    float maxCorrespDist = 0.05;
    //optical flow to calculate initial transformation
    oflowTransf = getOflow3Dtransf(src,tgt,maxCorrespDist);
    /*
    //try to obtain optical flow relaxing parameters
    while( oflowTransf == Eigen::Matrix4f::Identity() && maxCorrespDist < 0.2 ) {
        maxCorrespDist = maxCorrespDist + 0.05;
        oflowTransf = getOflow3Dtransf(src,tgt,maxCorrespDist);
    }

    //set the initial transf it as the previous transf
    if( oflowTransf == Eigen::Matrix4f::Identity() ) {

        std::cout << "Optical Flow Failed\n";

        oflowFound = false;
        //increment max correspondences distances
        icp.setMaxCorrespondenceDistance(0.15);

        if( stopIfOflowFails ) return;

        oflowTransf = prevTransf;

    } else {

        std::cout << "Optical Flow Transformation: \n";
        std::cout << oflowTransf << "\n";
        oflowFound = true;
    }
    /**/
    //oflowTransf = Eigen::Matrix4f::Identity();
    pcl::PointCloud<pcl::PointXYZRGB> sobTgt(640,480);
    pcl::PointCloud<pcl::PointXYZRGB> sobSrc(640,480);


    sobFilter.setInputCloud(tgt);
    sobFilter.applyFilter(sobTgt);
    sobFilter.setInputCloud(src);
    sobFilter.applyFilter(sobSrc);

    static bool saveSobel=true;
    if( saveSobel ) {
        pcl::io::savePCDFileASCII("sobTgtBef.pcd",sobTgt);
        pcl::io::savePCDFileASCII("sobSrcBef.pcd",sobSrc);
        saveSobel=false;
    }

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
    std::cout << "NON DENSE SIZE " << srcNonDense.size() << "\n";

    icp.setInputTarget(tgtNonDense.makeShared());
    icp.setInputSource(srcNonDense.makeShared());
    icp.align(cloud,oflowTransf);
    finalTransf = icp.getFinalTransformation();

    //use just optical flow transformation, without using ICP
    /**
    std::cout << "USING ONLY OFL00W \n";
    finalTransf = oflowTransf;
    /**/
    //correspondences = customCorresp->getCorrespondences();
    fitness = icp.getFitnessScore(); //segfault if you call this methoud without called icp.align first

    /** SECOND ICP WITH DIFFERENT MAX DISTANCE*/
//    icp.setMaxCorrespondenceDistance(0.015);
//    icp.align(cloud, finalTransf);
//    finalTransf = icp.getFinalTransformation();
//    correspondences = customCorresp->getCorrespondences();

//    icp.setMaxCorrespondenceDistance(0.0075);
//    icp.align(cloud, finalTransf);
//    finalTransf = icp.getFinalTransformation();
//    correspondences = customCorresp->getCorrespondences();


    /** SECOND ICP WITH POINTS DOWNSAMPLED
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icpFull;
    //icpFull.setMaxCorrespondenceDistance(0.025); //21 oct
    icpFull.setMaxCorrespondenceDistance(0.1);
    icpFull.setEuclideanFitnessEpsilon(1e-4);
    icpFull.setMaximumIterations(10);
    pcl::PointCloud<pcl::PointXYZRGB>  srcFull;
    pcl::PointCloud<pcl::PointXYZRGB>  tgtFull;
    pcl::PointCloud<pcl::PointXYZRGB>  srcDS;
    pcl::PointCloud<pcl::PointXYZRGB>  tgtDS;

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*tgt,tgtFull, indices);
    pcl::removeNaNFromPointCloud(*src,srcFull, indices);
    //Downsample
    voxelFilter.setLeafSize(0.005,0.005,0.005);
    voxelFilter.setInputCloud(tgtFull.makeShared());
    voxelFilter.filter(tgtDS);
    voxelFilter.setInputCloud(srcFull.makeShared());
    voxelFilter.filter(srcDS);
    std::cout << "ICP0.1 DOWNSAMPLED SIZE: " << tgtDS.size() << "\n";
    //apply ICP to downsampled clouds
    icpFull.setInputSource(srcDS.makeShared());
    icpFull.setInputTarget(tgtDS.makeShared());
    icpFull.align(cloud,finalTransf);
    finalTransf = icpFull.getFinalTransformation();

    //apply ICP to full clouds
//    icpFull.setMaxCorrespondenceDistance(0.01);
//    icpFull.setTransformationEpsilon(1e-4);
//    icpFull.setInputSource(srcFull.makeShared());
//    icpFull.setInputTarget(tgtFull.makeShared());
//    icpFull.align(cloud,finalTransf);
//    finalTransf = icpFull.getFinalTransformation();




    /**/

    /** // ICP with randomness

    float maxCorDist = 0.03;
    float maxFit = 0.01;
    int maxIter = 20;
    float bf = icp.getFitnessScore();
    int numCorresp = getCorrespondences().size();
    correspondences = customCorresp->getCorrespondences();
    Eigen::Vector3f yawPitchRoll = Eigen::Vector3f(0.015,0.01,0.01);
    Eigen::Vector3f xyzMaxDist = Eigen::Vector3f(0.08,0.04,0.04);
    std::cout << "CORRESPONDENCES BEFORE RANDOM:" << numCorresp << "\n";
    //bf and numCorresp are passed by reference, in order to save its values to the next call of randomICP!
    randomICP(yawPitchRoll,xyzMaxDist,maxCorDist,maxFit,maxIter,bf,numCorresp);
    prevTransf = finalTransf;

    /**/
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

void CustomICP::randomICP(Eigen::Vector3f maxYawPitchRoll, Eigen::Vector3f maxDist, float maxCorDist, float maxFit, int maxIter, float &bestFit, int &numCorresp)
{

    icp.setMaxCorrespondenceDistance(maxCorDist);
    Eigen::Matrix4f bestTransf = icp.getFinalTransformation();

    int iter=0;
    pcl::PointCloud<pcl::PointXYZRGB> notUsed(640,480);

    Eigen::Vector3f step = maxDist/12.0f;
    Eigen::Vector3f offset = -maxDist;

    float shift = -maxCorDist;
    //0=x,1=y,2=z
    int dim=0;
    bool loop=true;

    do {

        srand(time(NULL));

        //random yaw, pitch, roll angles between -maxXX,maxXX radians
        float yaw = 2*(0.5- ((float)rand())/RAND_MAX)*maxYawPitchRoll(0);
        float pitch = 2*(0.5- ((float)rand())/RAND_MAX)*maxYawPitchRoll(1);
        float roll = 2*(0.5- ((float)rand())/RAND_MAX)*maxYawPitchRoll(2);
        std::cout << "yaw,pit,r" << yaw << "," << pitch << "," << roll << "\n";
        //generate rotation matrix from random yaw,pitch,roll
        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());

        Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;

        Eigen::Matrix4f rotationMatrix = Eigen::Matrix4f::Identity(); //convert from 3D mat to 4f Mat
//        rotationMatrix(0,0) = q.matrix()(0,0);
//        rotationMatrix(0,1) = q.matrix()(0,1);
//        rotationMatrix(0,2) = q.matrix()(0,2);
//        rotationMatrix(1,0) = q.matrix()(1,0);
//        rotationMatrix(1,1) = q.matrix()(1,1);
//        rotationMatrix(1,2) = q.matrix()(1,2);
//        rotationMatrix(2,0) = q.matrix()(2,0);
//        rotationMatrix(2,1) = q.matrix()(2,1);
//        rotationMatrix(2,2) = q.matrix()(2,2);


        finalTransf = bestTransf;
        finalTransf = rotationMatrix*finalTransf;

        //random number between -maxDist(0),maxDist(0)
//        float randNum = 2*(0.5- ((float)rand())/RAND_MAX)*maxDist(0);
//        //x axis random move
//        finalTransf(0,3) = finalTransf(0,3) + randNum;
//        std::cout << "x move: " << finalTransf(0,3) << "\n";

//        //y axis random move
//        randNum = 2*(0.5- ((float)rand())/RAND_MAX)*maxDist(1);
//        finalTransf(1,3) = finalTransf(1,3) + randNum;
//        std::cout << "y move: " << finalTransf(1,3) << "\n";

//        //z axis random move
//        randNum = 2*(0.5- ((float)rand())/RAND_MAX)*maxDist(2);
//        finalTransf(2,3) = finalTransf(2,3) + randNum;
//        std::cout << "z move: " << finalTransf(2,3) << "\n";



        if( dim == 0 ) {

            finalTransf(dim,3) = finalTransf(dim,3) + offset(0);
            offset(0) = offset(0) + step(0);

            if( offset(0) > maxDist(0) ) dim = 1;

        } else if( dim == 1 ) {

            finalTransf(dim,3) = finalTransf(dim,3) + offset(1);
            offset(1) = offset(1) + step(1);

            if( offset(1) > maxDist(1) ) dim = 2;
        } else {

            finalTransf(dim,3) = finalTransf(dim,3) + offset(2);
            offset(2) = offset(2) + step(2);

            if( offset(2) > maxDist(2) ) loop = false;

        }


        icp.align(notUsed,finalTransf);
        std::cout << "nr iiter: " << iter << "\n";
        //save transformation if it has has a better fit (less distance between corresp) and more correspondences!
//        float corFact = (float)getCorrespondences().size()/(float)numCorresp;
//        if( corFact > 2 || corFact < 0.5 ) corFact = 1;
//        std::cout << "cf: " << corFact << "\n";
        if( icp.getFitnessScore() < bestFit
                &&  getCorrespondences().size() > numCorresp ) {
            bestFit = icp.getFitnessScore();
            numCorresp = getCorrespondences().size();
            bestTransf = icp.getFinalTransformation();
            //correspondences = customCorresp->getCorrespondences();
            std::cout << "\n\nsaving best FIT\n";
            std::cout << "Faf \n" << finalTransf << "\n";
            std::cout << "FITttt::: " << bestFit << "\n";
            std::cout << "numCorresp::: " << numCorresp << "\n";
        }

        iter++;

    } while( loop );

    finalTransf = bestTransf;

}


