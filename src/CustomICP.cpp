#include "CustomICP.h"

CustomICP::CustomICP()
{
    customCorresp = new CustomCorrespondenceEstimation<pcl::PointXYZRGB,pcl::PointXYZRGB,float>;
    icp.setCorrespondenceEstimation(
    boost::shared_ptr<pcl::registration::CorrespondenceEstimation<pcl::PointXYZRGB,pcl::PointXYZRGB,float> > (customCorresp));
    icp.setMaxCorrespondenceDistance(0.003);
    icp.setMaximumIterations (10);
    prevTransf = Eigen::Matrix4f::Identity();

}


void CustomICP::setInputSource( pcl::PointCloud<pcl::PointXYZRGB>::Ptr src ) {
    this->src = src;

}

void CustomICP::setInputTarget( pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt ) {
    this->tgt = tgt;

}

void CustomICP::align( pcl::PointCloud<pcl::PointXYZRGB> &cloud )
{
    //optical flow to calculate initial transformation
    oflowTransf = getOflow3Dtransf(src,tgt);
    //if we dont have a clue about an initial transf
    //set the initial transf it as the previous transf
    if( oflowTransf == Eigen::Matrix4f::Identity() ) {
        oflowTransf = prevTransf;
    }

    //oflowTransf = Eigen::Matrix4f::Identity();
    pcl::PointCloud<pcl::PointXYZRGB> sobTgt(640,480);
    pcl::PointCloud<pcl::PointXYZRGB> sobSrc(640,480);
    //source/target sobel cloud with globalTransform
    pcl::PointCloud<pcl::PointXYZRGB> sobSrcMoved(640,480);
    pcl::PointCloud<pcl::PointXYZRGB> sobTgtMoved(640,480);

    sobFilter.setInputCloud(tgt);
    sobFilter.applyFilter(sobTgt);
    sobFilter.setInputCloud(src);
    sobFilter.applyFilter(sobSrc);

    static bool saveSobel=true;
    if( saveSobel ) {
//        pcl::io::savePCDFileASCII("sobTgtBef.pcd",sobTgt);
//        pcl::io::savePCDFileASCII("sobSrcBef.pcd",sobSrc);

    }

    /** Generate clouds without NaN points to work with ICP **/
    tgtNonDense.clear();
    srcNonDense.clear();

    for(int k=0; k < sobTgt.size(); k++) {
        if( std::isnan(sobTgt.points[k].x) == false ) {
            tgtNonDense.push_back(sobTgt.at(k));
        }
    }

    for(int k=0; k < sobSrc.size(); k++) {
        if( std::isnan(sobSrc.points[k].x) == false ) {
            srcNonDense.push_back(sobSrc.at(k));
        }
    }

    if( saveSobel ) {
//        pcl::io::savePCDFileASCII("sobTgt.pcd",tgtNonDense);
//        pcl::io::savePCDFileASCII("sobSrc.pcd",srcNonDense);
        saveSobel = false;
    }

    std::cout << "sizeT...:::: " << tgtNonDense.size() << "\n";
    std::cout << "sizeS...:::: " << srcNonDense.size() << "\n";
    icp.setInputTarget(tgtNonDense.makeShared());
    icp.setInputSource(srcNonDense.makeShared());
    icp.align(cloud,oflowTransf);
    prevTransf = icp.getFinalTransformation();

    //apply ICP over original point clouds
//    std::vector<int> indices;
//    pcl::removeNaNFromPointCloud(*tgt,*tgt, indices);
//    pcl::removeNaNFromPointCloud(*src,*src, indices);

//    std::cout << "non sobel sizeT...:::: " << tgt->size() << "\n";
//    std::cout << "non sobel sizeS...:::: " << src->size() << "\n";
    //apply icp over entire cloud (no filter)
//    icp.setMaximumIterations(10);
//    icp.setInputTarget(tgt);
//    icp.setInputSource(src);
//    icp.align(cloud,sobelTransf);
    //icp.getFinalTransformation();
}

Eigen::Matrix4f CustomICP::getFinalTransformation() {

    return icp.getFinalTransformation();


}

pcl::Correspondences CustomICP::getCorrespondences()
{
    return customCorresp->getCorrespondences();
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

    return icp.getFitnessScore();

}


