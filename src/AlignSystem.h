#ifndef ALIGNSYSTEM_H
#define ALIGNSYSTEM_H
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <boost/thread/thread.hpp>
#include "CustomCorrespondenceEstimation.h"
#include "oflow_pcl.h"
#include "BilateralFilter.h"
#include <pcl/filters/fast_bilateral.h>
#include "SobelFilter.h"
#include "EdgeFilter.h"
#include <unsupported/Eigen/SparseExtra>

class CustomICP;
class GraphOptimizer_G2O;
class Surf;

class AlignSystem
{
public:

    AlignSystem(char* cloudsPath, char* outFile, char* global, int min, int max);

    void  align(bool loadedGlobal );

    void groundTruthAlign(char* ground, int increment );


private:
    Eigen::Matrix4f getBestGuess(CustomICP& icp, Surf& surf, const int indexA, const int indexB,pcl::PointCloud<pcl::PointXYZRGB> &cloudA,
                                    pcl::PointCloud<pcl::PointXYZRGB> &cloudB, float& photoCons);

    Eigen::Matrix4f getBestGuessUnidirectional(CustomICP& icp, Surf& surf, const int indexA, const int indexB,
                                    pcl::PointCloud<pcl::PointXYZRGB> &cloudA,pcl::PointCloud<pcl::PointXYZRGB> &cloudB, float& photoCons);

    bool readCloud(int index, char* path, pcl::PointCloud<pcl::PointXYZRGB>& cloud);
    void readPoses(std::string fileNamePrefix, std::map<int,Eigen::Matrix4f>& poseMap );
    void loadState(char* global,char* path, char* outFile);
    void saveState(const Eigen::Matrix4f& transf, char* outFile);



    void detectLoops(int i,float& currentPhotoCons, Eigen::Matrix4f& guess,pcl::PointCloud<pcl::PointXYZRGB>& currCloud);
    bool saveCloudImage(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, std::string imageName );

    CustomICP icp;
    GraphOptimizer_G2O optimizer;
    Surf surf;
    std::map<int,Eigen::Matrix4f> poseMap;
    std::map<int,pcl::PointCloud<pcl::PointXYZRGB> >  cloudMap;
    Eigen::Matrix4d transf;
    bool loadedGlobal;
    char *cloudsPath;
    char *outFile;
    char *global;
    int min;
    int max;

};

#endif // ALIGNSYSTEM_H
