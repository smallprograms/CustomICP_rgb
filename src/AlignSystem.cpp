#include <iostream>
#include <sstream>
#include <cmath>
#include "CustomICP.h"
#include "GraphOptimizer_G2O.h"
#include "Surf.h"
#include "BashUtils.h"
#include "opencv2/core/core.hpp"
#include "utils.h"
#include "AlignSystem.h"
#include "CustomICP.h"


/** loads different captures (.pcd files), align them with customICP and write them aligned in a single file (outFile) **/
AlignSystem::AlignSystem(char *cloudsPath_, char *outFile_, char *global_, int min_, int max_) :
    loadedGlobal(false), transf(Eigen::Matrix4d::Identity()),
    cloudsPath(cloudsPath_),outFile(outFile_),global(global_),min(min_),max(max_)
{
}

/** Read point clouds, save trajectory estimated with CustomICP (.movements) and after applying graph optimization (.optimized) */
void  AlignSystem::align( bool loadedGlobal ) {

    pcl::FastBilateralFilter<pcl::PointXYZRGB> fastBilFilter;
    pcl::VoxelGrid<pcl::PointXYZRGB> voxelFilter;
    const float METERS_SIDE=0.01;
    //set voxel size in meters
    voxelFilter.setLeafSize(METERS_SIDE,METERS_SIDE,METERS_SIDE);
    //uncomment to see icp iterations stuff
    //pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);

    if( loadedGlobal ) {
        //load global transf and load g2o graph
        loadState(global,cloudsPath,outFile);
    }

    //previous cloud
    pcl::PointCloud<pcl::PointXYZRGB> prevCloud(640,480);

    //read first cloud
    if( !readCloud(min,cloudsPath,prevCloud) ) {
        std::cout << "Failed to read first cloud \n";
        return;
    }
    //apply bilateral filter
    fastBilFilter.setInputCloud(prevCloud.makeShared());
    fastBilFilter.filter(prevCloud);

    int prevCloudIndex;

    //in order execute algorithm just between two arbitrary clouds (user must give max < min)
    //just do one iteration from max to max!
    if( max < min ) {
        prevCloudIndex = min;
        min = max -1;
    }
    if( max == min ) {
        prevCloudIndex = min;
        min = min -1;
    }

    //add first pose as identity if not previous trajectory was loaded
    if( !loadedGlobal ) {
        optimizer.addVertex( transf, min, true  );
        writeNumber(min,std::string(outFile)+".vertexId.txt");
        //to have one pose / vertex per cloud, the first cloud is in the origin with no rotation
        poseMap[min] = Eigen::Matrix4f::Identity();
        writeTransformationQuaternion(Eigen::Matrix4f::Identity() , std::string(outFile) + std::string(".movements"));
    }

    //read file by file
    for(int i=min+1; i <= max ; i++) {

        pcl::PointCloud<pcl::PointXYZRGB> currCloud(640,480);

        //read current cloud from file
        if( !readCloud(i,cloudsPath,currCloud) ) {
            std::cout << "Reading end at " << i << "\n";
            break;
        }

        std::cout << "Readed cloud: " << i << "\n";

        //Apply bilateral filter
        fastBilFilter.setInputCloud(currCloud.makeShared());
        fastBilFilter.filter(currCloud);

        //start ICP
        icp.setInputSource(currCloud.makeShared());
        icp.setInputTarget(prevCloud.makeShared());

        pcl::PointCloud<pcl::PointXYZRGB> finalCloud(640,480);
        float photoCons;
        Eigen::Matrix4f guess = getBestGuess(icp,surf,i,i-1,currCloud,prevCloud,photoCons);

        //update global transformation
        Eigen::Matrix4f icpTransf;
        const bool LOOP=false; //try to improve align if poor photocons
        icp.align (finalCloud, guess, LOOP );
        icpTransf = icp.getFinalTransformation();

        float currentPhotoCons = icp.getPhotoConsistency(icpTransf);
        //check if ICP gives better photoconsistency than guess obtained with visual clues
        if( icp.getPhotoConsistency(guess) < (currentPhotoCons-10) ) {
            icpTransf = guess;
            currentPhotoCons = icp.getPhotoConsistency(guess);
            std::cout << "preferred visual instead icp\n";
        } else {
            std::cout << "preferred icp\n";
        }

        std::cout << "current photo cons::: " << currentPhotoCons << "\n";
        transf = icpTransf.cast<double>() * transf;

        int vertexID = optimizer.addVertex(transf,i);
        writeNumber(i,std::string(outFile)+".vertexId.txt");
        poseMap[i] = transf.cast<float>();

        if( vertexID > 1 ) {
            int toIndex=vertexID-1;
            //add graph consecutive edges to optimizer
            Eigen::Matrix<double,6,6> infMatrix;
            //set diagonals of matrix as currentPhotoCons (All components with same weight)
            optimizer.fillInformationMatrix(infMatrix,currentPhotoCons);
            optimizer.addEdge(toIndex,vertexID,icpTransf,infMatrix);
            writeEdge(toIndex,vertexID,icpTransf,infMatrix,outFile);

            //add loop closure edges!
            detectLoops(i,currentPhotoCons,guess,currCloud);
        }

        //save current global transformation
        writeTransformationQuaternion(transf.cast<float>(), std::string(outFile) + std::string(".movements"));

        //set prevCloud as currCloud
        prevCloud.clear();
        pcl::copyPointCloud(currCloud,prevCloud);
        finalCloud.clear();

    }    //end for

    //save current pose graph
    std::string graphFileName = std::string(outFile) + ".g2o";
    optimizer.saveGraph(graphFileName);

    optimizer.optimizeGraph();
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >  poses;
    optimizer.getPoses(poses);

    //save optimized poses
    for( std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >::iterator iter = poses.begin(); iter != poses.end(); iter++ ) {
        writeTransformationQuaternion(*iter,std::string(outFile) + std::string(".optimized"));
    }

    //save last relative transf,
    std::string prevTransfFile(outFile);
    prevTransfFile += ".prevtransf";
    writeTransformationMatrix(icp.getFinalTransformation(),prevTransfFile);
    //save global transf
    saveState(transf.cast<float>(),outFile);

}

/** read positions and orientations from text file, apply to each point cloud and generate global voxelized point cloud **/
void AlignSystem::groundTruthAlign(char* ground, int increment ) {

    pcl::VoxelGrid<pcl::PointXYZRGB> voxelFilter;
    voxelFilter.setLeafSize(0.01,0.01,0.01);

    std::string groundT(ground);
    std::ifstream groundFile(groundT.c_str());
    float tx,ty,tz;
    float qw,qx,qy,qz;

    Eigen::Matrix4f transf;
    Eigen::Matrix4f prevTransf;
    Eigen::Matrix4f relTransf;
    //previous cloud
    pcl::PointCloud<pcl::PointXYZRGB> currCloud(640,480);
    pcl::PointCloud<pcl::PointXYZRGB> prevCloud(640,480);
    pcl::PointCloud<pcl::PointXYZRGB> finalCloud(640,480);

    //global cloud (to register aligned clouds)
    pcl::PointCloud<pcl::PointXYZRGB> globalCloud;

    int i = min;

    //read file line by line (vec3 position and rotation quaternion)
    while( groundFile >> tx >> ty >> tz >> qx >> qy >> qz >> qw ) {

        transf = quaternionToMatrix(tx,ty,tz,qx,qy,qz,qw);

        //read current cloud from file
        if( i > max || readCloud(i,cloudsPath,currCloud) == false ) {

            std::cout << "Reading end at " << i << "\n";
            break;

        }
        std::cout << "READED:" << i << "\n";
        //apply transformation to cloud and add cloud to global cloud
        pcl::transformPointCloud(currCloud,finalCloud,transf);
        globalCloud = globalCloud + finalCloud;
        voxelFilter.setInputCloud(globalCloud.makeShared());
        voxelFilter.filter(globalCloud);

        finalCloud.clear();

        /** calculate relative transform and photo cons
        if( i != min ) {

            relTransf = prevTransf.inverse()*transf;
            CustomICP icp;
            icp.setInputSource(currCloud.makeShared());
            icp.setInputTarget(prevCloud.makeShared());
            std::cout << "Photo cons:: " << icp.getPhotoConsistency(relTransf) << "\n";

        }
        /**/

        pcl::copyPointCloud(currCloud,prevCloud);
        prevTransf = transf;
        i+=increment;
    }
    //save global point cloud to HD
    pcl::io::savePCDFileBinary (outFile, globalCloud);
    std::cerr << "Saved " << globalCloud.points.size () << " data points to " << outFile << "\n";

}

/** If photoconsistency is bad, calculate transformation from B to A too. Because the order of images
 alters the result */
Eigen::Matrix4f AlignSystem::getBestGuess(CustomICP& icp, Surf& surf, const int indexA, const int indexB,pcl::PointCloud<pcl::PointXYZRGB> &cloudA,pcl::PointCloud<pcl::PointXYZRGB> &cloudB, float& photoCons) {

    Eigen::Matrix4f guess = getBestGuessUnidirectional(icp,surf,indexA,indexB,cloudA,cloudB,photoCons);

    if( photoCons > 40 ) {
        float prevPhotoCons = photoCons;
        Eigen::Matrix4f guessInverse = getBestGuessUnidirectional(icp,surf,indexB,indexA,cloudB,cloudA,photoCons).inverse();
        float guessInvPhotoCons =  icp.getPhotoConsistency(guessInverse);

        if( guessInvPhotoCons < prevPhotoCons ) {

            photoCons = guessInvPhotoCons;
            return guessInverse;
        } else {
            photoCons = prevPhotoCons;
            return guess;
        }
    }

    return guess;
}

/** just check correspondences from A to B */
Eigen::Matrix4f AlignSystem::getBestGuessUnidirectional(CustomICP& icp, Surf& surf, const int indexA, const int indexB,pcl::PointCloud<pcl::PointXYZRGB> &cloudA,pcl::PointCloud<pcl::PointXYZRGB> &cloudB, float& photoCons) {


    float maxCorrespDist = 0.05;
    Eigen::Matrix4f guess2 = getOflow3Dtransf(cloudA.makeShared(),cloudB.makeShared(),maxCorrespDist);

    while( guess2 == Eigen::Matrix4f::Identity() && maxCorrespDist < 0.2 ) {
        maxCorrespDist = maxCorrespDist + 0.05;
        guess2 = getOflow3Dtransf(cloudA.makeShared(),cloudB.makeShared(),maxCorrespDist);
    }

    float oflowP = icp.getPhotoConsistency(cloudA,cloudB,guess2);

    float GOOD_PHOTOCONS=20;
    //avoid to calculate SURF, this is good aproximation!
    if( oflowP < GOOD_PHOTOCONS  ) {
        photoCons = oflowP;
        return guess2;
    }

    Eigen::Matrix4f guess1 = surf.getSurfTransform(indexA,indexB,cloudA,cloudB);
    float surfP = icp.getPhotoConsistency(cloudA,cloudB,guess1);


    if( surfP < oflowP ) {
        photoCons = surfP;
        return guess1;

    } else {
        photoCons = oflowP;
        return guess2;
    }

}

/** returns true if cloud was saved in cloud variable **/
bool AlignSystem::readCloud(int index, char* path, pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
    //check if cloud is already loaded
    if( cloudMap.find( index ) == cloudMap.end() ) {

        //name of cloud file
        std::string command("/getPcdName.sh ");
        command = std::string(path) + command;
        command = command + boost::lexical_cast<std::string>(index);
        std::string filePath=bash::exec( const_cast<char*>(command.c_str()) );
        filePath.erase(std::remove(filePath.begin(), filePath.end(), '\n'), filePath.end());
        if( pcl::io::loadPCDFile<pcl::PointXYZRGB>(filePath, cloud) == -1 ) {

            return false;

        }

        cloudMap[index] = cloud;

        if( cloudMap.size() > 30 ) {
            cloudMap.erase(cloudMap.begin());
        }
    } else {
        cloud = cloudMap.at(index);
    }

    return true;
}

void AlignSystem::readPoses(std::string fileNamePrefix, std::map<int,Eigen::Matrix4f>& poseMap ) {
    std::string posesFileName = fileNamePrefix + ".movements";
    std::string vertexFileName = fileNamePrefix + ".vertexId.txt";
    std::ifstream posesFile(posesFileName.c_str());
    std::ifstream vertexFile(vertexFileName.c_str());

    float tx,ty,tz;
    float qw,qx,qy,qz;
    Eigen::Matrix4f transf;
    int vertexId;

    while( posesFile >> tx >> ty >> tz >> qx >> qy >> qz >> qw ) {
        vertexFile >> vertexId;
        transf = quaternionToMatrix(tx,ty,tz,qx,qy,qz,qw);
        poseMap[vertexId] = transf;
    }
}

/** Loads previous trajectory and g2o pose graph to continue from that point */
void AlignSystem::loadState(char *global, char *path, char *outFile) {

    std::string transfName(global);
    std::string prevTransfName(global);
    transfName += ".transf";
    transf = loadTransformationMatrix(transfName).cast<double>();
    prevTransfName += ".prevtransf";
    Eigen::Matrix4f prevTransf = Eigen::Matrix4f::Identity();
    prevTransf = loadTransformationMatrix(prevTransfName);
    icp.setPrevTransf(prevTransf);
    std::cout << "Readed previous global transform: " << transf << "\n";
    std::string cmd= std::string("./copyFiles.sh ") + std::string(global) + " " + std::string(outFile);
    bash::exec(const_cast<char*>(cmd.c_str()));
    readPoses(std::string(global),poseMap);
    optimizer.loadGraph(std::string(global) + std::string(".g2o"));
}

/** Saves the current global transformation **/
void AlignSystem::saveState(const Eigen::Matrix4f& transf, char* outFile) {

    std::string fileTransf(outFile);
    fileTransf += ".transf";
    writeTransformationMatrix(transf,fileTransf);
}





/** Look similar previous point clouds in order to add constraints to the graph */
void AlignSystem::detectLoops(int i, float& currentPhotoCons, Eigen::Matrix4f& guess,pcl::PointCloud<pcl::PointXYZRGB>& currCloud ) {

    int prevIndex;
    if( currentPhotoCons > 30 ) {
        prevIndex = i-2; //add more contraints for current cloud because it was poorly aligned with previous
    } else {
        prevIndex = i-15;
    }
    for( int k=prevIndex; k >= poseMap.begin()->first; k=k-1 ) {
        const float MAX_METERS = 0.75;
        //avoid to check visual similarity between too far clouds
        if ( matrixDistance(poseMap[i],poseMap[k]) < MAX_METERS )  {

            pcl::PointCloud<pcl::PointXYZRGB> pastCloud(640,480);
            if( readCloud(k,cloudsPath,pastCloud)  ) {
                float visualDist;
                int pixelDist;
                surf.visualDistance(i,k,currCloud,pastCloud,visualDist,pixelDist);
                //check if they are similar with SURF features
                if( (visualDist < 0.2 && pixelDist < 75) || (visualDist < 0.35 && pixelDist < 30) ) {

                    Eigen::Matrix4f relPose;
                    Eigen::Matrix<double,6,6> infMatrix;
                    float photoCons;
                    guess = getBestGuess(icp,surf,k,i,pastCloud,currCloud,photoCons);
                    //if ICP guess has very poor photoconsistency dont add this constraint
                    if( photoCons < 60 ) {

                        optimizer.genEdgeData(guess,pastCloud.makeShared(),currCloud.makeShared(),relPose,infMatrix,photoCons);

                        //if ICP align has good enough photoconsistency add constraint
                        if( relPose != Eigen::Matrix4f::Identity() && photoCons < 30 ) {

                            std::cout << "ADDING LOOP CLOSURE " << k << " - " << i << "\n";
                            std::cout << "photoCons:" << photoCons << "\n";
                            std::cout << "Visual dist: " <<  visualDist << "pixel DIst:" << pixelDist << "\n\n";
                            optimizer.addEdge(i,k,relPose,infMatrix);
                            //save edge on .txt files
                            writeEdge(i,k,relPose,infMatrix,outFile);


                        }
                    }
                }
            }
        }
    }
}

/** saves the image to Hard disk **/
bool AlignSystem::saveCloudImage(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, std::string imageName ) {

    cv::Mat imgAcolor(480,640,CV_8UC3);
    cloudToMat(cloud.makeShared(),imgAcolor);

    if ( imwrite(imageName.c_str(),imgAcolor) ) return true;

    return false;

}




