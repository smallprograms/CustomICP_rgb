#include <iostream>
#include <sstream>
#include <cmath>
#include "CustomICP.h"
#include "GraphOptimizer_G2O.h"
#include "Surf.h"
#include "BashUtils.h"
#include "opencv2/core/core.hpp"
#include "utils.h"

//flag used to press a key to process next capture
bool doNext = false;
bool saveAndQuit = false;
bool loadedGlobal = false;
std::map<int,Eigen::Matrix4f> poseMap;

std::map<int,pcl::PointCloud<pcl::PointXYZRGB> >  cloudMap;


/** capture keyboard keyDown event **/
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void *not_used)
{
    
    if (event.getKeySym () == "n" && event.keyDown ())
    {
        std::cout << "Processing next\n";
        doNext = true;
    } else if (event.getKeySym () == "s" && event.keyDown ())
    {
        std::cout << "Save and quit\n";
        saveAndQuit = true;
    }

}


void readPoses(std::string fileNamePrefix, std::map<int,Eigen::Matrix4f>& poseMap ) {
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


/** Saves the current global cloud and the transformation **/
void saveState(const pcl::PointCloud<pcl::PointXYZRGB>& globalCloud, const Eigen::Matrix4f& transf, char* outFile) {

    pcl::io::savePCDFileBinary (outFile, globalCloud);
    std::cerr << "Saved " << globalCloud.points.size () << " data points to " << outFile << "\n";
    std::string fileTransf(outFile);
    fileTransf += ".transf";
    writeTransformationMatrix(transf,fileTransf);
}



void groundTruthAlign(char* path,int min,int max,char* outFile,char* ground, int increment ) {

    pcl::VoxelGrid<pcl::PointXYZRGB> voxelFilter;
    //voxelFilter.setLeafSize(0.005,0.005,0.005);
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
        if( i > max || readCloud(i,path,currCloud) == false ) {

            std::cout << "Reading end at " << i << "\n";
            break;

        }
        std::cout << "READED:" << i << "\n";
        //apply transformation to cloud and add cloud to global cloud
        pcl::transformPointCloud(currCloud,finalCloud,transf);
        std::cout << "GLOBAL TRANSFORM:\n" << transf << "\n";
        //mergeClouds(globalCloud,finalCloud);
        globalCloud = globalCloud + finalCloud;
        //        std::cout << "global cloud size before voxel filter: " << globalCloud.size() << "\n";
        voxelFilter.setInputCloud(globalCloud.makeShared());
        voxelFilter.filter(globalCloud);
        //        std::cout << "global cloud size after voxel filter: " << globalCloud.size() << "\n";
        finalCloud.clear();

        /** calculate relative transform and photo cons **/
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

    pcl::io::savePCDFileBinary (outFile, globalCloud);
    std::cerr << "Saved " << globalCloud.points.size () << " data points to " << outFile << "\n";

}
/** returns true if cloud was saved in cloud variable **/
bool readCloud(int index, char* path, pcl::PointCloud<pcl::PointXYZRGB>& cloud) {

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
/** saves the image to Hard disk **/
bool saveCloudImage(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, std::string imageName ) {

    cv::Mat imgAcolor(480,640,CV_8UC3);
    cloudToMat(cloud.makeShared(),imgAcolor);

    if ( imwrite(imageName.c_str(),imgAcolor) ) return true;

    return false;

}
/** just check correspondences from A to B */
Eigen::Matrix4f getBestGuessUnidirectional(CustomICP& icp, Surf& surf, const int indexA, const int indexB,pcl::PointCloud<pcl::PointXYZRGB> &cloudA,pcl::PointCloud<pcl::PointXYZRGB> &cloudB, float& photoCons) {


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
/** If photoconsistency is bad, calculate transformation from B to A too */
Eigen::Matrix4f getBestGuess(CustomICP& icp, Surf& surf, const int indexA, const int indexB,pcl::PointCloud<pcl::PointXYZRGB> &cloudA,pcl::PointCloud<pcl::PointXYZRGB> &cloudB, float& photoCons) {

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


/** loads different captures (.pcd files), align them with customICP and write them aligned in a single file (outFile) **/
void  alignAndView( pcl::visualization::PCLVisualizer* viewer, char* path, int min, int max, char* outFile, char* global ) {

    CustomICP icp;
    GraphOptimizer_G2O optimizer;
    Surf surf;
    pcl::FastBilateralFilter<pcl::PointXYZRGB> fastBilFilter;
    pcl::VoxelGrid<pcl::PointXYZRGB> voxelFilter;
    voxelFilter.setLeafSize(0.01,0.01,0.01);

    //pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
    //to accumulate ICP transformations
    static Eigen::Matrix4d transf = Eigen::Matrix4d::Identity ();

    //if loading a previous reconstruction, transf is not identity!
    if( loadedGlobal ) {

        std::string transfName(global);
        std::string prevTransfName(global);
        transfName += ".transf";
        transf = loadTransformationMatrix(transfName).cast<double>();
        prevTransfName += ".prevtransf";
        Eigen::Matrix4f prevTransf = Eigen::Matrix4f::Identity();
        prevTransf = loadTransformationMatrix(prevTransfName);
        icp.setPrevTransf(prevTransf);
        std::cout << "readed : " << transf << "\n";
        std::cout << "and readed : " << prevTransf << "\n";
        std::string cmd= std::string("./copyFiles.sh ") + std::string(global) + " " + std::string(outFile);
        bash::exec(const_cast<char*>(cmd.c_str()));
        readPoses(std::string(global),poseMap);
        optimizer.loadGraph(std::string(global) + std::string(".g2o"));

    }
    //previous cloud
    pcl::PointCloud<pcl::PointXYZRGB> prevCloud(640,480);
    //global cloud (to register aligned clouds)
    pcl::PointCloud<pcl::PointXYZRGB> globalCloud;
    //register method to capture keyboard events
    viewer->registerKeyboardCallback( keyboardEventOccurred );

    if( !readCloud(min,path,prevCloud) ) {
        std::cout << "Failed to read first cloud \n";
        return;
    }

    fastBilFilter.setInputCloud(prevCloud.makeShared());
    fastBilFilter.filter(prevCloud);

    //if loading prev reconstr, globalCloud is not empty at init
    if( loadedGlobal ) {

        if( pcl::io::loadPCDFile<pcl::PointXYZRGB>(global, globalCloud) == -1 ) {

            std::cout << "Failed to read globalCloud cloud \n";
            return;
        }

    } else {
        //initialize globalCloud with first cloud
        globalCloud = prevCloud;
    }

    bool showCorrespondences = false;
    int prevCloudIndex;
    //in order execute algorithm just betwee two arbitrary clouds (user must give max < min)
    //just do one iteration from max to max!
    if( max < min ) {
        prevCloudIndex = min;
        min = max -1;
        showCorrespondences = true;
    }
    if( max == min ) {
        prevCloudIndex = min;
        min = min -1;
        showCorrespondences = true;
    }
    bool automatic=true;


    if( !loadedGlobal ) {
        optimizer.addVertex( transf, min, true  );
        writeNumber(min,std::string(outFile)+".vertexId.txt");
        //to have one pose / vertex per cloud, the first cloud is in the origin with no rotation
        poseMap[min] = Eigen::Matrix4f::Identity();
        writeTransformationQuaternion(Eigen::Matrix4f::Identity() , std::string(outFile) + std::string(".movements"));
    }
    //save last transforms!
    std::deque< Eigen::Matrix4f > transfQueue;
    const int TQUEUE_SIZE=30;

    //read file by file, save and quit if user press s
    for(int i=min+1; i <= max && saveAndQuit == false; i++) {

        //go ahead if user press n or is in automatic mode
        if( doNext || automatic ) {

            doNext = false;
            pcl::PointCloud<pcl::PointXYZRGB> currCloud(640,480);

            //read current cloud from file
            if( !readCloud(i,path,currCloud) ) {
                std::cout << "Reading end at " << i << "\n";
                break;
            }


            std::cout << "READED consecutiv: " << i << "\n";
            /** end read next cloud **/

            /**
            //apply voxel filter to curr cloud
            voxelFilter.setInputCloud(currCloudNotDense.makeShared());
            voxelFilter.filter(currCloudNotDense);
            /**/

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


            if( transfQueue.size() >= TQUEUE_SIZE ) {
                transfQueue.pop_front();
            }


            float currentPhotoCons = icp.getPhotoConsistency(icpTransf);
            if( icp.getPhotoConsistency(guess) < (currentPhotoCons-10) ) {
                icpTransf = guess;
                currentPhotoCons = icp.getPhotoConsistency(guess);
                std::cout << "preferred visual instead icp\n";
            } else {
                std::cout << "preferred icp\n";
            }


            std::cout << "current photo cons::: " << currentPhotoCons << "\n";

            transfQueue.push_back(icpTransf);
            transf = icpTransf.cast<double>() * transf;

            /**/

            if( showCorrespondences ) {
                float visualDist;
                int pixelDist;

                surf.visualDistance(i,prevCloudIndex,currCloud,prevCloud,visualDist,pixelDist);
                std::cout << "Visual dist: " <<  visualDist << "pixel DIst:" << pixelDist << "\n";
                std::cout << "currCloud: " <<  i << "prevCloud:" << prevCloudIndex << "\n";
                std::cout << "Photo cons " << icp.getPhotoConsistency() << "\n";
            }

            if( showCorrespondences == false  ) {
                int vertexID = optimizer.addVertex(transf,i);
                writeNumber(i,std::string(outFile)+".vertexId.txt");
                poseMap[i] = transf.cast<float>();

                if( vertexID > 1 ) {
                    int toIndex=vertexID-1;
                    //add graph consecutive edges to optimizer
                    //mem leak
                    Eigen::Matrix4f* relPose = new Eigen::Matrix4f;
                    Eigen::Matrix<double,6,6>* infMatrix = new Eigen::Matrix<double,6,6>;

                    optimizer.fillInformationMatrix(*infMatrix,currentPhotoCons);
                    optimizer.addEdge(toIndex,vertexID,icpTransf,*infMatrix);
                    writeEdge(toIndex,vertexID,icpTransf,*infMatrix,outFile);


                    //add loop closure edges!
                    /**/
                    int prevIndex;
                    if( currentPhotoCons > 30 ) {
                        prevIndex = i-2; //add more contraints for current cloud because it was poorly aligned with previous
                    } else {
                        prevIndex = i-15;
                    }
                    for( int k=prevIndex; k >= poseMap.begin()->first; k=k-1 ) {
                        const float MAX_METERS = 0.75;

                        if ( matrixDistance(poseMap[i],poseMap[k]) < MAX_METERS )  {

                            pcl::PointCloud<pcl::PointXYZRGB> pastCloud(640,480);
                            if( readCloud(k,path,pastCloud)  ) {
                                float visualDist;
                                int pixelDist;
                                surf.visualDistance(i,k,currCloud,pastCloud,visualDist,pixelDist);
//                                std::cout << "Visual dist: " <<  visualDist << "pixel DIst:" << pixelDist << "\n";
//                                std::cout << "i: " <<  i << "k:" << k << "\n";
                                if( (visualDist < 0.2 && pixelDist < 75) || (visualDist < 0.35 && pixelDist < 30) ) {

                                    Eigen::Matrix4f* relPose = new Eigen::Matrix4f;
                                    Eigen::Matrix<double,6,6>* infMatrix = new Eigen::Matrix<double,6,6>;
                                    float photoCons;
                                    guess = getBestGuess(icp,surf,k,i,pastCloud,currCloud,photoCons);
                                    if( photoCons < 60 ) {

                                        optimizer.genEdgeData(guess,pastCloud.makeShared(),currCloud.makeShared(),*relPose,*infMatrix,photoCons);


                                        if( *relPose != Eigen::Matrix4f::Identity() && photoCons < 30 ) {

                                            std::cout << "ADDING LOOP CLOSURE " << k << " - " << i << "\n";
                                            std::cout << "photoCons:" << photoCons << "\n";
                                            std::cout << "Visual dist: " <<  visualDist << "pixel DIst:" << pixelDist << "\n\n";
                                            optimizer.addEdge(i,k,*relPose,*infMatrix);
                                            //save edge on .txt files
                                            writeEdge(i,k,*relPose,*infMatrix,outFile);


                                        }
                                    }
                                }
                            }
                        }
                    } /**/

                }
            }

            /**/

            writeTransformationQuaternion(transf.cast<float>(), std::string(outFile) + std::string(".movements"));
            pcl::transformPointCloud(currCloud,finalCloud,transf.cast<float>());


            prevCloud.clear();
            pcl::copyPointCloud(currCloud,prevCloud);


            if( showCorrespondences )
                globalCloud = globalCloud + finalCloud;


            finalCloud.clear();

            if( showCorrespondences ) {

                showCorrespondences = false;
                viewer->addPointCloud(globalCloud.makeShared());


            } else {

                viewer->removeAllPointClouds();


            }


            if(saveAndQuit) {
                std::cout << "breaking\n";
                break;
            }

        } else {

            while( !viewer->wasStopped() ) {

                viewer->spinOnce (100);
                boost::this_thread::sleep (boost::posix_time::microseconds (100000));
                if( doNext || saveAndQuit ) break; //quit from visualizer loop, go to for loop

            }
            //dont skip the current capture
            --i;
        }


    } //end for

    while( !viewer->wasStopped() && showCorrespondences ) {

        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));

    }

    std::string graphFileName = std::string(outFile) + ".g2o";
    optimizer.saveGraph(graphFileName);

    if( showCorrespondences == false ) {
        /**/
        optimizer.optimizeGraph();
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >  poses;
        optimizer.getPoses(poses);


        for( std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >::iterator iter = poses.begin(); iter != poses.end(); iter++ ) {
            writeTransformationQuaternion(*iter,std::string(outFile) + std::string(".optimized"));
        }
    }
    /**/
    std::string prevTransfFile(outFile);
    prevTransfFile += ".prevtransf";
    writeTransformationMatrix(icp.getFinalTransformation(),prevTransfFile);
    saveState(globalCloud,transf.cast<float>(),outFile);
    return;


}



int main (int argc, char** argv)
{
    //path of files to load, min and max index of files
    int min;
    int max;
    char* path;
    char* outFile;
    char* global; //previously constructed cloud path
    char* groundTruth; //ground truth cam movement path

    if( argc < 5) {

        std::cout << "Usage:\n " << argv[0] << " [global_cloud] path min_index max_index out_file\n";
        std::cout << "Example:\n " << argv[0] << " car 2 10 car.pcd\n";
        std::cout << "Will use car/cap2 until car/cap10 and save the aligned clouds in car.pcd\n";
        std::cout <<  argv[0] << " global_car.pcd car 2 10 car.pcd\n";
        std::cout << "Will load and apply global_car.pcd.transf and then add  car/cap2 until car/cap10 to global_car.pcd and save the aligned clouds in car.pcd\n";
        std::cout <<  argv[0] << " movements.txt car 2 10 car.pcd\n";
        std::cout << "Will apply movements to each cloud and will generate a global cloud car.pcd with them\n";
        std::cout <<  argv[0] << " car.pcd car.pcd.optimized_poses\n";
        std::cout << "Will run graph optimizer (it will read car.pcd.movements,car.pcd..from_to.txt,car.pcd.fitness.txt,car.pcd.relativeTransf.txt to read graph) \n";
        return 0;


    } else {

        if(argc >= 6) {

            min = atoi(argv[3]);
            max = atoi(argv[4]);
            path = argv[2];
            outFile = argv[5];

            int increment = 1;
            if( argc == 7 ) {
                increment = atoi(argv[6]);
            }

            if( endswith(argv[1], ".pcd") == false ) {

                groundTruth = argv[1];
                groundTruthAlign( path, min, max, outFile, groundTruth, increment );
                return 0;

            } else {
                global = argv[1];
                loadedGlobal = true;
            }



        } else {

            min = atoi(argv[2]);
            max = atoi(argv[3]);
            path = argv[1];
            outFile = argv[4];

        }

    }

    pcl::visualization::PCLVisualizer viewer("Dots");
    viewer.setBackgroundColor (0, 0, 0);
    viewer.addCoordinateSystem (1.0);
    alignAndView(&viewer, path, min, max, outFile, global);


}
