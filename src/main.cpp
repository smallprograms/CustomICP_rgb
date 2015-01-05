#include <iostream>
#include <sstream>
#include <cmath>
#include "CustomICP.h"
#include "GraphOptimizer_G2O.h"
#include "Surf.h"
#include "BashUtils.h"
#include "opencv2/core/core.hpp"


//flag used to press a key to process next capture
bool doNext = false;
bool saveAndQuit = false;
bool loadedGlobal = false;
std::map<int,Eigen::Matrix4f> poseMap;

std::map<int,pcl::PointCloud<pcl::PointXYZRGB> >  cloudMap;

bool readCloud(int index, char* path, pcl::PointCloud<pcl::PointXYZRGB>& cloud);

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

void writeTransformationQuaternion(Eigen::Matrix4f transf, std::string fileName) {

    //std::cout << transf.col(0)[0];
    Eigen::Matrix3f rot = transf.block(0,0,3,3);
    Eigen::Quaternionf quat(rot);
    quat.normalize();
    std::ofstream file(fileName.c_str(), ios::out | ios::app);
    file << transf.col(3)[0] << " " << transf.col(3)[1] << " " << transf.col(3)[2] << " " << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << "\n";
    file.close();
}

void writeNumber(float number, std::string fileName) {

    std::ofstream file(fileName.c_str(), ios::out | ios::app);
    file << number << "\n";
    file.close();
}

void writeNumber(int number, std::string fileName) {

    std::ofstream file(fileName.c_str(), ios::out | ios::app);
    file << number << "\n";
    file.close();
}

void writeTwoNumbers(int fromIndex, int toIndex , std::string fileName) {

    std::ofstream file(fileName.c_str(), ios::out | ios::app);
    file << fromIndex << " " << toIndex << "\n";
    file.close();
}

void writeEdge(const int fromIndex,const int toIndex,Eigen::Matrix4f& relativePose, Eigen::Matrix<double,6,6>& informationMatrix, std::string fileNamePrefix) {

    writeTwoNumbers(fromIndex, toIndex, fileNamePrefix + ".from_to.txt");
    writeTransformationQuaternion(relativePose,fileNamePrefix + ".relativeTransf.txt");
    writeNumber((float)informationMatrix.col(0)[0],fileNamePrefix + ".fitness.txt");
}


Eigen::Matrix4f quaternionToMatrix(float tx, float ty, float tz, float qx, float qy, float qz, float qw ) {

    Eigen::Matrix4f transf = Eigen::Matrix4f::Identity();

    //check if quaternion is identity
    if( (tx==0 && ty==0 && tz==0 && qx==0 && qy==0 && qz == 0) == false ) {
        //generate transformation matrix
        Eigen::Quaternion<float> quat (qw,qx,qy,qz);
        std::cout << "creating quat(w,x,y,z):: " << qw << " " << qx << " " << qy << " " << qz << "\n";
        std::cout << "readed quat:: (w,x,y,z)" << quat.w() << " " << quat.x() << " " << quat.y() << " " << quat.z() << "\n";
        Eigen::Matrix3f rot = quat.toRotationMatrix();

        //transformation matrix containing only rotation
        transf << rot(0,0),rot(0,1),rot(0,2),0,rot(1,0),rot(1,1),rot(1,2),0,rot(2,0),rot(2,1),rot(2,2),0,0,0,0,1;
        //translation vector
        Eigen::Vector3f v(tx,ty,tz);
        //update transf. matrix with translation vector
        transf = transf + Eigen::Affine3f(Eigen::Translation3f(v)).matrix();
        //substract identity because translation matrix vector has diagonal with ones
        transf = transf - Eigen::Matrix4f::Identity();
        transf(3,3) = 1;
    }

    return transf;
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

bool readGraph(std::string fileNamePrefix, GraphOptimizer_G2O& optimizer) {

    std::string posesFileName = fileNamePrefix + ".movements";
    std::cout << posesFileName << "\n";
    std::string vertexFileName = fileNamePrefix + ".vertexId.txt";
    std::ifstream groundFile(posesFileName.c_str());
    std::ifstream vertexFile(vertexFileName.c_str());

    float tx,ty,tz;
    float qw,qx,qy,qz;
    Eigen::Matrix4f transf;
    Eigen::Matrix4d transfD;
    int vertexId;
    bool isFixed=true;
    std::cout << "READ GR\n";
    //read file line by line (vec3 position and rotation quaternion)
    while( groundFile >> tx >> ty >> tz >> qx >> qy >> qz >> qw ) {
        vertexFile >> vertexId;
        transf = quaternionToMatrix(tx,ty,tz,qx,qy,qz,qw);
        transfD = transf.cast<double>();
        std::cout << "ADDING::: " << vertexId << "\n";
        optimizer.addVertex(transfD,vertexId,isFixed);
        isFixed=false;
    }
    std::cout << "READ GR\n";

    //the three files should have the same amount of lines, an edge is formed reading one line from each file
    //from index, to index
    std::string edges = fileNamePrefix + ".from_to.txt";
    std::ifstream edgesFile(edges.c_str());
    //relative transformation between from and to position
    std::string relativeTransfs = fileNamePrefix + ".relativeTransf.txt";
    std::ifstream relativeTransfsFile(relativeTransfs.c_str());
    //fitness score of the relative transformation
    std::string fitness = fileNamePrefix + ".fitness.txt";
    std::ifstream fitnessFile(fitness.c_str());

    int from,to;
    float fitnessVal;

    while( edgesFile >> from >> to ) {

        if ( relativeTransfsFile >> tx >> ty >> tz >> qx >> qy >> qz >> qw ) {
            transf = quaternionToMatrix(tx,ty,tz,qx,qy,qz,qw);
        } else {
            return false; //files entries doesnt match!
        }

        if( fitnessFile >> fitnessVal ) {
            Eigen::Matrix<double,6,6> infoMat = Eigen::Matrix<double,6,6>::Identity()*fitnessVal;
            //add readed edge to optimizer
            optimizer.addEdge(from,to,transf,infoMat);
        } else {
            return false; //files entries doesnt match!
        }
    }
}

//read edges and poses of the graph
bool readAndOptimizeGraph(std::string fileNamePrefix, std::string outFile) {

    GraphOptimizer_G2O optimizer;
    std::string posesFileName = fileNamePrefix + ".movements";
    std::string vertexFileName = fileNamePrefix + ".vertexId.txt";
    std::ifstream vertexFile(vertexFileName.c_str());

    std::ifstream groundFile(posesFileName.c_str());

    float tx,ty,tz;
    float qw,qx,qy,qz;
    Eigen::Matrix4f transf;
    Eigen::Matrix4d transfD;
    int vertexId;
    bool isFixed=true;

    //read file line by line (vec3 position and rotation quaternion)
    while( groundFile >> tx >> ty >> tz >> qx >> qy >> qz >> qw ) {

        transf = quaternionToMatrix(tx,ty,tz,qx,qy,qz,qw);
        transfD = transf.cast<double>();
        optimizer.addVertex(transfD,vertexId,isFixed);
        isFixed=false;
    }

    //the three files should have the same amount of lines, an edge is formed reading one line from each file
    //from index, to index
    std::string edges = fileNamePrefix + ".from_to.txt";
    std::ifstream edgesFile(edges.c_str());
    //relative transformation between from and to position
    std::string relativeTransfs = fileNamePrefix + ".relativeTransf.txt";
    std::ifstream relativeTransfsFile(relativeTransfs.c_str());
    //fitness score of the relative transformation
    std::string fitness = fileNamePrefix + ".fitness.txt";
    std::ifstream fitnessFile(fitness.c_str());

    int from,to;
    float fitnessVal;

    while( edgesFile >> from >> to ) {

        if ( relativeTransfsFile >> tx >> ty >> tz >> qx >> qy >> qz >> qw ) {
            transf = quaternionToMatrix(tx,ty,tz,qx,qy,qz,qw);
        } else {
            return false; //files entries doesnt match!
        }

        if( fitnessFile >> fitnessVal ) {
            Eigen::Matrix<double,6,6> infoMat = Eigen::Matrix<double,6,6>::Identity()*fitnessVal;
            //add readed edge to optimizer
            optimizer.addEdge(from,to,transf,infoMat);
        } else {
            return false; //files entries doesnt match!
        }
    }

    //run optimization on the graph
    optimizer.optimizeGraph();
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >  poses;
    optimizer.getPoses(poses);

    //write new poses (optimized)
    for( std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >::iterator iter = poses.begin(); iter != poses.end(); iter++ ) {
        writeTransformationQuaternion(*iter,std::string(outFile));
    }

    return true;
}

void reorthogonalizeMatrix(Eigen::Matrix4d& transf) {
    Eigen::Matrix3d rot = transf.block(0,0,3,3);
    Eigen::Quaterniond quat(rot);
    quat.normalize();
    transf.block(0,0,3,3) = quat.toRotationMatrix();
}

void writeTransformationMatrix(Eigen::Matrix4f transf, std::string fileName) {

    std::ofstream ofs(fileName.c_str());
    if( ofs.is_open() ) {
        std::cout << "write to " << fileName << "\n";
        ofs << transf;
    }
    ofs.close();
}



float matrixDistance(Eigen::Matrix4f transf1,Eigen::Matrix4f transf2) {
    //euclidean distance between poses
    float dist = (transf1.col(3)[0] - transf2.col(3)[0])*(transf1.col(3)[0] - transf2.col(3)[0]);
    dist += (transf1.col(3)[1] - transf2.col(3)[1])*(transf1.col(3)[1] - transf2.col(3)[1]);
    dist += (transf1.col(3)[2] - transf2.col(3)[2])*(transf1.col(3)[2] - transf2.col(3)[2]);
    dist = std::sqrt(dist);
    return dist;
}


Eigen::Matrix4f loadTransformationMatrix(std::string fileName) {

    std::ifstream transfFile(fileName.c_str());
    Eigen::Matrix4f transf;

    if( transfFile.is_open() ) {
        float num[16];
        int k=0;
        while( transfFile >> num[k]) {
            k++;

        }
        transf << num[0],num[1],num[2],num[3],num[4],num[5],num[6],num[7],num[8],num[9],num[10],num[11],num[12],num[13],num[14],num[15];
        transfFile.close();
        return transf;
    } else {
        std::cout << "Can't read transformation file\n";
        return Eigen::Matrix4f::Identity();
    }

}

int endswith(const char* haystack, const char* needle)
{
    size_t hlen;
    size_t nlen;
    /* find the length of both arguments -
    if needle is longer than haystack, haystack can't end with needle */
    hlen = strlen(haystack);
    nlen = strlen(needle);
    if(nlen > hlen) return 0;

    /* see if the end of haystack equals needle */
    return (strcmp(&haystack[hlen-nlen], needle)) == 0;
}

char rand_alnum()
{
    char c;
    while (!std::isalnum(c = static_cast<char>(std::rand())))
        ;
    return c;
}
/** random string generator **/
std::string rand_alnum_str (std::string::size_type sz)
{
    std::string s;
    s.reserve  (sz);
    generate_n (std::back_inserter(s), sz, rand_alnum);
    return s;
}
/** Saves the current global cloud and the transformation **/
void saveState(const pcl::PointCloud<pcl::PointXYZRGB>& globalCloud, const Eigen::Matrix4f& transf, char* outFile) {

    pcl::io::savePCDFileBinary (outFile, globalCloud);
    std::cerr << "Saved " << globalCloud.points.size () << " data points to " << outFile << "\n";
    std::string fileTransf(outFile);
    fileTransf += ".transf";
    writeTransformationMatrix(transf,fileTransf);
}

void mergeClouds(pcl::PointCloud<pcl::PointXYZRGB>& globalCloud, const pcl::PointCloud<pcl::PointXYZRGB>& newCloud) {

    pcl::KdTreeFLANN<pcl::PointXYZRGB> tree;
    tree.setInputCloud(globalCloud.makeShared());
    std::vector<int> index;
    std::vector<float> dist;
    for(int k=0; k < newCloud.size(); k++) {
        //avoid adding redundant points
        if( tree.radiusSearch(newCloud[k],0.005,index,dist,1) == 0 ) {
            globalCloud.push_back(newCloud[k]);
        }

    }
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

    //cloudQueue.resize(4);
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

            icp.align (finalCloud, guess );
            icpTransf = icp.getFinalTransformation();


            if( transfQueue.size() >= TQUEUE_SIZE ) {
                transfQueue.pop_front();
            }


            float currentPhotoCons = icp.getPhotoConsistency(icpTransf);

            //search previous transformation to see if we have a better photoconsitency
//            if( currentPhotoCons  > 30 ) {

//                bool improvedPhotoCons = false;
//                for(std::deque<Eigen::Matrix4f>::iterator it=transfQueue.begin();it != transfQueue.end(); it++) {
//                    float temp = icp.getPhotoConsistency(*it);
//                    temp = temp;
//                    if( (temp+15) < currentPhotoCons ) {
//                        std::cout << "improved photo cons! " << currentPhotoCons << ".." << temp << "\n";
//                        currentPhotoCons = temp;
//                        icpTransf = *it;
//                        improvedPhotoCons = true;
//                    }
//                }
//                //run ICP again, initalizing it with the previously found transf
//                if( improvedPhotoCons ) {
//                    icp.align(finalCloud, icpTransf);
//                    icpTransf = icp.getFinalTransformation();
//                    currentPhotoCons = icp.getPhotoConsistency(icpTransf);
//                }
//            }

            std::cout << "current photo cons::: " << currentPhotoCons << "\n";


//            if( currentPhotoCons > 40 ) {
//                std::cout << "looking ahead " << "\n";
//                //look ahead to avoid outlayer
//                for( int k=(i+1); k < (i+5); k=k+1 ) {

//                        pcl::PointCloud<pcl::PointXYZRGB> futureCloud(640,480);
//                        if( readCloud(k,path,futureCloud)  ) {
//                            guess = getBestGuess(icp,surf,k,i-1,futureCloud,prevCloud,photoCons);
//                            icp.setInputSource(futureCloud.makeShared());
//                            icp.align(finalCloud,guess);
//                            Eigen::Matrix4f candidateTransf = icp.getFinalTransformation();
//                            float candidatePhotoCons = icp.getPhotoConsistency();
//                            if(  candidatePhotoCons < currentPhotoCons ) {
//                                std::cout << "IMPROVED from FUTURE!! " << currentPhotoCons << " to " << candidatePhotoCons << "\n";
//                                icpTransf = candidateTransf;
//                                currentPhotoCons = candidatePhotoCons;
//                                i=k; //go to future index!
//                                pcl::copyPointCloud(futureCloud,currCloud);
//                            }

//                        }
//                }

//                }
            transfQueue.push_back(icpTransf);
            //round to four decimals
//            icpTransf.col(3)[0] = roundf(icpTransf.col(3)[0]*10000.0f)/10000.0f;
//            icpTransf.col(3)[1] = roundf(icpTransf.col(3)[1]*10000.0f)/10000.0f;
//            icpTransf.col(3)[2] = roundf(icpTransf.col(3)[2]*10000.0f)/10000.0f;

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

//                    guess = getBestGuess(icp,surf,i-1,i,prevCloud,currCloud,photoCons);
                    //optimizer.genEdgeData(guess,prevCloud.makeShared(),currCloud.makeShared(),*relPose,*infMatrix,photoCons);
                    //if( *relPose != Eigen::Matrix4f::Identity() ) {

                        optimizer.addEdge(toIndex,vertexID,icpTransf,*infMatrix);
                        writeEdge(toIndex,vertexID,icpTransf,*infMatrix,outFile);

//                        optimizer.addEdge(vertexID,toIndex,*relPose,*infMatrix);
//                        //save edge on .txt files
//                        writeEdge(vertexID,toIndex,*relPose,*infMatrix,outFile);

                    //}

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
    std::cout << "out for\n\n";

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

        if( argc == 3 ) {
            if( readAndOptimizeGraph(argv[1],argv[2]) ) {
                std::cout << "Optimized positions saved in " << argv[2] << "\n";

            } else {
                std::cout << "Error reading files, check that .movements, .from_to.txt, .fitness.txt, .relativeTransf.txt are ok\n";

            }
            return 0;
        } else {

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
        }

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
