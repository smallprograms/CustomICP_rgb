#include <iostream>
#include <sstream>
#include <cmath>
#include "CustomICP.h"
#include "GraphOptimizer_G2O.h"
#include "BashUtils.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

//flag used to press a key to process next capture
bool doNext = false;
bool saveAndQuit = false;
bool loadedGlobal = false;
std::vector<Eigen::Matrix4f> poses;

std::map<int,std::vector<cv::KeyPoint> > cloudKeyPoints;
std::map<int,cv::Mat>  cloudDescriptors;

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

void writeTwoNumbers(int fromIndex, int toIndex , std::string fileName) {

    std::ofstream file(fileName.c_str(), ios::out | ios::app);
    file << fromIndex << " " << toIndex << "\n";
    file.close();
}

void writeEdge(const int fromIndex,const int toIndex,Eigen::Matrix4f& relativePose, Eigen::Matrix<double,6,6>& informationMatrix, std::string fileNamePrefix) {

    writeTwoNumbers(fromIndex, toIndex, fileNamePrefix + ".from_to.txt");
    writeTransformationQuaternion(relativePose,fileNamePrefix + ".relativeTransf.txt");
    writeNumber(informationMatrix.col(0)[0],fileNamePrefix + ".fitness.txt");

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

//read edges and poses of the graph
bool readAndOptimizeGraph(std::string fileNamePrefix, std::string outFile) {

    GraphOptimizer_G2O optimizer;
    std::string posesFileName = fileNamePrefix + ".movements";

    std::ifstream groundFile(posesFileName.c_str());

    float tx,ty,tz;
    float qw,qx,qy,qz;
    Eigen::Matrix4f transf;
    Eigen::Matrix4d transfD;

    //read file line by line (vec3 position and rotation quaternion)
    while( groundFile >> tx >> ty >> tz >> qx >> qy >> qz >> qw ) {

        transf = quaternionToMatrix(tx,ty,tz,qx,qy,qz,qw);
        transfD = transf.cast<double>();
        optimizer.addVertex(transfD);
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
    //previous cloud
    pcl::PointCloud<pcl::PointXYZRGB> currCloud(640,480);
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

        i+=increment;
    }

    pcl::io::savePCDFileBinary (outFile, globalCloud);
    std::cerr << "Saved " << globalCloud.points.size () << " data points to " << outFile << "\n";

}
/** returns true if cloud was saved in cloud variable **/
bool readCloud(int index, char* path, pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
    //name of cloud file
    std::string command("/home/not/code/computer_vision/PCL/customICP_rgb-build/freiburg1_desk/getPcdName.sh ");
    command = command + boost::lexical_cast<std::string>(index);
    std::string filePath=bash::exec( const_cast<char*>(command.c_str()) );
    filePath.erase(std::remove(filePath.begin(), filePath.end(), '\n'), filePath.end());
    if( pcl::io::loadPCDFile<pcl::PointXYZRGB>(filePath, cloud) == -1 ) {

        return false;

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

/** save keypoints and descriptors of each cloud in two std::maps */
void saveCloudDescriptors(int cloudIndex, const pcl::PointCloud<pcl::PointXYZRGB>& cloudA) {

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

/** uses SURF for comparison of two clouds, saves the feature distance in featureDist parameter and detected points pixel distance in pixelDist paramter **/
void visualDistance( const int indexA, const int indexB, float& featureDist, int& pixelDist ) {

    using namespace cv;

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


/** loads different captures (.pcd files), align them with customICP and write them aligned in a single file (outFile) **/
void  alignAndView( pcl::visualization::PCLVisualizer* viewer, char* path, int min, int max, char* outFile, char* global ) {

    CustomICP icp;
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

    saveCloudDescriptors(min,prevCloud);
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


    GraphOptimizer_G2O optimizer;
    std::deque< pcl::PointCloud<pcl::PointXYZRGB> > cloudQueue;
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

    bool automatic=true;
    //to have one pose / vertex per cloud, the first cloud is in the origin with no rotation
    poses.push_back( Eigen::Matrix4f::Identity()  );
    writeTransformationQuaternion(Eigen::Matrix4f::Identity() , std::string(outFile) + std::string(".movements"));

    optimizer.addVertex( transf  );
    int prev_i=-1;
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
            saveCloudDescriptors(i,currCloud);

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

            if( showCorrespondences ) {
                float visualDist;
                int pixelDist;

                visualDistance(i,prevCloudIndex,visualDist,pixelDist);
                std::cout << "Visual dist: " <<  visualDist << "pixel DIst:" << pixelDist << "\n";
                std::cout << "currCloud: " <<  i << "prevCloud:" << prevCloudIndex << "\n";
            }

            //start ICP
            icp.setInputSource(currCloud.makeShared());
            icp.setInputTarget(prevCloud.makeShared());
            pcl::PointCloud<pcl::PointXYZRGB> finalCloud(640,480);;
            icp.align (finalCloud);

            //update global transformation
            Eigen::Matrix4f icpTransf = icp.getFinalTransformation();
            //round to two decimals
            icpTransf.col(3)[0] = roundf(icpTransf.col(3)[0]*100.0f)/100.0f;
            icpTransf.col(3)[1] = roundf(icpTransf.col(3)[1]*100.0f)/100.0f;
            icpTransf.col(3)[2] = roundf(icpTransf.col(3)[2]*100.0f)/100.0f;

            transf = icpTransf.cast<double>() * transf;
            //round to two decimals
            transf.col(3)[0] = roundf(((float)transf.col(3)[0])*100.0f)/100.0f;
            transf.col(3)[1] = roundf(((float)transf.col(3)[1])*100.0f)/100.0f;
            transf.col(3)[2] = roundf(((float)transf.col(3)[2])*100.0f)/100.0f;

            reorthogonalizeMatrix(transf);
            //std::cout << "Fitness: " << icp.getFitnessScore() << " - Correspondences: " <<  icp.getCorrespondences().size() << "\n";

            /**/
            int vertexID = optimizer.addVertex(transf);
            //std::cout << "VERTEX ID::: " << vertexID << "\n";
            poses.push_back(transf.cast<float>());

            if( vertexID > 0 ) {
                int fromIndex=vertexID-1;
                //add graph consecutive edges to optimizer
                //mem leak
                Eigen::Matrix4f* relPose = new Eigen::Matrix4f;
                Eigen::Matrix<double,6,6>* infMatrix = new Eigen::Matrix<double,6,6>;

                //optimizer.genEdgeData(iter->makeShared(),currCloud.makeShared(),*relPose,*infMatrix);
                optimizer.genEdgeData(prevCloud.makeShared(),currCloud.makeShared(),*relPose,*infMatrix);
                if( *relPose != Eigen::Matrix4f::Identity() ) {
                    //std::cout << "ADDING EDGE " << fromIndex << " - " << vertexID << "\n";

                    optimizer.addEdge(vertexID,fromIndex,*relPose,*infMatrix);
                    //save edge on .txt files
                    writeEdge(vertexID,fromIndex,*relPose,*infMatrix,outFile);

                }

                //add loop closure edges!
                /**/
                for( int k=(i-2); k >= (min +1); k=k-1 ) {
                    const float MAX_METERS = 1;
                    int currPoseIndex = i - min;
                    int pastPoseIndex = k - min;

                    if ( matrixDistance(poses.at(currPoseIndex),poses.at(pastPoseIndex)) < MAX_METERS  && prev_i != (i-1) )  {

                        pcl::PointCloud<pcl::PointXYZRGB> pastCloud(640,480);
                        if( readCloud(k,path,pastCloud)  ) {
                            float visualDist;
                            int pixelDist;
                            visualDistance(i,k,visualDist,pixelDist);
                            std::cout << "Visual dist: " <<  visualDist << "pixel DIst:" << pixelDist << "\n";
                            std::cout << "i: " <<  i << "k:" << k << "\n";
                            if( (visualDist < 0.2 && pixelDist < 50) || (visualDist < 0.28 && pixelDist < 30) ) {

                                Eigen::Matrix4f* relPose = new Eigen::Matrix4f;
                                Eigen::Matrix<double,6,6>* infMatrix = new Eigen::Matrix<double,6,6>;
                                optimizer.genEdgeData(pastCloud.makeShared(),currCloud.makeShared(),*relPose,*infMatrix);

                                if( *relPose != Eigen::Matrix4f::Identity() ) {
                                    saveCloudImage(currCloud, "cloud_" +  boost::lexical_cast<std::string>(i) + "-" + boost::lexical_cast<std::string>(k) + ".jpg" );
                                    saveCloudImage(pastCloud, "cloud_" +  boost::lexical_cast<std::string>(k) + "-" + boost::lexical_cast<std::string>(i) + ".jpg" );
                                    std::cout << "ADDING LOOP CLOSURE " << k-1 << " - " << i-1 << "\n";
                                    optimizer.addEdge(currPoseIndex,pastPoseIndex,*relPose,*infMatrix);
                                    //save edge on .txt files
                                    writeEdge(currPoseIndex,pastPoseIndex,*relPose,*infMatrix,outFile);
                                    prev_i  = i;

                                }
                            }
                        }
                    }
                }

            }
            /**/
            /**/
            //writeTransformation(icp.getFinalTransformation(), std::string(outFile) + std::string(".movements"));
            writeTransformationQuaternion(transf.cast<float>(), std::string(outFile) + std::string(".movements"));
            pcl::transformPointCloud(currCloud,finalCloud,transf.cast<float>());
            //std::cout << "Global transformation \n" << transf << "\n";

            prevCloud.clear();
            pcl::copyPointCloud(currCloud,prevCloud);

            //std::cout << "FINAL CLOUD SIZE:   " << finalCloud.size() << "\n\n";
            //mergeClouds(globalCloud,finalCloud);
            if( showCorrespondences )
                globalCloud = globalCloud + finalCloud;

            //std::cout << "GLOBAL CLOUD SIZE:   " << globalCloud.size() << "\n\n";

            //std::cout << "Global cloud with: " << globalCloud.points.size() << "\n";

            finalCloud.clear();

            if( showCorrespondences ) {

                showCorrespondences = false;
                viewer->addPointCloud(globalCloud.makeShared());
                /** Show Corresp
                pcl::Correspondences cor = icp.getCorrespondences();
                viewer->addPointCloud(icp.getSourceFiltered().makeShared(),"source");
                viewer->addPointCloud(icp.getTargetFiltered().makeShared(),"target");
                std::cout << "correspondences finded: " << cor.size() << "\n";
                for(int k=0; k < cor.size()/5; k++) {
                    pcl::Correspondence corresp = cor.at(k);
                    viewer->addLine( icp.getSourceFiltered().points[corresp.index_query],icp.getTargetFiltered().points[corresp.index_match],rand_alnum_str(k+2) );
                }
                /**/

            } else {
                /** Apply Voxel Filter*
                voxelFilter.setInputCloud(globalCloud.makeShared());
                voxelFilter.filter(globalCloud);
                /**/
                viewer->removeAllPointClouds();
                //viewer->addPointCloud(globalCloud.makeShared());

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

    while( !viewer->wasStopped() ) {

        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));

    }
    /**/
    optimizer.optimizeGraph();
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >  poses;
    optimizer.getPoses(poses);


    for( std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >::iterator iter = poses.begin(); iter != poses.end(); iter++ ) {
        writeTransformationQuaternion(*iter,std::string(outFile) + std::string(".optimized"));
    }
    /**/
    std::string prevTransfFile(outFile);
    prevTransfFile += ".prevtransf";
    writeTransformationMatrix(icp.getFinalTransformation(),prevTransfFile);
    saveState(globalCloud,transf.cast<float>(),outFile);
    return;

    /*
    while( !viewer->wasStopped() ) {

        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));

    }

    voxelFilter.setInputCloud(globalCloud.makeShared());
    voxelFilter.filter(globalCloud);


    pcl::io::savePCDFileBinary (outFile, globalCloud);
    std::cerr << "Saved " << globalCloud.points.size () << " data points to " << outFile << "\n"; */

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
