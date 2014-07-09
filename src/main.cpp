#include <iostream>
#include <sstream>
#include <cmath>
#include "CustomICP.h"
#include "GraphOptimizer_G2O.h"

//flag used to press a key to process next capture
bool doNext = false;
bool saveAndQuit = false;
bool loadedGlobal = false;
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

    std::cout << transf.col(0)[0];
    Eigen::Matrix3f rot = transf.block(0,0,3,3);
    Eigen::Quaternionf quat(rot);
    std::ofstream file(fileName.c_str(), ios::out | ios::app);
    file << transf.col(3)[0] << " " << transf.col(3)[1] << " " << transf.col(3)[2] << " " << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << "\n";
    file.close();
}

void writeTransformationMatrix(Eigen::Matrix4f transf, std::string fileName) {

    std::ofstream ofs(fileName.c_str());
    if( ofs.is_open() ) {
        std::cout << "write to " << fileName << "\n";
        ofs << transf;
    }
    ofs.close();
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


void groundTruthAlign(char* path,int min,int max,char* outFile,char* ground ) {

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
    std::stringstream ss;

    int i = min;

    while( groundFile >> tx >> ty >> tz >> qx >> qy >> qz >> qw ) {

        //generate transformation matrix
        Eigen::Quaternion<float> quat (qw,qx,qy,qz);
        Eigen::Matrix3f rot = quat.toRotationMatrix();
        transf = Eigen::Matrix4f::Identity();
        transf << rot(0,0),rot(0,1),rot(0,2),0,rot(1,0),rot(1,1),rot(1,2),0,rot(2,0),rot(2,1),rot(2,2),0,0,0,0,1;
        Eigen::Vector3f v(tx,ty,tz);
        transf = transf + Eigen::Affine3f(Eigen::Translation3f(v)).matrix();
        transf(3,3) = 1;
        std::cout << transf << "\n";
        ss.str(""); //reset string
        ss << path << "/cap" << i << ".pcd";

        std::cout <<  "reading " << ss.str() << "\n";

        //read current cloud from file
        if( i > max || pcl::io::loadPCDFile<pcl::PointXYZRGB>(ss.str(), currCloud) == -1 ) {

            std::cout << "Reading end at " << i << "\n";
            break;

        }

        /** end read next cloud **/
        pcl::transformPointCloud(currCloud,finalCloud,transf);
        //mergeClouds(globalCloud,finalCloud);
        globalCloud = globalCloud + finalCloud;
        i++;
    }

    pcl::io::savePCDFileBinary (outFile, globalCloud);
    std::cerr << "Saved " << globalCloud.points.size () << " data points to " << outFile << "\n";

}

/** loads different captures (.pcd files), align them with customICP and write them aligned in a single file (outFile) **/
void  alignAndView( pcl::visualization::PCLVisualizer* viewer, char* path, int min, int max, char* outFile, char* global ) {

    CustomICP icp;
    pcl::FastBilateralFilter<pcl::PointXYZRGB> fastBilFilter;
    pcl::VoxelGrid<pcl::PointXYZRGB> voxelFilter;
    voxelFilter.setLeafSize(0.0025,0.0025,0.0025);

    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
    //to accumulate ICP transformations
    static Eigen::Matrix4f transf = Eigen::Matrix4f::Identity ();

    //if loading a previous reconstruction, transf is not identity!
    if( loadedGlobal ) {

        std::string transfName(global);
        std::string prevTransfName(global);
        transfName += ".transf";
        transf = loadTransformationMatrix(transfName);
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

    //name of cloud file
    std::stringstream ss;
    ss << path << "/cap" << min << ".pcd";
    if( pcl::io::loadPCDFile<pcl::PointXYZRGB>(ss.str(), prevCloud) == -1 ) {

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


    GraphOptimizer_G2O optimizer;
    std::deque< pcl::PointCloud<pcl::PointXYZRGB> > cloudQueue;
    //cloudQueue.resize(4);


    //read file by file
    for(int i=min+1; i <= max; i++) {


        //go ahead if user press n
        if( doNext || true ) {

            doNext = false; 

            /** read next cloud **/
            ss.str(""); //reset string
            ss << path << "/cap" << i << ".pcd";

            pcl::PointCloud<pcl::PointXYZRGB> currCloud(640,480);

            std::cout <<  "reading " << ss.str() << "\n";

            //read current cloud from file
            if( pcl::io::loadPCDFile<pcl::PointXYZRGB>(ss.str(), currCloud) == -1 ) {

                std::cout << "Reading end at " << i << "\n";
                break;

            }

            /** end read next cloud **/


            //apply voxel filter to curr cloud
//            voxelFilter.setInputCloud(currCloudNotDense.makeShared());
//            voxelFilter.filter(currCloudNotDense);
            //Apply bilateral filter
            fastBilFilter.setInputCloud(currCloud.makeShared());
            fastBilFilter.filter(currCloud);

            //start ICP
            icp.setInputSource(currCloud.makeShared());
            icp.setInputTarget(prevCloud.makeShared());
            pcl::PointCloud<pcl::PointXYZRGB> finalCloud(640,480);;
            icp.align (finalCloud);
            pcl::copyPointCloud(currCloud,prevCloud);
            std::cout << "bef update curr " << i << transf << "\n";
            //update global transformation
            transf = icp.getFinalTransformation() * transf;
            std::cout << "af update curr " << i << transf << "\n";
            int vertexID = optimizer.addVertex(transf);

            if( vertexID > 0 ) {
                int toIndex=vertexID-1;
                //add edges to optimizer
                //read cloud by cloud starting by the newest
                for(std::deque< pcl::PointCloud<pcl::PointXYZRGB> >::reverse_iterator iter = cloudQueue.rbegin();
                    iter != cloudQueue.rend(); iter++ ) {

                    Eigen::Matrix4f* relPose = new Eigen::Matrix4f;
                    Eigen::Matrix<double,6,6>* infMatrix = new Eigen::Matrix<double,6,6>;
                    optimizer.genEdgeData(currCloud.makeShared(),iter->makeShared(),*relPose,*infMatrix);
                    optimizer.addEdge(vertexID, toIndex,*relPose,*infMatrix);
                    toIndex--;

                }
            }

            //add cloud to queue
            cloudQueue.push_back(currCloud);
            const int QSIZE=3;
            //mantain a constant size
            if ( cloudQueue.size() >= QSIZE ) {
                cloudQueue.pop_front();
            }

            //writeTransformation(icp.getFinalTransformation(), std::string(outFile) + std::string(".movements"));
            writeTransformationQuaternion(transf, std::string(outFile) + std::string(".movements"));
            pcl::transformPointCloud(currCloud,finalCloud,transf);

            pcl::Correspondences cor = icp.getCorrespondences();
            //viewer->removeAllPointClouds();
            //viewer->removeAllShapes();
//            viewer->addPointCloud(icp.getSourceFiltered().makeShared(),"source");
//            viewer->addPointCloud(icp.getTargetFiltered().makeShared(),"target");

//            for(int k=0; k < cor.size(); k++) {
//                pcl::Correspondence corresp = cor.at(k);
//                viewer->addLine( icp.getSourceFiltered().points[corresp.index_query],icp.getTargetFiltered().points[corresp.index_match],rand_alnum_str(k+2) );
//            }

            prevCloud.clear();
            pcl::copyPointCloud(currCloud,prevCloud);
            //globalCloud = globalCloud + finalCloud;
            std::cout << "FINAL CLOUD SIZE:   " << finalCloud.size() << "\n\n";
            //COMMENTED TO JUST SAVE TRAJECTORY
            mergeClouds(globalCloud,finalCloud);

            std::cout << "GLOBAL CLOUD SIZE:   " << globalCloud.size() << "\n\n";

            std::cout << "Global cloud with: " << globalCloud.points.size() << "\n";

            finalCloud.clear();

            //viewer->addPointCloud(globalCloud.makeShared());


            if(saveAndQuit) {
                break;
            }

        } else {
            while( !viewer->wasStopped() ) {
                viewer->spinOnce (100);
                boost::this_thread::sleep (boost::posix_time::microseconds (100000));
                if( doNext || saveAndQuit ) break;

            }
            //dont skip the current capture
            --i;
        }
    }

    optimizer.optimizeGraph();
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >  poses;
    optimizer.getPoses(poses);


    for( std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >::iterator iter = poses.begin(); iter != poses.end(); iter++ ) {
        writeTransformationQuaternion(*iter,std::string(outFile) + std::string(".optimized"));
    }

    std::string prevTransfFile(outFile);
    prevTransfFile += ".prevtransf";
    writeTransformationMatrix(icp.getFinalTransformation(),prevTransfFile);
    saveState(globalCloud,transf,outFile);
    return;


    while( !viewer->wasStopped() ) {

        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));

    }

    voxelFilter.setInputCloud(globalCloud.makeShared());
    voxelFilter.filter(globalCloud);


    pcl::io::savePCDFileBinary (outFile, globalCloud);
    std::cerr << "Saved " << globalCloud.points.size () << " data points to " << outFile << "\n";

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

        return 0;

    } else {

        if(argc == 6) {

            min = atoi(argv[3]);
            max = atoi(argv[4]);
            path = argv[2];
            outFile = argv[5];

            if( endswith(argv[1], ".txt") ) {

                groundTruth = argv[1];
                groundTruthAlign( path, min, max, outFile, groundTruth );
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
