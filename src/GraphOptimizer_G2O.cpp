#define ENABLE_DEBUG_G2O 0 //Don't enable if not necessary

#include "GraphOptimizer_G2O.h"
#include "PhotoEdge.h"

#if ENABLE_DEBUG_G2O
#include "../include/Miscellaneous.h" //Save matrix
#endif

GraphOptimizer_G2O::GraphOptimizer_G2O()
{

    optimizer.setVerbose(true);
    // variable-size block solver
    /**/
    //g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setWriteDebug(true);
    solver->setUserLambdaInit(0.01);
    optimizer.setAlgorithm(solver);
    /**/


    //Set the vertex index to 0
    vertexIdx=0;
}

int GraphOptimizer_G2O::addVertex(Eigen::Matrix4d& vertexPose, int id, bool isFixed)
{
    static double yaw,pitch,roll;

//    #if ENABLE_DEBUG_G2O
////    char* fileName = (char*) malloc(100);
////    std::sprintf(fileName,"../results/matrices/pose_%i.txt",vertexIdx);
////    Miscellaneous::saveMatrix(vertexPose.cast<float>(),fileName);
////    delete fileName;
//    #endif

    //Transform Eigen::Matrix4f into 3D traslation and rotation for g2o
    yaw = atan2f((float)vertexPose(1,0),(float)vertexPose(0,0));
    pitch = asinf(-(float)vertexPose(2,0));
    roll = atan2f((float)vertexPose(2,1),(float)vertexPose(2,2));

    g2o::Vector3d t(vertexPose(0,3),vertexPose(1,3),vertexPose(2,3));
    g2o::Quaterniond q;
    q.x()=sin(roll/2)*cos(pitch/2)*cos(yaw/2)-cos(roll/2)*sin(pitch/2)*sin(yaw/2);
    q.y()=cos(roll/2)*sin(pitch/2)*cos(yaw/2)+sin(roll/2)*cos(pitch/2)*sin(yaw/2);
    q.z()=cos(roll/2)*cos(pitch/2)*sin(yaw/2)-sin(roll/2)*sin(pitch/2)*cos(yaw/2);
    q.w()=cos(roll/2)*cos(pitch/2)*cos(yaw/2)+sin(roll/2)*sin(pitch/2)*sin(yaw/2);

    //g2o::SE3Quat pose(q,t);	// vertex pose

    // set up node
    g2o::VertexSE3 *vc = new g2o::VertexSE3();
    Eigen::Isometry3d cam; // camera pose
    cam = q;
    cam.translation() = t;

    vc->setEstimate(cam);
    vc->setId(id);      // vertex id

    //set pose fixed
    if (isFixed) {
        vc->setFixed(true);
    }

    // add to optimizer
    optimizer.addVertex(vc);
    vertexIdVec.push_back(id);

    return id;
}

void GraphOptimizer_G2O::addEdge(const int fromIdx,
                                 const int toIdx,
                                 Eigen::Matrix4f& relativePose,
                                 Eigen::Matrix<double,6,6>& informationMatrix)
{
    static double yaw,pitch,roll;

    #if ENABLE_DEBUG_G2O
    char* fileName = (char*) malloc(100);
    std::sprintf(fileName,"../results/matrices/edge_%i_to_%i.txt",fromIdx,toIdx);
    Miscellaneous::saveMatrix(relativePose,fileName);
    delete fileName;
    #endif

    //Transform Eigen::Matrix4f into 3D traslation and rotation for g2o
    yaw = atan2f(relativePose(1,0),relativePose(0,0));
    pitch = asinf(-relativePose(2,0));
    roll = atan2f(relativePose(2,1),relativePose(2,2));

    g2o::Vector3d t(relativePose(0,3),relativePose(1,3),relativePose(2,3));
    g2o::Quaterniond q;
    q.x()=sin(roll/2)*cos(pitch/2)*cos(yaw/2)-cos(roll/2)*sin(pitch/2)*sin(yaw/2);
    q.y()=cos(roll/2)*sin(pitch/2)*cos(yaw/2)+sin(roll/2)*cos(pitch/2)*sin(yaw/2);
    q.z()=cos(roll/2)*cos(pitch/2)*sin(yaw/2)-sin(roll/2)*sin(pitch/2)*cos(yaw/2);
    q.w()=cos(roll/2)*cos(pitch/2)*cos(yaw/2)+sin(roll/2)*sin(pitch/2)*sin(yaw/2);

    //std::cout << "RELATIVE POSE:::: \n\n" << relativePose << "\n";


    g2o::Quaternionf q2;
    Eigen::Matrix3f rot = relativePose.block(0,0,3,3);
    q2 = rot;


    g2o::SE3Quat transf(q,t);	// relative transformation

    g2o::EdgeSE3* edge = new g2o::EdgeSE3;
    //g2o::PhotoEdge* edge = new g2o::PhotoEdge;
    edge->vertices()[0] = optimizer.vertex(fromIdx);
    edge->vertices()[1] = optimizer.vertex(toIdx);
    edge->setMeasurement(transf);
//    edge->setOffset(offset); //to recover cloud id
//    edge->setCloudsPath(cloudsPath); //place where point clouds are in hd
    //Set the information matrix to identity
    edge->setInformation(informationMatrix);

    optimizer.addEdge(edge);
}

void GraphOptimizer_G2O::optimizeGraph()
{

    //Prepare and run the optimization
    std::cout << "PREPARING OPT:: " << optimizer.initializeOptimization() << "\n";

    //Set the initial Levenberg-Marquardt lambda
    //optimizer.setUserLambdaInit(0.01);

    optimizer.setVerbose(true);
    //Run optimization
    std::cout << "OPTIMIZE ITERATIONS::::: " << optimizer.optimize(100) << "\n";


}

void GraphOptimizer_G2O::getPoses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f > >& poses)
{
    poses.clear();
    bool firstVertex=false;
    for( int k=1; ; k++) {


        //Transform the vertex pose from G2O quaternion to Eigen::Matrix4f
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(optimizer.vertex( k ));
        if( vertex != NULL ) {
            firstVertex = true;
        } else {
            if( firstVertex || k > 100000 ) break;
            continue;
        }
        double optimizedPoseQuaternion[7];
        vertex->getEstimateData(optimizedPoseQuaternion);

        Eigen::Matrix4f optimizedPose;
        static double qx,qy,qz,qr,qx2,qy2,qz2,qr2;

        qx=optimizedPoseQuaternion[3];
        qy=optimizedPoseQuaternion[4];
        qz=optimizedPoseQuaternion[5];
        qr=optimizedPoseQuaternion[6];
        qx2=qx*qx;
        qy2=qy*qy;
        qz2=qz*qz;
        qr2=qr*qr;

        optimizedPose(0,0)=qr2+qx2-qy2-qz2;
        optimizedPose(0,1)=2*(qx*qy-qr*qz);
        optimizedPose(0,2)=2*(qz*qx+qr*qy);
        optimizedPose(0,3)=optimizedPoseQuaternion[0];
        optimizedPose(1,0)=2*(qx*qy+qr*qz);
        optimizedPose(1,1)=qr2-qx2+qy2-qz2;
        optimizedPose(1,2)=2*(qy*qz-qr*qx);
        optimizedPose(1,3)=optimizedPoseQuaternion[1];
        optimizedPose(2,0)=2*(qz*qx-qr*qy);
        optimizedPose(2,1)=2*(qy*qz+qr*qx);
        optimizedPose(2,2)=qr2-qx2-qy2+qz2;
        optimizedPose(2,3)=optimizedPoseQuaternion[2];
        optimizedPose(3,0)=0;
        optimizedPose(3,1)=0;
        optimizedPose(3,2)=0;
        optimizedPose(3,3)=1;

        //Set the optimized pose to the vector of poses
        //poses[k]=optimizedPose;
        poses.push_back(optimizedPose);
    }
}

void GraphOptimizer_G2O::saveGraph(std::string fileName)
{
        //Save the graph to file
    optimizer.save(fileName.c_str(),0);
}

void GraphOptimizer_G2O::loadGraph(std::string fileName)
{
    if( optimizer.load(fileName.c_str()) ) {
        std::cout << fileName << " LOADED\n\n";
    } else {
        std::cout << "FAIL \n\n";
    }
}
void GraphOptimizer_G2O::fillInformationMatrix(Eigen::Matrix<double,6,6>&  infMatrix, float photoCons) {
    float weight = photoCons;
    if( weight < 10 ) weight = 0;
    if( weight > 150 ) weight = 150;
    weight = weight/150;
    weight = 1 - weight; //1 is perfect, 0 is very bad
    infMatrix = Eigen::Matrix<double,6,6>::Identity()*weight;
}

void GraphOptimizer_G2O::genEdgeData(Eigen::Matrix4f guess, pcl::PointCloud<pcl::PointXYZRGB>::Ptr  src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt,  Eigen::Matrix4f& relPose, Eigen::Matrix<double,6,6>&  infMatrix, float& photoCons)
{
    CustomICP icp;
    pcl::PointCloud<pcl::PointXYZRGB> notUsed(640,480);
    icp.setOflowStop(true);
    icp.setInputSource(src);
    icp.setInputTarget(tgt);
    icp.align(notUsed, guess);
    relPose = icp.getFinalTransformation();
    photoCons = icp.getPhotoConsistency();
    fillInformationMatrix(infMatrix,photoCons);
}

