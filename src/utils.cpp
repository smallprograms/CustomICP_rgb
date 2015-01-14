#include "utils.h"

void cloudToMat(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, cv::Mat& outMat){

    for(int w=0; w < cloud->width; w++) {
        for( int h=0; h < cloud->height; h++) {

                outMat.at<cv::Vec3b>(h,w)[0] = (*cloud)(w,h).b;
                outMat.at<cv::Vec3b>(h,w)[1] = (*cloud)(w,h).g;
                outMat.at<cv::Vec3b>(h,w)[2] = (*cloud)(w,h).r;
         }
    }

}

void setCloudAsNaN(pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
    for( size_t m=0; m < cloud.width;m++ ) {
        for( size_t n=0; n < cloud.height; n++) {
            cloud(m,n).x=cloud(m,n).y=cloud(m,n).z = std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void removeFarPoints(pcl::PointCloud<pcl::PointXYZRGB> &cloud, float z)
{
    for( size_t m=0; m < cloud.width;m++ ) {
        for( size_t n=0; n < cloud.height; n++) {
            if( cloud(m,n).z > z ) {
                cloud(m,n).x=cloud(m,n).y=cloud(m,n).z = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void filterMask(const pcl::PointCloud<pcl::PointXYZRGB>& cloudIn, cv::Mat mask,pcl::PointCloud<pcl::PointXYZRGB> &cloudOut) {

    for(size_t m=0; m < cloudIn.width; m++) {
        for(size_t n=0; n < cloudIn.height; n++) {


            if( cloudIn.at(m,n).z > 0.1 ) {

                if( mask.at<uchar>(n,m) > 0 ) {

                    cloudOut.at(m,n) = cloudIn.at(m,n);
                }

            }
        }
    }
}

void writeTransformationQuaternion(Eigen::Matrix4f transf, std::string fileName) {
    using namespace std;
    //std::cout << transf.col(0)[0];
    Eigen::Matrix3f rot = transf.block(0,0,3,3);
    Eigen::Quaternionf quat(rot);
    quat.normalize();
    std::ofstream file(fileName.c_str(), ios::out | ios::app);
    file << transf.col(3)[0] << " " << transf.col(3)[1] << " " << transf.col(3)[2] << " " << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << "\n";
    file.close();
}

void writeNumber(float number, std::string fileName) {
    using namespace std;
    std::ofstream file(fileName.c_str(), ios::out | ios::app);
    file << number << "\n";
    file.close();
}

void writeNumber(int number, std::string fileName) {
    using namespace std;
    std::ofstream file(fileName.c_str(), ios::out | ios::app);
    file << number << "\n";
    file.close();
}

void writeTwoNumbers(int fromIndex, int toIndex , std::string fileName) {
    using namespace std;
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

void reorthogonalizeMatrix(Eigen::Matrix4d& transf) {
    Eigen::Matrix3d rot = transf.block(0,0,3,3);
    Eigen::Quaterniond quat(rot);
    quat.normalize();
    transf.block(0,0,3,3) = quat.toRotationMatrix();
}

void writeTransformationMatrix(Eigen::Matrix4f transf, std::string fileName) {

    std::ofstream ofs(fileName.c_str());
    if( ofs.is_open() ) {
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

int endswith(const char* haystack, const char* needle) {
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

