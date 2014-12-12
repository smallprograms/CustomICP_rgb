#include "PhotoEdge.h"
#include "g2o/types/slam3d/isometry3d_gradients.h"
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "g2o/stuff/opengl_wrapper.h"

//helper function defined in main.cpp
bool readCloud(int index, char* path, pcl::PointCloud<pcl::PointXYZRGB>& cloud);
//function from customICP.cpp
float photoConsistency(pcl::PointCloud<pcl::PointXYZRGB> &cloudSrc, pcl::PointCloud<pcl::PointXYZRGB> &cloudTgt,Eigen::Matrix4f transf);

namespace g2o {
  using namespace std;

  PhotoEdge::PhotoEdge() : BaseBinaryEdge<6, Eigen::Isometry3d, VertexSE3, VertexSE3>() {
    information().setIdentity();
  }

  bool PhotoEdge::read(std::istream& is) {
    Vector7d meas;
    for (int i=0; i<7; i++) 
      is >> meas[i];
    // normalize the quaternion to recover numerical precision lost by storing as human readable text
    Vector4d::MapType(meas.data()+3).normalize();
    setMeasurement(internal::fromVectorQT(meas));

    if (is.bad()) {
      return false;
    }
    for ( int i=0; i<information().rows() && is.good(); i++)
      for (int j=i; j<information().cols() && is.good(); j++){
        is >> information()(i,j);
        if (i!=j)
          information()(j,i)=information()(i,j);
      }
    if (is.bad()) {
      //  we overwrite the information matrix with the Identity
      information().setIdentity();
    } 
    return true;
  }

  bool PhotoEdge::write(std::ostream& os) const {
    Vector7d meas=internal::toVectorQT(_measurement);
    for (int i=0; i<7; i++) os  << meas[i] << " ";
    for (int i=0; i<information().rows(); i++)
      for (int j=i; j<information().cols(); j++) {
        os <<  information()(i,j) << " ";
      }
    return os.good();
  }

  void PhotoEdge::computeError() {
    VertexSE3 *from = static_cast<VertexSE3*>(_vertices[0]);
    VertexSE3 *to   = static_cast<VertexSE3*>(_vertices[1]);
    Eigen::Isometry3d delta =  from->estimate().inverse()*to->estimate();
    Eigen::Matrix4f transf = (delta.matrix()).cast<float>();
    pcl::PointCloud<pcl::PointXYZRGB> cloudFrom;
    pcl::PointCloud<pcl::PointXYZRGB> cloudTo;
    if( !readCloud(_vertices[0]->id()+offset,cloudsPath,cloudFrom) ||
            !readCloud(_vertices[1]->id()+offset,cloudsPath,cloudTo)) {
        std::cout << "ERROR READING CLOUDS at PhotoEdge::computeError!!\n";
    } else {
        float photoCons = photoConsistency(cloudFrom,cloudTo,transf);
        _error << photoCons,photoCons,photoCons,photoCons,photoCons,photoCons;
//        std::cout << "clouds: " << _vertices[0]->id()+offset << "::" << _vertices[1]->id()+offset << "\n";
//        std::cout << "transf:: " << transf << "\n";
//        std::cout << "ERROR::: " << _error << "\n";
    }
  }

  bool PhotoEdge::setMeasurementFromState(){
    VertexSE3 *from = static_cast<VertexSE3*>(_vertices[0]);
    VertexSE3 *to   = static_cast<VertexSE3*>(_vertices[1]);
    Eigen::Isometry3d delta = from->estimate().inverse() * to->estimate();
    setMeasurement(delta);
    return true;
  }
  
  void PhotoEdge::linearizeOplus(){
    
    // BaseBinaryEdge<6, Eigen::Isometry3d, VertexSE3, VertexSE3>::linearizeOplus();
    // return;

    VertexSE3 *from = static_cast<VertexSE3*>(_vertices[0]);
    VertexSE3 *to   = static_cast<VertexSE3*>(_vertices[1]);
    Eigen::Isometry3d E;
    const Eigen::Isometry3d& Xi=from->estimate();
    const Eigen::Isometry3d& Xj=to->estimate();
    const Eigen::Isometry3d& Z=_measurement;
    internal::computeEdgeSE3Gradient(E, _jacobianOplusXi , _jacobianOplusXj, Z, Xi, Xj);
  }

  void PhotoEdge::initialEstimate(const OptimizableGraph::VertexSet& from_, OptimizableGraph::Vertex* /*to_*/) {
    VertexSE3 *from = static_cast<VertexSE3*>(_vertices[0]);
    VertexSE3 *to   = static_cast<VertexSE3*>(_vertices[1]);

    if (from_.count(from) > 0) {
      to->setEstimate(from->estimate() * _measurement);
    } else
      from->setEstimate(to->estimate() * _measurement.inverse());
    //cerr << "IE" << endl;
  }

  PhotoEdgeWriteGnuplotAction::PhotoEdgeWriteGnuplotAction(): WriteGnuplotAction(typeid(PhotoEdge).name()){}

  HyperGraphElementAction* PhotoEdgeWriteGnuplotAction::operator()(HyperGraph::HyperGraphElement* element, HyperGraphElementAction::Parameters* params_){
    if (typeid(*element).name()!=_typeName)
      return 0;
    WriteGnuplotAction::Parameters* params=static_cast<WriteGnuplotAction::Parameters*>(params_);
    if (!params->os){
      std::cerr << __PRETTY_FUNCTION__ << ": warning, on valid os specified" << std::endl;
      return 0;
    }

    PhotoEdge* e =  static_cast<PhotoEdge*>(element);
    VertexSE3* fromEdge = static_cast<VertexSE3*>(e->vertices()[0]);
    VertexSE3* toEdge   = static_cast<VertexSE3*>(e->vertices()[1]);
    Vector6d fromV, toV;
    fromV=internal::toVectorMQT(fromEdge->estimate());
    toV=internal::toVectorMQT(toEdge->estimate());
    for (int i=0; i<6; i++){
      *(params->os) << fromV[i] << " ";
    }
    for (int i=0; i<6; i++){
      *(params->os) << toV[i] << " ";
    }
    *(params->os) << std::endl;
    return this;
  }

}
