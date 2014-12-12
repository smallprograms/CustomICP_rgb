#ifndef G2O_PHOTOEDGE_SE3_H_
#define G2O_PHOTOEDGE_SE3_H_

#include "g2o/core/base_binary_edge.h"

#include "g2o/types/slam3d/g2o_types_slam3d_api.h"
#include "g2o/types/slam3d/vertex_se3.h"

namespace g2o {

  /**
   * \brief Edge between two 3D pose vertices
   *
   * The transformation between the two vertices is given as an Isometry3d.
   * If z denotes the measurement, then the error function is given as follows:
   * z^-1 * (x_i^-1 * x_j)
   */
  class G2O_TYPES_SLAM3D_API PhotoEdge : public BaseBinaryEdge<6, Eigen::Isometry3d, VertexSE3, VertexSE3> {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
      PhotoEdge();
      virtual bool read(std::istream& is);
      virtual bool write(std::ostream& os) const;

      void computeError();

      virtual void setMeasurement(const Eigen::Isometry3d& m){
        _measurement = m;
        _inverseMeasurement = m.inverse();
      }

      virtual bool setMeasurementData(const double* d){
        Map<const Vector7d> v(d);
        setMeasurement(internal::fromVectorQT(v));
        return true;
      }

      virtual bool getMeasurementData(double* d) const{
        Map<Vector7d> v(d);
        v = internal::toVectorQT(_measurement);
        return true;
      }

      void setOffset(int offset_) {
          offset = offset_;
      }

      void setCloudsPath(char *p) {
          cloudsPath = p;
      }

      void linearizeOplus();

      virtual int measurementDimension() const {return 7;}

      virtual bool setMeasurementFromState() ;

      virtual double initialEstimatePossible(const OptimizableGraph::VertexSet& /*from*/, 
          OptimizableGraph::Vertex* /*to*/) { 
        return 1.;
      }

      virtual void initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* to);

    protected:
      Eigen::Isometry3d _inverseMeasurement;
      int offset; //sum it to vertexs ids to know point cloud indexs!
      char* cloudsPath; //path to read point clouds from hd
  };

  /**
   * \brief Output the pose-pose constraint to Gnuplot data file
   */
  class G2O_TYPES_SLAM3D_API PhotoEdgeWriteGnuplotAction: public WriteGnuplotAction {
  public:
    PhotoEdgeWriteGnuplotAction();
    virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element, 
            HyperGraphElementAction::Parameters* params_);
  };

#ifdef G2O_HAVE_OPENGL
  /**
   * \brief Visualize a 3D pose-pose constraint
   */
  class G2O_TYPES_SLAM3D_API PhotoEdgeDrawAction: public DrawAction{
  public:
    PhotoEdgeDrawAction();
    virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element, 
            HyperGraphElementAction::Parameters* params_);
  };
#endif

} // end namespace
#endif
