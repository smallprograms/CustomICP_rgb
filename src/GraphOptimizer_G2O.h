#include "GraphOptimizer.h"

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/types/slam3d/types_slam3d.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"

#include "CustomICP.h"


/*!This class encapsulates the functionality of the G2O library to perform 6D graph optimization.*/
class GraphOptimizer_G2O
{
private:
    int vertexIdx;
    g2o::SparseOptimizer optimizer;
    std::vector<int> vertexIdVec;


public:
  GraphOptimizer_G2O();
  /*!Adds a new vertex to the graph. The provided 4x4 matrix will be considered as the pose of the new added vertex. It returns the index of the added vertex.*/
  int addVertex(Eigen::Matrix4d& pose, int id, bool isFixed=false);
  /*!Adds an edge that defines a spatial constraint between the vertices "fromIdx" and "toIdx" with information matrix that determines the weight of the added edge.
offset is summed to vertex indexs in order to know point cloud id!*/
  void addEdge(const int fromIdx, const int toIdx, Eigen::Matrix4f& relPose, Eigen::Matrix<double,6,6>& infMatrix);
  /*!Calls the graph optimization process to determine the pose configuration that best satisfies the constraints defined by the edges.*/
  void optimizeGraph();
  /*!Returns a vector with all the optimized poses of the graph.*/
  void getPoses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f > >&);
  /*!Saves the graph to file.*/
  void saveGraph(std::string fileName);
  /*!load the graph from  file.*/
  void loadGraph(std::string fileName);
  /*!Generate relPose and information matrix.*/
  void genEdgeData(pcl::PointCloud<pcl::PointXYZRGB>::Ptr src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt,Eigen::Matrix4f& relPose,Eigen::Matrix<double,6,6>& infMatrix, float& photoCons);

};
