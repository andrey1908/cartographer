#pragma once

#include <vector>
#include <list>
#include <map>
#include <utility>

#include "cartographer/common/time.h"
#include "cartographer/mapping/id.h"
#include "cartographer/transform/rigid_transform.h"
#include "cartographer/mapping/constraint.h"
#include "cartographer/mapping/internal/optimization/optimization_problem_3d.h"
#include "cartographer/mapping/trajectory_node.h"
#include "cartographer/mapping/pose_graph.h"

#include "cartographer/mapping/proto/loop_trimmer_options.pb.h"

namespace cartographer {
namespace mapping {

class LoopsTrimmer {
public:
  LoopsTrimmer(bool log_number_of_trimmed_loops);
  ~LoopsTrimmer() = default;

  void AddLoopsTrimmerOptions(int trajectory_id, const proto::LoopTrimmerOptions& options);

  void AddNode(const NodeId& node, double accum_rotation, double travelled_distance);
  void AddSubmap(const SubmapId& submap, const NodeId& first_node_in_submap);
  void AddLoop(const Constraint& new_loop);
  void AddLoops(const std::vector<Constraint>& new_loops);

  void RemoveLoop(const SubmapId& submap, const NodeId& node);

  std::vector<Constraint> TrimFalseDetectedLoops(
      const std::vector<Constraint>& new_loops,
      const MapById<SubmapId, InternalSubmapData>& global_submap_poses_3d,
      const MapById<NodeId, TrajectoryNode>& trajectory_nodes) const;
  void TrimCloseLoops(std::vector<Constraint>& constraints);

  void TrimFalseLoops(
      std::vector<Constraint>& constraints,
      double max_rotation_error, double max_translation_error,
      const MapById<SubmapId, InternalSubmapData>& global_submap_poses_3d,
      const MapById<NodeId, TrajectoryNode>& trajectory_nodes);

private:
  std::pair<double, double> GetARTDWithLoops(
      const NodeId& node_1, const NodeId& node_2, float min_score) const;
  std::pair<double, double> GetLoopError(
      const Constraint& loop,
      const MapById<SubmapId, InternalSubmapData>& global_submap_poses_3d,
      const MapById<NodeId, TrajectoryNode>& trajectory_nodes) const;

private:
  bool log_number_of_trimmed_loops_;

  std::map<int, proto::LoopTrimmerOptions> loop_trimmer_options_;

  // key: [(submap, node), (node_1, node_2)], node_1 < node_2
  // value: score
  std::map<std::tuple<SubmapId, NodeId, NodeId, NodeId>, float> loops_;
  std::map<SubmapId, NodeId> submap_to_node_id_;

  std::map<NodeId, std::pair<double, double>> artds_;
};

}
}
