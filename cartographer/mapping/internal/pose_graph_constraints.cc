#include "cartographer/mapping/internal/pose_graph_constraints.h"

#include <type_traits>

namespace cartographer {
namespace mapping {

void PoseGraphConstraints::SetTravelledDistance(
    const NodeId& node_id, double travelled_distance) {
  CHECK(travelled_distance_.count(node_id) == 0);
  travelled_distance_.emplace(node_id, travelled_distance);
}

void PoseGraphConstraints::SetFirstNodeIdForSubmap(
      const NodeId& first_node_id, const SubmapId& submap_id) {
  CHECK(first_node_id_for_submap_.count(submap_id) == 0);
  first_node_id_for_submap_.emplace(submap_id, first_node_id);
}

double PoseGraphConstraints::GetTravelledDistanceWithLoops(
    const NodeId& node_1, const NodeId& node_2, float min_score) {
  if (node_1.trajectory_id == node_2.trajectory_id) {
    return GetTravelledDistanceWithLoopsSameTrajectory(node_1, node_2, min_score);
  } else {
    return GetTravelledDistanceWithLoopsDifferentTrajectories(node_1, node_2, min_score);
  }
}

double PoseGraphConstraints::GetTravelledDistanceWithLoopsSameTrajectory(
    NodeId node_1, NodeId node_2, float min_score) {
  CHECK(node_1.trajectory_id == node_2.trajectory_id);
  if (node_1 == node_2) {
    return 0.0;
  }
  if (node_2 < node_1) {
    std::swap(node_1, node_2);
  }
  CHECK(travelled_distance_.count(node_1));
  CHECK(travelled_distance_.count(node_2));
  double travelled_distance =
      travelled_distance_.at(node_2) - travelled_distance_.at(node_1);
  CHECK(travelled_distance >= 0.0);
  for (const Constraint& loop : constraints_) {
    if (loop.tag == Constraint::INTRA_SUBMAP) {
      continue;
    }
    if (loop.score < min_score) {
      continue;
    }
    // check that loop lies on the same trajectory as node_1 and node_2
    if (loop.submap_id.trajectory_id != node_1.trajectory_id ||
        loop.node_id.trajectory_id != node_1.trajectory_id) {
      continue;
    }
    CHECK(first_node_id_for_submap_.count(loop.submap_id));
    NodeId loop_node_1 = first_node_id_for_submap_.at(loop.submap_id);
    NodeId loop_node_2 = loop.node_id;
    if (loop_node_2 < loop_node_1) {
      std::swap(loop_node_1, loop_node_2);
    }
    // check that loops intersect
    if (loop_node_2.node_index <= node_1.node_index ||
        loop_node_1.node_index >= node_2.node_index) {
      continue;
    }
    CHECK(travelled_distance_.count(loop_node_1));
    CHECK(travelled_distance_.count(loop_node_2));
    double travelled_distance_with_loop =
        std::abs(travelled_distance_.at(loop_node_1) - travelled_distance_.at(node_1)) +
        std::abs(travelled_distance_.at(loop_node_2) - travelled_distance_.at(node_2));
    travelled_distance = std::min(travelled_distance, travelled_distance_with_loop);
  }
  return travelled_distance;
}

double PoseGraphConstraints::GetTravelledDistanceWithLoopsDifferentTrajectories(
    NodeId node_1, NodeId node_2, float min_score) {
  CHECK(node_1.trajectory_id != node_2.trajectory_id);
  if (node_2 < node_1) {
    std::swap(node_1, node_2);
  }

  double travelled_distance = std::numeric_limits<double>::max();
  for (const Constraint& loop : constraints_) {
    if (loop.tag == Constraint::INTRA_SUBMAP) {
      continue;
    }
    if (loop.score < min_score) {
      continue;
    }
    // check that loop connects the same trajectories on which
    // node_1 and node_2 lie
    if (std::minmax(loop.submap_id.trajectory_id, loop.node_id.trajectory_id) !=
        std::minmax(node_1.trajectory_id, node_2.trajectory_id)) {
      continue;
    }
    CHECK(first_node_id_for_submap_.count(loop.submap_id));
    NodeId loop_node_1 = first_node_id_for_submap_.at(loop.submap_id);
    NodeId loop_node_2 = loop.node_id;
    if (loop_node_2 < loop_node_1) {
      std::swap(loop_node_1, loop_node_2);
    }
    double travelled_distance_with_loop =
        GetTravelledDistanceWithLoopsSameTrajectory(node_1, loop_node_1, min_score) +
        GetTravelledDistanceWithLoopsSameTrajectory(node_2, loop_node_2, min_score);
    travelled_distance = std::min(travelled_distance, travelled_distance_with_loop);
  }

  for (const TrimmedLoop& loop : loops_from_trimmed_submaps_) {
    if (loop.score < min_score) {
      continue;
    }
    // check that loop connects the same trajectories on which
    // node_1 and node_2 lie
    if (std::minmax(loop.submap_id.trajectory_id, loop.node_id.trajectory_id) !=
        std::minmax(node_1.trajectory_id, node_2.trajectory_id)) {
      continue;
    }
    CHECK(first_node_id_for_submap_.count(loop.submap_id));
    NodeId loop_node_1 = first_node_id_for_submap_.at(loop.submap_id);
    NodeId loop_node_2 = loop.node_id;
    if (loop_node_2 < loop_node_1) {
      std::swap(loop_node_1, loop_node_2);
    }
    double travelled_distance_with_loop =
        GetTravelledDistanceWithLoopsSameTrajectory(node_1, loop_node_1, min_score) +
        GetTravelledDistanceWithLoopsSameTrajectory(node_2, loop_node_2, min_score);
    travelled_distance = std::min(travelled_distance, travelled_distance_with_loop);
  }
  return travelled_distance;
}

bool PoseGraphConstraints::IsLoopLast(const Constraint& loop) {
  CHECK(loop.tag == Constraint::INTER_SUBMAP);
  CHECK(last_loop_submap_index_for_trajectory_.count(loop.submap_id.trajectory_id));
  CHECK(last_loop_node_index_for_trajectory_.count(loop.node_id.trajectory_id));
  CHECK(last_loop_submap_index_for_trajectory_.at(loop.submap_id.trajectory_id) >=
      loop.submap_id.submap_index);
  CHECK(last_loop_node_index_for_trajectory_.at(loop.node_id.trajectory_id) >=
      loop.node_id.node_index);
  return
      (last_loop_submap_index_for_trajectory_.at(loop.submap_id.trajectory_id) ==
          loop.submap_id.submap_index) ||
      (last_loop_node_index_for_trajectory_.at(loop.node_id.trajectory_id) ==
          loop.node_id.node_index);
}

}
}
