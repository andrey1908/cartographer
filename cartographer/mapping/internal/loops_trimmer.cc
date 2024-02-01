#include "cartographer/mapping/internal/loops_trimmer.h"

#include "glog/logging.h"

#include "kas_utils/utils.hpp"
#include "kas_utils/time_measurer.h"
#include "kas_utils/collection.hpp"

#include <cmath>
#include <algorithm>

using kas_utils::fastErase;

namespace cartographer {
namespace mapping {

LoopsTrimmer::LoopsTrimmer(bool log_number_of_trimmed_loops) :
    log_number_of_trimmed_loops_(log_number_of_trimmed_loops) {}

void LoopsTrimmer::AddLoopsTrimmerOptions(int trajectory_id, const proto::LoopTrimmerOptions& options) {
  const auto& [_, emplaced] = loop_trimmer_options_.emplace(trajectory_id, options);
  CHECK(emplaced);
}

void LoopsTrimmer::AddNode(const NodeId& node, double accum_rotation, double travelled_distance) {
  const auto& [_, emplaced] = artds_.emplace(
      node, std::make_pair(accum_rotation, travelled_distance));
  CHECK(emplaced);
}

void LoopsTrimmer::AddSubmap(const SubmapId& submap, const NodeId& first_node_in_submap) {
  const auto& [_, emplaced] = submap_to_node_id_.emplace(submap, first_node_in_submap);
  CHECK(emplaced);
}

void LoopsTrimmer::AddLoop(const Constraint& new_loop) {
  NodeId node_1 = submap_to_node_id_.at(new_loop.submap_id);
  NodeId node_2 = new_loop.node_id;
  if (node_2 < node_1) {
    std::swap(node_1, node_2);
  }
  const auto& [_, emplaced] = loops_.emplace(
      std::make_tuple(new_loop.submap_id, new_loop.node_id, node_1, node_2),
      new_loop.score);
  CHECK(emplaced);
}

void LoopsTrimmer::AddLoops(const std::vector<Constraint>& new_loops) {
  for (const Constraint& new_loop : new_loops) {
    AddLoop(new_loop);
  }
}

void LoopsTrimmer::RemoveLoop(const SubmapId& submap, const NodeId& node) {
  NodeId node_1 = submap_to_node_id_.at(submap);
  NodeId node_2 = node;
  if (node_2 < node_1) {
    std::swap(node_1, node_2);
  }
  bool erased = loops_.erase(std::make_tuple(submap, node, node_1, node_2));
  CHECK(erased);
}

std::vector<Constraint> LoopsTrimmer::TrimFalseDetectedLoops(
    const std::vector<Constraint>& new_loops,
    const MapById<SubmapId, InternalSubmapData>& global_submap_poses_3d,
    const MapById<NodeId, TrajectoryNode>& trajectory_nodes) const {
  MEASURE_BLOCK_TIME(LoopsTrimmer__TrimFalseDetectedLoops);

  static kas_utils::Collection<std::tuple<double, double, double, double>> trim_loops_col("trim_loops", nullptr,
      [](std::ostream& out) {
        out << "max_rotation_error" << ' ' << "rotation_error" << ' ' <<
            "max_translation_error" << ' ' << "translation_error";
      },
      [](std::ostream& out, const std::tuple<double, double, double, double>& data) {
        const auto& [max_re, re, max_te, te] = data;
        out << std::fixed << std::setprecision(6) <<
            max_re << ' ' << re << ' ' << max_te << ' ' << te;
      });

  std::vector<Constraint> true_detected_loops;
  for (const Constraint& new_loop : new_loops) {
    std::map<int, proto::LoopTrimmerOptions>::const_iterator options_it;
    auto submap_options_it = loop_trimmer_options_.find(new_loop.submap_id.trajectory_id);
    if (submap_options_it != loop_trimmer_options_.end()) {
      options_it = submap_options_it;
    } else {
      auto node_options_it = loop_trimmer_options_.find(new_loop.node_id.trajectory_id);
      if (node_options_it != loop_trimmer_options_.end()) {
        options_it = node_options_it;
      } else {
        continue;
      }
    }
    const proto::LoopTrimmerOptions& options = options_it->second;

    if (!options.trim_false_detected_loops()) {
      continue;
    }

    NodeId node_1 = submap_to_node_id_.at(new_loop.submap_id);
    NodeId node_2 = new_loop.node_id;
    if (node_2 < node_1) {
      std::swap(node_1, node_2);
    }

    const auto& [accum_rotation, travelled_distance] = GetARTDWithLoops(
        node_1, node_2, new_loop.score);
    double max_rotation_error =
        options.rotation_error_rate() * accum_rotation +
        options.translation_to_rotation_error() * travelled_distance;
    double max_translation_error =
        options.translation_error_rate() * travelled_distance +
        options.rotation_to_translation_error_rate() * accum_rotation * travelled_distance;

    const auto& [rotation_error, translation_error] = GetLoopError(
        new_loop, global_submap_poses_3d, trajectory_nodes);

    if (rotation_error < max_rotation_error && translation_error < max_translation_error) {
      true_detected_loops.push_back(new_loop);
    }

    trim_loops_col.add(std::make_tuple(max_rotation_error, rotation_error, max_translation_error, translation_error));
  }

  if (log_number_of_trimmed_loops_) {
    int num_before = new_loops.size();
    int num_after = true_detected_loops.size();
    LOG(INFO) << "Trimmed false detected loops - " << num_before - num_after <<
        " (before: " << num_before << ", after: " << num_after << ")";
  }

  return true_detected_loops;
}

std::pair<double, double> LoopsTrimmer::GetARTDWithLoops(
    const NodeId& node_1, const NodeId& node_2, float min_score) const {
  CHECK(node_1 < node_2);

  const auto& [ar_node_1, td_node_1] = artds_.at(node_1);
  const auto& [ar_node_2, td_node_2] = artds_.at(node_2);

  double accum_rotation, travelled_distance;
  if (node_1.trajectory_id == node_2.trajectory_id) {
    accum_rotation = ar_node_2 - ar_node_1;
    travelled_distance = td_node_2 - td_node_1;
  } else {
    accum_rotation = std::numeric_limits<double>::max();
    travelled_distance = std::numeric_limits<double>::max();
  }

  for (const auto& [loop_ids, loop_score] : loops_) {
    if (loop_score < min_score) {
      continue;
    }
    const NodeId& loop_node_1 = std::get<2>(loop_ids);
    const NodeId& loop_node_2 = std::get<3>(loop_ids);
    // check that trajectories of nodes and loop are the same
    if (node_1.trajectory_id != loop_node_1.trajectory_id || node_2.trajectory_id != loop_node_2.trajectory_id) {
      continue;
    }
    // check that if trajectory is one than nodes and loop overlap
    if ((node_1.trajectory_id == node_2.trajectory_id) &&
        (node_1.node_index >= loop_node_2.node_index || node_2.node_index <= loop_node_1.node_index)) {
      continue;
    }

    const auto& [ar_loop_node_1, td_loop_node_1] = artds_.at(loop_node_1);
    const auto& [ar_loop_node_2, td_loop_node_2] = artds_.at(loop_node_2);

    double accum_rotation_with_loop =
        std::abs(ar_node_1 - ar_loop_node_1) + std::abs(ar_node_2 - ar_loop_node_2);
    double travelled_distance_with_loop =
        std::abs(td_node_1 - td_loop_node_1) + std::abs(td_node_2 - td_loop_node_2);

    accum_rotation = std::min(accum_rotation, accum_rotation_with_loop);
    travelled_distance = std::min(travelled_distance, travelled_distance_with_loop);
  }

  return std::make_pair(accum_rotation, travelled_distance);
}

std::pair<double, double> LoopsTrimmer::GetLoopError(
    const Constraint& loop,
    const MapById<SubmapId, InternalSubmapData>& global_submap_poses_3d,
    const MapById<NodeId, TrajectoryNode>& trajectory_nodes) const {
  const transform::Rigid3d& global_submap_pose =
      global_submap_poses_3d.at(loop.submap_id).global_pose;
  const transform::Rigid3d& global_node_pose =
      trajectory_nodes.at(loop.node_id).global_pose;
  transform::Rigid3d relative_pose = global_submap_pose.inverse() * global_node_pose;
  transform::Rigid3d error = loop.pose.zbar_ij.inverse() * relative_pose;

  double rotation_error = 2 * std::acos(error.rotation().w());
  if (rotation_error > M_PI) {
    rotation_error -= 2 * M_PI;
  }
  rotation_error = std::abs(rotation_error);
  double translation_error = error.translation().norm();

  return std::make_pair(rotation_error, translation_error);
}

void LoopsTrimmer::TrimCloseLoops(std::vector<Constraint>& constraints) {
  MEASURE_BLOCK_TIME(LoopsTrimmer__TrimCloseLoops);
  int num_before = loops_.size();

  SubmapId current_submap(-1, -1);
  int current_nodes_trajectory = -1;
  std::vector<std::pair<float, NodeId>> submap_nodes;
  std::map<int, proto::LoopTrimmerOptions>::const_iterator options_it =
      loop_trimmer_options_.end();
  std::set<std::pair<SubmapId, NodeId>> loops_to_remove;
  for (const auto& [loop_ids, loop_score] : loops_) {
    const SubmapId& submap = std::get<0>(loop_ids);
    const NodeId& node = std::get<1>(loop_ids);
    if (submap != current_submap || node.trajectory_id != current_nodes_trajectory) {
      if (options_it != loop_trimmer_options_.end() && options_it->second.trim_close_loops() &&
          submap_nodes.size()) {
        const proto::LoopTrimmerOptions& options = options_it->second;
        std::sort(submap_nodes.rbegin(), submap_nodes.rend());
        for (auto weak_it = std::next(submap_nodes.begin()); weak_it != submap_nodes.end(); ++weak_it) {
          for (auto strong_it = submap_nodes.begin(); strong_it != weak_it; ++strong_it) {
            int distance = std::abs(strong_it->second.node_index - weak_it->second.node_index);
            if (distance < options.min_distance_in_nodes()) {
              loops_to_remove.emplace(current_submap, weak_it->second);
              auto next_weak_it = submap_nodes.erase(weak_it);
              weak_it = std::prev(next_weak_it);
              break;
            }
          }
        }
      }
      current_submap = submap;
      current_nodes_trajectory = node.trajectory_id;
      submap_nodes.clear();
      options_it = loop_trimmer_options_.find(submap.trajectory_id);
    }
    if (options_it == loop_trimmer_options_.end() || !options_it->second.trim_close_loops()) {
      continue;
    }
    submap_nodes.emplace_back(loop_score, node);
  }

  for (const auto& [submap, node] : loops_to_remove) {
    RemoveLoop(submap, node);
  }

  int num_erased = 0;
  int check_num_before = 0;
  for (auto constraint_it = constraints.begin(); constraint_it != constraints.end();) {
    const Constraint& constraint = *constraint_it;
    if (constraint.tag == Constraint::INTRA_SUBMAP) {
      ++constraint_it;
      continue;
    }
    check_num_before++;

    const SubmapId& submap = constraint.submap_id;
    const NodeId& node = constraint.node_id;
    if (loops_to_remove.count(std::make_pair(submap, node))) {
      num_erased++;
      constraint_it = fastErase(constraints, constraint_it);
      continue;
    }
    ++constraint_it;
  }
  CHECK(num_erased == loops_to_remove.size());
  CHECK(num_before == check_num_before);

  if (log_number_of_trimmed_loops_) {
    int num_after = loops_.size();
    LOG(INFO) << "Trimmed close loops - " << num_before - num_after <<
        " (before: " << num_before << ", after: " << num_after << ")";
  }
}

void LoopsTrimmer::TrimFalseLoops(
    std::vector<Constraint>& constraints,
    double max_rotation_error, double max_translation_error,
    const MapById<SubmapId, InternalSubmapData>& global_submap_poses_3d,
    const MapById<NodeId, TrajectoryNode>& trajectory_nodes) {
  for (auto constraint_it = constraints.begin(); constraint_it != constraints.end();) {
    const Constraint& constraint = *constraint_it;
    if (constraint.tag == Constraint::INTRA_SUBMAP) {
      ++constraint_it;
      continue;
    }

    const auto& [rotation_error, translation_error] = GetLoopError(
        constraint, global_submap_poses_3d, trajectory_nodes);
    if (rotation_error >= max_rotation_error || translation_error >= max_translation_error) {
      RemoveLoop(constraint.submap_id, constraint.node_id);
      constraint_it = fastErase(constraints, constraint_it);
      continue;
    }
    ++constraint_it;
  }
}

}
}
