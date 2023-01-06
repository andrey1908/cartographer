#pragma once

#include <vector>
#include <map>

#include "cartographer/mapping/id.h"
#include "cartographer/mapping/constraint.h"

namespace cartographer {
namespace mapping {

struct TrimmedLoop {
  SubmapId submap_id;
  NodeId node_id;
  float score;
};

class PoseGraphConstraints {
public:
  template <typename T>
  void InsertConstraint(T&& constraint) {
    if (constraint.tag == Constraint::INTER_SUBMAP) {
      UpdateLastLoopIndices(constraint);
    }
    constraints_.emplace_back(std::forward<T>(constraint));
  }

  template <typename T>
  void InsertConstraints(const T& begin, const T& end) {
    for (T it = begin; it != end; ++it) {
      const Constraint& loop = *it;
      if (loop.tag != Constraint::INTER_SUBMAP) {
        continue;
      }
      UpdateLastLoopIndices(loop);
    }
    constraints_.insert(constraints_.end(), begin, end);
  }

  template <typename T>
  void SetConstraints(T&& constraints) {
    last_loop_submap_index_for_trajectory_.clear();
    last_loop_node_index_for_trajectory_.clear();
    constraints_ = std::forward<T>(constraints);
    for (const Constraint& loop : constraints_) {
      if (loop.tag != Constraint::INTER_SUBMAP) {
        continue;
      }
      UpdateLastLoopIndices(loop);
    }
  }

  template <typename T>
  void InsertLoopFromTrimmedSubmap(T&& trimmed_loop) {
    loops_from_trimmed_submaps_.emplace_back(std::forward<T>(trimmed_loop));
  }

  void SetTravelledDistance(
      const NodeId& node_id, double travelled_distance);
  void SetFirstNodeIdForSubmap(
      const NodeId& first_node_id, const SubmapId& submap_id);

  const std::vector<Constraint>& constraints() const {
    return constraints_;
  }

  double GetTravelledDistanceWithLoops(
      const NodeId& node_1, const NodeId& node_2, float min_score);
  bool IsLoopLast(const Constraint& loop);

  std::vector<Constraint>::iterator begin() {
    return constraints_.begin();
  }
  std::vector<Constraint>::iterator end() {
    return constraints_.end();
  }
  std::vector<Constraint>::const_iterator begin() const {
    return constraints_.cbegin();
  }
  std::vector<Constraint>::const_iterator end() const {
    return constraints_.cend();
  }
  std::vector<Constraint>::const_iterator cbegin() const {
    return constraints_.cbegin();
  }
  std::vector<Constraint>::const_iterator cend() const {
    return constraints_.cend();
  }
  size_t size() const {
    return constraints_.size();
  }

private:
  void UpdateLastLoopIndices(const Constraint& loop);
  double GetTravelledDistanceWithLoopsSameTrajectory(
      NodeId node_1, NodeId node_2, float min_score);
  double GetTravelledDistanceWithLoopsDifferentTrajectories(
      NodeId node_1, NodeId node_2, float min_score);

private:
  std::vector<Constraint> constraints_;
  std::vector<TrimmedLoop> loops_from_trimmed_submaps_;
  std::map<NodeId, double> travelled_distance_;
  std::map<SubmapId, NodeId> first_node_id_for_submap_;
  std::map<int, int> last_loop_submap_index_for_trajectory_;
  std::map<int, int> last_loop_node_index_for_trajectory_;
};

}
}
