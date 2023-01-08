#pragma once

#include "cartographer/mapping/trajectory_state.h"
#include "cartographer/mapping/internal/trajectory_connectivity_state.h"

namespace cartographer {
namespace mapping {

struct InternalTrajectoryState {
  enum class DeletionState {
    NORMAL,
    SCHEDULED_FOR_DELETION,
    WAIT_FOR_DELETION
  };

  TrajectoryState state = TrajectoryState::ACTIVE;
  DeletionState deletion_state = DeletionState::NORMAL;
};

class PoseGraphTrajectoryStates {
public:
  void AddTrajectory(int trajectory_id);

  bool ContainsTrajectory(int trajectory_id) const;
  bool CanModifyTrajectory(int trajectory_id) const;

  bool IsTrajectoryActive(int trajectory_id) const;
  bool IsTrajectoryFinished(int trajectory_id) const;
  bool IsTrajectoryFrozen(int trajectory_id) const;
  bool IsTrajectoryDeleted(int trajectory_id) const;
  bool IsTrajectoryDeletionStateNormal(int trajectory_id) const;
  bool IsTrajectoryScheduledForDeletion(int trajectory_id) const;
  bool IsTrajectoryWaitingForDeletion(int trajectory_id) const;

  void MarkTrajectoryAsFinished(int trajectory_id);
  void MarkTrajectoryAsFrozen(int trajectory_id);

  void ScheduleTrajectoryForDeletion(int trajectory_id);
  void PrepareTrajectoryForDeletion(int trajectory_id);
  void MarkTrajectoryAsDeleted(int trajectory_id);

  common::Time LastConnectionTime(int trajectory_a, int trajectory_b) const;
  bool TransitivelyConnected(int trajectory_a, int trajectory_b) const;
  void Connect(int trajectory_a, int trajectory_b, common::Time time);
  std::vector<std::vector<int>> Components() const;

  const TrajectoryState& state(int trajectory_id) const {
    CHECK(ContainsTrajectory(trajectory_id));
    return trajectory_states_.at(trajectory_id).state;
  }
  const InternalTrajectoryState::DeletionState& deletion_state(int trajectory_id) const {
    CHECK(ContainsTrajectory(trajectory_id));
    return trajectory_states_.at(trajectory_id).deletion_state;
  }
  const std::map<int, InternalTrajectoryState>& trajectory_states() const {
    return trajectory_states_;
  }

  std::map<int, InternalTrajectoryState>::const_iterator begin() const {
    return trajectory_states_.cbegin();
  }
  std::map<int, InternalTrajectoryState>::const_iterator end() const {
    return trajectory_states_.cend();
  }
  std::map<int, InternalTrajectoryState>::const_iterator cbegin() const {
    return trajectory_states_.cbegin();
  }
  std::map<int, InternalTrajectoryState>::const_iterator cend() const {
    return trajectory_states_.cend();
  }

private:
  std::map<int, InternalTrajectoryState> trajectory_states_;
  TrajectoryConnectivityState trajectory_connectivity_state_;
};

}
}
