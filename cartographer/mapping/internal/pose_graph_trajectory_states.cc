#include "cartographer/mapping/internal/pose_graph_trajectory_states.h"

namespace cartographer {
namespace mapping {

void PoseGraphTrajectoryStates::AddTrajectory(int trajectory_id) {
  CHECK(!ContainsTrajectory(trajectory_id));
  trajectory_states_[trajectory_id];
  trajectory_connectivity_state_.Add(trajectory_id);
}

bool PoseGraphTrajectoryStates::ContainsTrajectory(int trajectory_id) const {
  return trajectory_states_.count(trajectory_id);
}

bool PoseGraphTrajectoryStates::CanModifyTrajectory(int trajectory_id) const {
  CHECK(ContainsTrajectory(trajectory_id));
  return
      IsTrajectoryActive(trajectory_id) &&
      IsTrajectoryDeletionStateNormal(trajectory_id);
}

bool PoseGraphTrajectoryStates::IsTrajectoryActive(int trajectory_id) const {
  CHECK(ContainsTrajectory(trajectory_id));
  const InternalTrajectoryState& trajectory_state = trajectory_states_.at(trajectory_id);
  return trajectory_state.state == TrajectoryState::ACTIVE;
}

bool PoseGraphTrajectoryStates::IsTrajectoryFinished(int trajectory_id) const {
  CHECK(ContainsTrajectory(trajectory_id));
  const InternalTrajectoryState& trajectory_state = trajectory_states_.at(trajectory_id);
  return trajectory_state.state == TrajectoryState::FINISHED;
}

bool PoseGraphTrajectoryStates::IsTrajectoryFrozen(int trajectory_id) const {
  CHECK(ContainsTrajectory(trajectory_id));
  const InternalTrajectoryState& trajectory_state = trajectory_states_.at(trajectory_id);
  return trajectory_state.state == TrajectoryState::FROZEN;
}

bool PoseGraphTrajectoryStates::IsTrajectoryDeleted(int trajectory_id) const {
  CHECK(ContainsTrajectory(trajectory_id));
  const InternalTrajectoryState& trajectory_state = trajectory_states_.at(trajectory_id);
  return trajectory_state.state == TrajectoryState::DELETED;
}

bool PoseGraphTrajectoryStates::IsTrajectoryDeletionStateNormal(int trajectory_id) const {
  CHECK(ContainsTrajectory(trajectory_id));
  const InternalTrajectoryState& trajectory_state = trajectory_states_.at(trajectory_id);
  return trajectory_state.deletion_state ==
      InternalTrajectoryState::DeletionState::NORMAL;
}

bool PoseGraphTrajectoryStates::IsTrajectoryScheduledForDeletion(int trajectory_id) const {
  CHECK(ContainsTrajectory(trajectory_id));
  const InternalTrajectoryState& trajectory_state = trajectory_states_.at(trajectory_id);
  return trajectory_state.deletion_state ==
      InternalTrajectoryState::DeletionState::SCHEDULED_FOR_DELETION;
}

bool PoseGraphTrajectoryStates::IsTrajectoryWaitingForDeletion(int trajectory_id) const {
  CHECK(ContainsTrajectory(trajectory_id));
  const InternalTrajectoryState& trajectory_state = trajectory_states_.at(trajectory_id);
  return trajectory_state.deletion_state ==
      InternalTrajectoryState::DeletionState::WAIT_FOR_DELETION;
}

void PoseGraphTrajectoryStates::MarkTrajectoryAsFinished(int trajectory_id) {
  CHECK(CanModifyTrajectory(trajectory_id));
  InternalTrajectoryState& trajectory_state = trajectory_states_.at(trajectory_id);
  trajectory_state.state = TrajectoryState::FINISHED;
}

void PoseGraphTrajectoryStates::MarkTrajectoryAsFrozen(int trajectory_id) {
  CHECK(CanModifyTrajectory(trajectory_id));
  InternalTrajectoryState& trajectory_state = trajectory_states_.at(trajectory_id);
  trajectory_state.state = TrajectoryState::FROZEN;
}

void PoseGraphTrajectoryStates::ScheduleTrajectoryForDeletion(int trajectory_id) {
  CHECK(!IsTrajectoryDeleted(trajectory_id));
  CHECK(IsTrajectoryDeletionStateNormal(trajectory_id));
  InternalTrajectoryState& trajectory_state = trajectory_states_.at(trajectory_id);
  trajectory_state.deletion_state =
      InternalTrajectoryState::DeletionState::SCHEDULED_FOR_DELETION;
}

void PoseGraphTrajectoryStates::PrepareTrajectoryForDeletion(int trajectory_id) {
  CHECK(IsTrajectoryScheduledForDeletion(trajectory_id));
  InternalTrajectoryState& trajectory_state = trajectory_states_.at(trajectory_id);
  trajectory_state.deletion_state =
      InternalTrajectoryState::DeletionState::WAIT_FOR_DELETION;
}

void PoseGraphTrajectoryStates::MarkTrajectoryAsDeleted(int trajectory_id) {
  CHECK(IsTrajectoryWaitingForDeletion(trajectory_id));
  InternalTrajectoryState& trajectory_state = trajectory_states_.at(trajectory_id);
  trajectory_state.state = TrajectoryState::DELETED;
  trajectory_state.deletion_state = InternalTrajectoryState::DeletionState::NORMAL;
}

common::Time PoseGraphTrajectoryStates::LastConnectionTime(
    int trajectory_a, int trajectory_b) const {
  return trajectory_connectivity_state_.LastConnectionTime(trajectory_a, trajectory_b);
}

bool PoseGraphTrajectoryStates::TransitivelyConnected(
    int trajectory_a, int trajectory_b) const {
  return trajectory_connectivity_state_.TransitivelyConnected(trajectory_a, trajectory_b);
}

void PoseGraphTrajectoryStates::Connect(
    int trajectory_a, int trajectory_b, common::Time time) {
  return trajectory_connectivity_state_.Connect(trajectory_a, trajectory_b, time);
}

std::vector<std::vector<int>> PoseGraphTrajectoryStates::Components() const {
  return trajectory_connectivity_state_.Components();
}

}
}
