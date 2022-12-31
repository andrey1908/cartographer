/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer/mapping/internal/3d/pose_graph_3d.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

#include "Eigen/Eigenvalues"
#include "absl/memory/memory.h"
#include "cartographer/common/math.h"
#include "cartographer/mapping/proto/pose_graph/constraint_builder_options.pb.h"
#include "cartographer/sensor/compressed_point_cloud.h"
#include "cartographer/sensor/internal/voxel_filter.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

static auto* kWorkQueueDelayMetric = metrics::Gauge::Null();
static auto* kWorkQueueSizeMetric = metrics::Gauge::Null();
static auto* kConstraintsSameTrajectoryMetric = metrics::Gauge::Null();
static auto* kConstraintsDifferentTrajectoryMetric = metrics::Gauge::Null();
static auto* kActiveSubmapsMetric = metrics::Gauge::Null();
static auto* kFrozenSubmapsMetric = metrics::Gauge::Null();
static auto* kDeletedSubmapsMetric = metrics::Gauge::Null();

PoseGraph3D::PoseGraph3D(
    const proto::PoseGraphOptions& options,
    std::unique_ptr<optimization::OptimizationProblem3D> optimization_problem,
    common::ThreadPool* thread_pool)
    : options_(options),
      optimization_problem_(std::move(optimization_problem)),
      constraint_builder_(options_.constraint_builder_options(), thread_pool),
      thread_pool_(thread_pool),
      num_nodes_since_last_loop_closure_(0),
      num_local_constraints_to_compute_(0.0),
      num_global_constraints_to_compute_(0.0) {}

PoseGraph3D::~PoseGraph3D() {
  WaitForAllComputations();
  absl::MutexLock locker(&work_queue_mutex_);
  CHECK(work_queue_ == nullptr);
}

void PoseGraph3D::AddWorkItem(
    const std::function<WorkItem::Result()>& work_item) {
  absl::MutexLock locker(&work_queue_mutex_);
  if (work_queue_ == nullptr) {
    work_queue_ = absl::make_unique<WorkQueue>();
    auto task = absl::make_unique<common::Task>();
    task->SetWorkItem([this]() { DrainWorkQueue(); });
    thread_pool_->Schedule(std::move(task));
  }
  const auto now = std::chrono::steady_clock::now();
  work_queue_->push_back({now, work_item});
  kWorkQueueSizeMetric->Set(work_queue_->size());
  kWorkQueueDelayMetric->Set(
      std::chrono::duration_cast<std::chrono::duration<double>>(
          now - work_queue_->front().time).count());
}

void PoseGraph3D::AddTrajectoryIfNeeded(int trajectory_id) {
  data_.trajectories_state[trajectory_id];
  CHECK(data_.trajectories_state.at(trajectory_id).state !=
        TrajectoryState::FINISHED);
  CHECK(data_.trajectories_state.at(trajectory_id).state !=
        TrajectoryState::DELETED);
  CHECK(data_.trajectories_state.at(trajectory_id).deletion_state ==
        InternalTrajectoryState::DeletionState::NORMAL);
  data_.trajectory_connectivity_state.Add(trajectory_id);
}

void PoseGraph3D::DeleteTrajectoriesIfNeeded() {
  TrimmingHandle trimming_handle(this);
  for (auto& it : data_.trajectories_state) {
    if (it.second.deletion_state ==
        InternalTrajectoryState::DeletionState::WAIT_FOR_DELETION) {
      // TODO(gaschler): Consider directly deleting from data_, which may be
      // more complete.
      auto submap_ids = trimming_handle.GetSubmapIds(it.first);
      for (auto& submap_id : submap_ids) {
        trimming_handle.TrimSubmap(submap_id);
      }
      it.second.state = TrajectoryState::DELETED;
      it.second.deletion_state = InternalTrajectoryState::DeletionState::NORMAL;
    }
  }
}

void PoseGraph3D::AddImuData(
    int trajectory_id,
    const sensor::ImuData& imu_data) {
  AddWorkItem([=]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      optimization_problem_->AddImuData(trajectory_id, imu_data);
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddOdometryData(
    int trajectory_id,
    const sensor::OdometryData& odometry_data) {
  AddWorkItem([=]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      optimization_problem_->AddOdometryData(trajectory_id, odometry_data);
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddFixedFramePoseData(
    int trajectory_id,
    const sensor::FixedFramePoseData& fixed_frame_pose_data) {
  AddWorkItem([=]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      optimization_problem_->AddFixedFramePoseData(trajectory_id,
                                                   fixed_frame_pose_data);
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddLandmarkData(
    int trajectory_id,
    const sensor::LandmarkData& landmark_data) {
  AddWorkItem([=]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      for (const auto& observation : landmark_data.landmark_observations) {
        data_.landmark_nodes[observation.id].landmark_observations.emplace_back(
            PoseGraphInterface::LandmarkNode::LandmarkObservation{
                trajectory_id, landmark_data.time,
                observation.landmark_to_tracking_transform,
                observation.translation_weight, observation.rotation_weight});
      }
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

std::pair<bool, bool> PoseGraph3D::CheckIfConstraintCanBeAdded(
    const NodeId& node_id, const SubmapId& submap_id) {
  bool local_constraint_can_be_added = false;
  bool global_constraint_can_be_added = false;

  common::Time latest_node_time;
  common::Time last_connection_time;
  bool connected;
  {
    absl::MutexLock locker(&mutex_);
    CHECK(data_.submap_data.at(submap_id).state == SubmapState::kFinished);
    if (!data_.submap_data.at(submap_id).submap->insertion_finished()) {
      return std::make_pair(false, false);
    }
    latest_node_time = GetLatestNodeTime(node_id, submap_id);
    last_connection_time = data_.trajectory_connectivity_state.LastConnectionTime(
        node_id.trajectory_id, submap_id.trajectory_id);
    connected = data_.trajectory_connectivity_state.TransitivelyConnected(
        node_id.trajectory_id, submap_id.trajectory_id);
  }
  common::Duration global_constraint_search_after_n_seconds =
      common::FromSeconds(options_.global_constraint_search_after_n_seconds());
  bool recently_connected =
      (last_connection_time + global_constraint_search_after_n_seconds > latest_node_time);
  if (node_id.trajectory_id == submap_id.trajectory_id ||
      (connected &&
        (global_constraint_search_after_n_seconds == common::Duration() || recently_connected))) {
    transform::Rigid3d global_node_pose =
        optimization_problem_->node_data().at(node_id).global_pose;
    transform::Rigid3d global_submap_pose =
        optimization_problem_->submap_data().at(submap_id).global_pose;
    double distance =
        (global_node_pose.translation() - global_submap_pose.translation()).norm();
    if (distance <= options_.max_local_constraint_distance()) {
      local_constraint_can_be_added = true;
    }
  } else {
    global_constraint_can_be_added = true;
  }
  return std::make_pair(local_constraint_can_be_added, global_constraint_can_be_added);
}

std::pair<std::vector<SubmapId>, std::vector<SubmapId>>
PoseGraph3D::ComputeCandidatesForConstraints(const NodeId& node_id) {
  bool pure_localization_trajectory =
      pure_localization_trajectory_ids_.count(node_id.trajectory_id);
  std::vector<SubmapId> local_candidates;
  std::vector<SubmapId> global_candidates;
  for (const auto& submap_id_data : optimization_problem_->submap_data()) {
    const SubmapId& submap_id = submap_id_data.id;
    if (pure_localization_trajectory && node_id.trajectory_id == submap_id.trajectory_id) {
      continue;
    }
    {
      absl::MutexLock locker(&mutex_);
      if (data_.submap_data.at(submap_id).node_ids.count(node_id)) {
        continue;
      }
    }
    bool local_constraint_can_be_added, global_constraint_can_be_added;
    std::tie(local_constraint_can_be_added, global_constraint_can_be_added) =
        CheckIfConstraintCanBeAdded(node_id, submap_id);
    CHECK(!(local_constraint_can_be_added && global_constraint_can_be_added));
    if (local_constraint_can_be_added) {
      local_candidates.emplace_back(submap_id);
    }
    if (global_constraint_can_be_added) {
      global_candidates.emplace_back(submap_id);
    }
  }
  return std::make_pair(std::move(local_candidates), std::move(global_candidates));
}

std::pair<std::vector<NodeId>, std::vector<NodeId>>
PoseGraph3D::ComputeCandidatesForConstraints(const SubmapId& submap_id) {
  std::set<NodeId> submap_node_ids;
  {
    absl::MutexLock locker(&mutex_);
    submap_node_ids = data_.submap_data.at(submap_id).node_ids;
  }
  bool pure_localization_trajectory =
      pure_localization_trajectory_ids_.count(submap_id.trajectory_id);
  CHECK(!pure_localization_trajectory);
  std::vector<NodeId> local_candidates;
  std::vector<NodeId> global_candidates;
  for (const auto& node_id_data : optimization_problem_->node_data()) {
    const NodeId& node_id = node_id_data.id;
    if (submap_node_ids.count(node_id)) {
      continue;
    }
    bool local_constraint_can_be_added, global_constraint_can_be_added;
    std::tie(local_constraint_can_be_added, global_constraint_can_be_added) =
        CheckIfConstraintCanBeAdded(node_id, submap_id);
    CHECK(!(local_constraint_can_be_added && global_constraint_can_be_added));
    if (local_constraint_can_be_added) {
      local_candidates.emplace_back(node_id);
    }
    if (global_constraint_can_be_added) {
      global_candidates.emplace_back(node_id);
    }
  }
  return std::make_pair(std::move(local_candidates), std::move(global_candidates));
}

std::pair<std::vector<SubmapId>, std::vector<SubmapId>>
PoseGraph3D::SelectCandidatesForConstraints(
    const std::vector<SubmapId>& local_candidates,
    const std::vector<SubmapId>& global_candidates) {
  std::vector<SubmapId> submaps_for_local_constraints;
  auto local_candidate_it = local_candidates.begin();
  bool local_first_pass = true;
  while (local_candidate_it != local_candidates.end() &&
      submaps_used_for_local_constraints_.count(*local_candidate_it)) {
    ++local_candidate_it;
  }
  std::set<SubmapId> used_local_candidates;
  while (num_local_constraints_to_compute_ >= 1.0) {
    if (local_first_pass) {
      while (local_candidate_it != local_candidates.end() &&
          submaps_used_for_local_constraints_.count(*local_candidate_it)) {
        ++local_candidate_it;
      }
    } else {
      while (local_candidate_it != local_candidates.end() &&
          used_local_candidates.count(*local_candidate_it)) {
        ++local_candidate_it;
      }
    }
    if (local_candidate_it == local_candidates.end()) {
      if (local_first_pass) {
        submaps_used_for_local_constraints_.clear();
        local_candidate_it = local_candidates.begin();
        local_first_pass = false;
        continue;
      } else {
        CHECK(used_local_candidates.size() == local_candidates.size());
        num_local_constraints_to_compute_ = 1.0;
        break;
      }
    }
    const SubmapId& submap_id = *local_candidate_it;
    submaps_for_local_constraints.emplace_back(submap_id);
    submaps_used_for_local_constraints_.emplace(submap_id);
    used_local_candidates.emplace(submap_id);
    num_local_constraints_to_compute_ -= 1.0;
    ++local_candidate_it;
  }

  std::vector<SubmapId> submaps_for_global_constraints;
  auto global_candidate_it = global_candidates.begin();
  bool global_first_pass = true;
  while (global_candidate_it != global_candidates.end() &&
      submaps_used_for_global_constraints_.count(*global_candidate_it)) {
    ++global_candidate_it;
  }
  std::set<SubmapId> used_global_candidates;
  while (num_global_constraints_to_compute_ >= 1.0) {
    if (global_first_pass) {
      while (global_candidate_it != global_candidates.end() &&
          submaps_used_for_global_constraints_.count(*global_candidate_it)) {
        ++global_candidate_it;
      }
    } else {
      while (global_candidate_it != global_candidates.end() &&
          used_global_candidates.count(*global_candidate_it)) {
        ++global_candidate_it;
      }
    }
    if (global_candidate_it == global_candidates.end()) {
      if (global_first_pass) {
        submaps_used_for_global_constraints_.clear();
        global_candidate_it = global_candidates.begin();
        global_first_pass = false;
        continue;
      } else {
        CHECK(used_global_candidates.size() == global_candidates.size());
        num_global_constraints_to_compute_ = 1.0;
        break;
      }
    }
    const SubmapId& submap_id = *global_candidate_it;
    submaps_for_global_constraints.emplace_back(submap_id);
    submaps_used_for_global_constraints_.emplace(submap_id);
    used_global_candidates.emplace(submap_id);
    num_global_constraints_to_compute_ -= 1.0;
    ++global_candidate_it;
  }

  return std::make_pair(std::move(submaps_for_local_constraints),
      std::move(submaps_for_global_constraints));
}

std::pair<std::vector<NodeId>, std::vector<NodeId>>
PoseGraph3D::SelectCandidatesForConstraints(
    const std::vector<NodeId>& local_candidates,
    const std::vector<NodeId>& global_candidates) {
  std::vector<NodeId> nodes_for_local_constraints;
  int num_local_constraints_to_compute = std::floor(num_local_constraints_to_compute_);
  if (num_local_constraints_to_compute > local_candidates.size()) {
    num_local_constraints_to_compute = local_candidates.size();
  }
  for (int n = 0; n < num_local_constraints_to_compute; n++) {
    int i = std::lround(1.0 * n / num_local_constraints_to_compute * local_candidates.size());
    nodes_for_local_constraints.emplace_back(local_candidates[i]);
  }
  num_local_constraints_to_compute_ -= std::floor(num_local_constraints_to_compute_);

  std::vector<NodeId> nodes_for_global_constraints;
  int num_global_constraints_to_compute = std::floor(num_global_constraints_to_compute_);
  if (num_global_constraints_to_compute > global_candidates.size()) {
    num_global_constraints_to_compute = global_candidates.size();
  }
  for (int n = 0; n < num_global_constraints_to_compute; n++) {
    int i = std::lround(1.0 * n / num_global_constraints_to_compute * global_candidates.size());
    nodes_for_global_constraints.emplace_back(global_candidates[i]);
  }
  num_global_constraints_to_compute_ -= std::floor(num_global_constraints_to_compute_);

  return std::make_pair(std::move(nodes_for_local_constraints),
      std::move(nodes_for_global_constraints));
}

void PoseGraph3D::MaybeAddConstraints(const NodeId& node_id,
    const std::vector<SubmapId>& local_submap_ids,
    const std::vector<SubmapId>& global_submap_ids) {
  transform::Rigid3d global_node_pose =
      optimization_problem_->node_data().at(node_id).global_pose;
  const TrajectoryNode::Data* constant_data;
  {
    absl::MutexLock locker(&mutex_);
    constant_data = data_.trajectory_nodes.at(node_id).constant_data.get();
  }

  for (const SubmapId& submap_id : local_submap_ids) {
    transform::Rigid3d global_submap_pose =
        optimization_problem_->submap_data().at(submap_id).global_pose;
    const Submap3D* submap;
    {
      absl::MutexLock locker(&mutex_);
      submap = static_cast<const Submap3D*>(data_.submap_data.at(submap_id).submap.get());
    }
    constraint_builder_.MaybeAddConstraint(
        submap_id, submap, node_id, constant_data,
        global_node_pose, global_submap_pose);
  }

  for (const SubmapId& submap_id : global_submap_ids) {
    transform::Rigid3d global_submap_pose =
        optimization_problem_->submap_data().at(submap_id).global_pose;
    const Submap3D* submap;
    {
      absl::MutexLock locker(&mutex_);
      submap = static_cast<const Submap3D*>(data_.submap_data.at(submap_id).submap.get());
    }
    constraint_builder_.MaybeAddGlobalConstraint(
        submap_id, submap, node_id, constant_data,
        global_node_pose.rotation(), global_submap_pose.rotation());
  }
}

void PoseGraph3D::MaybeAddConstraints(const SubmapId& submap_id,
      const std::vector<NodeId>& local_node_ids,
      const std::vector<NodeId>& global_node_ids) {
  transform::Rigid3d global_submap_pose =
      optimization_problem_->submap_data().at(submap_id).global_pose;
  const Submap3D* submap;
  {
    absl::MutexLock locker(&mutex_);
    submap = static_cast<const Submap3D*>(data_.submap_data.at(submap_id).submap.get());
  }

  for (const NodeId& node_id : local_node_ids) {
    transform::Rigid3d global_node_pose =
        optimization_problem_->node_data().at(node_id).global_pose;
    const TrajectoryNode::Data* constant_data;
    {
      absl::MutexLock locker(&mutex_);
      constant_data = data_.trajectory_nodes.at(node_id).constant_data.get();
    }
    constraint_builder_.MaybeAddConstraint(
        submap_id, submap, node_id, constant_data,
        global_node_pose, global_submap_pose);
  }

  for (const NodeId& node_id : global_node_ids) {
    transform::Rigid3d global_node_pose =
        optimization_problem_->node_data().at(node_id).global_pose;
    const TrajectoryNode::Data* constant_data;
    {
      absl::MutexLock locker(&mutex_);
      constant_data = data_.trajectory_nodes.at(node_id).constant_data.get();
    }
    constraint_builder_.MaybeAddGlobalConstraint(
        submap_id, submap, node_id, constant_data,
        global_node_pose.rotation(), global_submap_pose.rotation());
  }
}

std::vector<SubmapId> PoseGraph3D::InitializeGlobalSubmapPoses(
    int trajectory_id, const common::Time time,
    const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps) {
  CHECK(!insertion_submaps.empty());
  const auto& submap_data = optimization_problem_->submap_data();
  if (insertion_submaps.size() == 1) {
    // If we don't already have an entry for the first submap, add one.
    if (submap_data.SizeOfTrajectoryOrZero(trajectory_id) == 0) {
      if (data_.initial_trajectory_poses.count(trajectory_id) > 0) {
        data_.trajectory_connectivity_state.Connect(
            trajectory_id,
            data_.initial_trajectory_poses.at(trajectory_id).to_trajectory_id,
            time);
      }
      optimization_problem_->AddSubmap(
          trajectory_id, ComputeLocalToGlobalTransform(
                             data_.global_submap_poses_3d, trajectory_id) *
                             insertion_submaps[0]->local_pose());
    }
    CHECK_EQ(1, submap_data.SizeOfTrajectoryOrZero(trajectory_id));
    const SubmapId submap_id{trajectory_id, 0};
    CHECK(data_.submap_data.at(submap_id).submap == insertion_submaps.front());
    return {submap_id};
  }
  CHECK_EQ(2, insertion_submaps.size());
  const auto end_it = submap_data.EndOfTrajectory(trajectory_id);
  CHECK(submap_data.BeginOfTrajectory(trajectory_id) != end_it);
  const SubmapId last_submap_id = std::prev(end_it)->id;
  if (data_.submap_data.at(last_submap_id).submap ==
      insertion_submaps.front()) {
    // In this case, 'last_submap_id' is the ID of 'insertions_submaps.front()'
    // and 'insertions_submaps.back()' is new.
    const auto& first_submap_pose = submap_data.at(last_submap_id).global_pose;
    optimization_problem_->AddSubmap(
        trajectory_id, first_submap_pose *
                           insertion_submaps[0]->local_pose().inverse() *
                           insertion_submaps[1]->local_pose());
    return {last_submap_id,
            SubmapId{trajectory_id, last_submap_id.submap_index + 1}};
  }
  CHECK(data_.submap_data.at(last_submap_id).submap ==
        insertion_submaps.back());
  const SubmapId front_submap_id{trajectory_id,
                                 last_submap_id.submap_index - 1};
  CHECK(data_.submap_data.at(front_submap_id).submap ==
        insertion_submaps.front());
  return {front_submap_id, last_submap_id};
}

WorkItem::Result PoseGraph3D::ComputeConstraintsForNode(
    const NodeId& node_id,
    std::vector<std::shared_ptr<const Submap3D>> insertion_submaps,
    const bool newly_finished_submap) {
  absl::MutexLock queue_locker(&executing_work_item_mutex_);
  std::vector<SubmapId> submap_ids;
  {
    absl::MutexLock locker(&mutex_);
    const auto& constant_data =
        data_.trajectory_nodes.at(node_id).constant_data;
    submap_ids = InitializeGlobalSubmapPoses(
        node_id.trajectory_id, constant_data->time, insertion_submaps);
    CHECK_EQ(submap_ids.size(), insertion_submaps.size());
    const SubmapId matching_id = submap_ids.front();
    const transform::Rigid3d& local_pose = constant_data->local_pose;
    const transform::Rigid3d global_pose =
        optimization_problem_->submap_data().at(matching_id).global_pose *
        insertion_submaps.front()->local_pose().inverse() * local_pose;
    optimization_problem_->AddTrajectoryNode(
        matching_id.trajectory_id,
        optimization::NodeSpec3D{constant_data->time, local_pose, global_pose});
    for (size_t i = 0; i < insertion_submaps.size(); ++i) {
      const SubmapId submap_id = submap_ids[i];
      // Even if this was the last node added to 'submap_id', the submap will
      // only be marked as finished in 'data_.submap_data' further below.
      CHECK(data_.submap_data.at(submap_id).state ==
            SubmapState::kNoConstraintSearch);
      data_.submap_data.at(submap_id).node_ids.emplace(node_id);
      data_.trajectory_nodes.at(node_id).submap_ids.emplace_back(submap_id);
      const transform::Rigid3d constraint_transform =
          insertion_submaps[i]->local_pose().inverse() * local_pose;
      data_.constraints.push_back(Constraint{
          submap_id,
          node_id,
          {constraint_transform, options_.matcher_translation_weight(),
           options_.matcher_rotation_weight()},
          Constraint::INTRA_SUBMAP});
    }
    if (newly_finished_submap) {
      const SubmapId newly_finished_submap_id = submap_ids.front();
      InternalSubmapData& finished_submap_data =
          data_.submap_data.at(newly_finished_submap_id);
      CHECK(finished_submap_data.state == SubmapState::kNoConstraintSearch);
      finished_submap_data.state = SubmapState::kFinished;
    }
  }
  bool pure_localization_trajectory =
      pure_localization_trajectory_ids_.count(node_id.trajectory_id);

  std::vector<SubmapId> submap_candidates_for_local_constraints;
  std::vector<SubmapId> submap_candidates_for_global_constraints;
  std::tie(submap_candidates_for_local_constraints, submap_candidates_for_global_constraints) =
      ComputeCandidatesForConstraints(node_id);
  num_local_constraints_to_compute_ += options_.local_constraints_per_node();
  num_global_constraints_to_compute_ += options_.global_constraints_per_node();
  std::vector<SubmapId> submaps_for_local_constraints;
  std::vector<SubmapId> submaps_for_global_constraints;
  std::tie(submaps_for_local_constraints, submaps_for_global_constraints) =
      SelectCandidatesForConstraints(submap_candidates_for_local_constraints,
          submap_candidates_for_global_constraints);
  CHECK(std::set<SubmapId>(submaps_for_local_constraints.begin(),
      submaps_for_local_constraints.end()).size() ==
      submaps_for_local_constraints.size());
  CHECK(std::set<SubmapId>(submaps_for_global_constraints.begin(),
      submaps_for_global_constraints.end()).size() ==
      submaps_for_global_constraints.size());
  MaybeAddConstraints(node_id, submaps_for_local_constraints, submaps_for_global_constraints);

  if (newly_finished_submap && !pure_localization_trajectory) {
    const SubmapId newly_finished_submap_id = submap_ids.front();
    int newly_finished_submap_num_nodes;
    {
      absl::MutexLock locker(&mutex_);
      newly_finished_submap_num_nodes = data_.submap_data.at(newly_finished_submap_id).node_ids.size();
    }
    std::vector<NodeId> node_candidates_for_local_constraints;
    std::vector<NodeId> node_candidates_for_global_constraints;
    std::tie(node_candidates_for_local_constraints, node_candidates_for_global_constraints) =
        ComputeCandidatesForConstraints(newly_finished_submap_id);
    num_local_constraints_to_compute_ +=
        options_.local_constraints_per_node() * newly_finished_submap_num_nodes / 2;
    num_global_constraints_to_compute_ +=
        options_.global_constraints_per_node() * newly_finished_submap_num_nodes / 2;
    std::vector<NodeId> nodes_for_local_constraints;
    std::vector<NodeId> nodes_for_global_constraints;
    std::tie(nodes_for_local_constraints, nodes_for_global_constraints) =
        SelectCandidatesForConstraints(node_candidates_for_local_constraints,
            node_candidates_for_global_constraints);
    CHECK(std::set<NodeId>(nodes_for_local_constraints.begin(),
        nodes_for_local_constraints.end()).size() ==
        nodes_for_local_constraints.size());
    CHECK(std::set<NodeId>(nodes_for_global_constraints.begin(),
        nodes_for_global_constraints.end()).size() ==
        nodes_for_global_constraints.size());
    MaybeAddConstraints(newly_finished_submap_id, nodes_for_local_constraints, nodes_for_global_constraints);
  }

  constraint_builder_.NotifyEndOfNode();
  absl::MutexLock locker(&mutex_);
  ++num_nodes_since_last_loop_closure_;
  if (options_.optimize_every_n_nodes() > 0 &&
      num_nodes_since_last_loop_closure_ > options_.optimize_every_n_nodes()) {
    return WorkItem::Result::kRunOptimization;
  }
  return WorkItem::Result::kDoNotRunOptimization;
}

NodeId PoseGraph3D::AppendNode(
    std::shared_ptr<const TrajectoryNode::Data> constant_data,
    int trajectory_id,
    const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps,
    const transform::Rigid3d& optimized_pose) {
  absl::MutexLock locker(&mutex_);
  AddTrajectoryIfNeeded(trajectory_id);
  if (!CanAddWorkItemModifying(trajectory_id)) {
    LOG(WARNING) << "AddNode was called for finished or deleted trajectory.";
  }
  const NodeId node_id = data_.trajectory_nodes.Append(
      trajectory_id, TrajectoryNode{constant_data, optimized_pose});
  data_.trajectory_nodes.at(node_id).submap_ids.reserve(2);
  ++data_.num_trajectory_nodes;
  // Test if the 'insertion_submap.back()' is one we never saw before.
  if (data_.submap_data.SizeOfTrajectoryOrZero(trajectory_id) == 0 ||
      std::prev(data_.submap_data.EndOfTrajectory(trajectory_id))
              ->data.submap != insertion_submaps.back()) {
    // We grow 'data_.submap_data' as needed. This code assumes that the first
    // time we see a new submap is as 'insertion_submaps.back()'.
    const SubmapId submap_id =
        data_.submap_data.Append(trajectory_id, InternalSubmapData());
    data_.submap_data.at(submap_id).submap = insertion_submaps.back();
    // LOG(INFO) << "Inserted submap " << submap_id << ".";
    kActiveSubmapsMetric->Increment();
  }
  return node_id;
}

NodeId PoseGraph3D::AddNode(
    std::shared_ptr<const TrajectoryNode::Data> constant_data,
    int trajectory_id,
    const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps) {
  const transform::Rigid3d optimized_pose(
      GetLocalToGlobalTransform(trajectory_id) * constant_data->local_pose);

  const NodeId node_id = AppendNode(constant_data, trajectory_id,
                                    insertion_submaps, optimized_pose);
  // We have to check this here, because it might have changed by the time we
  // execute the lambda.
  const bool newly_finished_submap =
      insertion_submaps.front()->insertion_finished();
  AddWorkItem([=]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    return ComputeConstraintsForNode(node_id, insertion_submaps,
                                     newly_finished_submap);
  });
  return node_id;
}

common::Time PoseGraph3D::GetLatestNodeTime(
    const NodeId& node_id, const SubmapId& submap_id) const {
  common::Time time = data_.trajectory_nodes.at(node_id).constant_data->time;
  const InternalSubmapData& submap_data = data_.submap_data.at(submap_id);
  if (!submap_data.node_ids.empty()) {
    const NodeId last_submap_node_id =
        *data_.submap_data.at(submap_id).node_ids.rbegin();
    time = std::max(
        time,
        data_.trajectory_nodes.at(last_submap_node_id).constant_data->time);
  }
  return time;
}

void PoseGraph3D::UpdateTrajectoryConnectivity(const Constraint& constraint) {
  CHECK_EQ(constraint.tag, PoseGraphInterface::Constraint::INTER_SUBMAP);
  const common::Time time =
      GetLatestNodeTime(constraint.node_id, constraint.submap_id);
  data_.trajectory_connectivity_state.Connect(
      constraint.node_id.trajectory_id, constraint.submap_id.trajectory_id,
      time);
}

double PoseGraph3D::GetTravelledDistanceWithLoopsSameTrajectory(
    NodeId node_1, NodeId node_2, float min_score) {
  CHECK(node_1.trajectory_id == node_2.trajectory_id);
  if (node_1 == node_2) {
    return 0.0;
  }
  if (node_2 < node_1) {
    std::swap(node_1, node_2);
  }
  double travelled_distance =
      data_.trajectory_nodes.at(node_2).constant_data->travelled_distance -
      data_.trajectory_nodes.at(node_1).constant_data->travelled_distance;
  CHECK(travelled_distance >= 0.0);
  for (const Constraint& loop : data_.constraints) {
    if (loop.tag == Constraint::INTRA_SUBMAP) {
      continue;
    }
    if (loop.score < min_score) {
      continue;
    }
    if (loop.node_id.trajectory_id != loop.submap_id.trajectory_id ||
        loop.node_id.trajectory_id != node_1.trajectory_id) {
      continue;
    }
    NodeId loop_node_1 = loop.node_id;
    NodeId loop_node_2 = *data_.submap_data.at(loop.submap_id).node_ids.begin();
    if (loop_node_2 < loop_node_1) {
      std::swap(loop_node_1, loop_node_2);
    }
    if (loop_node_2.node_index <= node_1.node_index ||
        loop_node_1.node_index >= node_2.node_index) {
      continue;
    }
    double travelled_distance_with_loop =
        std::abs(
            data_.trajectory_nodes.at(node_1).constant_data->travelled_distance -
            data_.trajectory_nodes.at(loop_node_1).constant_data->travelled_distance) +
        std::abs(
            data_.trajectory_nodes.at(node_2).constant_data->travelled_distance -
            data_.trajectory_nodes.at(loop_node_2).constant_data->travelled_distance);
    travelled_distance = std::min(travelled_distance, travelled_distance_with_loop);
  }
  return travelled_distance;
}

double PoseGraph3D::GetTravelledDistanceWithLoopsDifferentTrajectories(
    NodeId node_1, NodeId node_2, float min_score) {
  CHECK(node_1.trajectory_id != node_2.trajectory_id);
  if (node_2 < node_1) {
    std::swap(node_1, node_2);
  }
  double travelled_distance = std::numeric_limits<double>::max();
  for (const Constraint& loop : data_.constraints) {
    if (loop.tag == Constraint::INTRA_SUBMAP) {
      continue;
    }
    if (loop.score < min_score) {
      continue;
    }
    if (std::minmax(loop.node_id.trajectory_id, loop.submap_id.trajectory_id) !=
        std::minmax(node_1.trajectory_id, node_2.trajectory_id)) {
      continue;
    }
    NodeId loop_node_1 = loop.node_id;
    NodeId loop_node_2 = *data_.submap_data.at(loop.submap_id).node_ids.begin();
    if (loop_node_2 < loop_node_1) {
      std::swap(loop_node_1, loop_node_2);
    }
    double travelled_distance_with_loop =
        GetTravelledDistanceWithLoopsSameTrajectory(node_1, loop_node_1, min_score) +
        GetTravelledDistanceWithLoopsSameTrajectory(node_2, loop_node_2, min_score);
    travelled_distance = std::min(travelled_distance, travelled_distance_with_loop);
  }
  for (const TrimmedLoop& loop : trimmed_loops_) {
    if (loop.score < min_score) {
      continue;
    }
    if (std::minmax(loop.node_id.trajectory_id, loop.submap_id.trajectory_id) !=
        std::minmax(node_1.trajectory_id, node_2.trajectory_id)) {
      continue;
    }
    if (loop.submap_id.trajectory_id != node_1.trajectory_id) {
      std::swap(node_1, node_2);
    }
    double travelled_distance_1, travelled_distance_2;
    if (data_.submap_data.Contains(loop.submap_id)) {
      const NodeId& loop_node_1 = *data_.submap_data.at(loop.submap_id).node_ids.begin();
      travelled_distance_1 =
          GetTravelledDistanceWithLoopsSameTrajectory(node_1, loop_node_1, min_score);
    } else {
      travelled_distance_1 = std::abs(
          data_.trajectory_nodes.at(node_1).constant_data->travelled_distance -
          loop.travelled_distance_for_submap);
    }
    if (data_.trajectory_nodes.Contains(loop.node_id)) {
      travelled_distance_2 =
          GetTravelledDistanceWithLoopsSameTrajectory(node_2, loop.node_id, min_score);
    } else {
      travelled_distance_2 = std::abs(
          data_.trajectory_nodes.at(node_2).constant_data->travelled_distance -
          loop.travelled_distance_for_node);
    }
    double travelled_distance_with_loop =
        travelled_distance_1 + travelled_distance_2;
    travelled_distance = std::min(travelled_distance, travelled_distance_with_loop);
  }
  return travelled_distance;
}

double PoseGraph3D::GetTravelledDistanceWithLoops(
    NodeId node_1, NodeId node_2, float min_score) {
  if (node_1.trajectory_id == node_2.trajectory_id) {
    return GetTravelledDistanceWithLoopsSameTrajectory(node_1, node_2, min_score);
  } else {
    return GetTravelledDistanceWithLoopsDifferentTrajectories(node_1, node_2, min_score);
  }
}

std::vector<PoseGraphInterface::Constraint>
PoseGraph3D::TrimFalseDetectedLoops(const std::vector<PoseGraphInterface::Constraint>& new_loops) {
  absl::MutexLock locker(&mutex_);
  int num_before = new_loops.size();
  std::vector<PoseGraphInterface::Constraint> true_detected_loops;
  true_detected_loops.reserve(new_loops.size());
  for (const auto& new_loop : new_loops) {
    int loop_trimmer_trajectory_id;
    if (loop_trimmer_options_.count(new_loop.submap_id.trajectory_id)) {
      loop_trimmer_trajectory_id = new_loop.submap_id.trajectory_id;
    } else {
      if (loop_trimmer_options_.count(new_loop.node_id.trajectory_id)) {
        loop_trimmer_trajectory_id = new_loop.node_id.trajectory_id; 
      } else {
        true_detected_loops.push_back(new_loop);
        continue;
      }
    }
    const proto::LoopTrimmerOptions& loop_trimmer_options =
        loop_trimmer_options_.at(loop_trimmer_trajectory_id);
    if (!loop_trimmer_options.trim_false_detected_loops()) {
      true_detected_loops.push_back(new_loop);
      continue;
    }
    const NodeId& new_loop_node_1 = new_loop.node_id;
    const NodeId& new_loop_node_2 =
        *data_.submap_data.at(new_loop.submap_id).node_ids.begin();
    double travelled_distance =
        GetTravelledDistanceWithLoops(new_loop_node_1, new_loop_node_2, new_loop.score);
    const transform::Rigid3d& global_submap_pose =
        data_.global_submap_poses_3d.at(new_loop.submap_id).global_pose;
    const transform::Rigid3d& global_node_pose =
        data_.trajectory_nodes.at(new_loop.node_id).global_pose;
    const transform::Rigid3d relative_node_pose =
        global_submap_pose.inverse() * global_node_pose;
    const double translation_error =
        (new_loop.pose.zbar_ij.translation() - relative_node_pose.translation()).norm();
    if (translation_error / travelled_distance <
        loop_trimmer_options.max_translation_error_travelled_distance_ratio()) {
      true_detected_loops.push_back(new_loop);
    }
  }
  int num_after = true_detected_loops.size();
  if (options_.log_number_of_trimmed_loops()) {
    LOG(INFO) << "Trimmed " << num_before - num_after <<
        " (" << num_before << " -> " << num_after << ") false detected loops";
  }
  return true_detected_loops;
}

void PoseGraph3D::TrimLoopsInWindow() {
  absl::MutexLock locker(&mutex_);
  int num_before = 0;
  MapById<SubmapId, MapById<NodeId, float>> sorted_loops;
  for (const auto& constraint : data_.constraints) {
    if (constraint.tag == Constraint::INTRA_SUBMAP) {
      continue;
    }
    num_before++;
    if (loop_trimmer_options_.count(constraint.submap_id.trajectory_id) == 0 &&
        loop_trimmer_options_.count(constraint.node_id.trajectory_id) == 0) {
      continue;
    }
    if (!sorted_loops.Contains(constraint.submap_id)) {
      sorted_loops.Insert(constraint.submap_id, {});
    }
    sorted_loops.at(constraint.submap_id).Insert(constraint.node_id, constraint.score);
  }

  std::set<std::pair<SubmapId, NodeId>> loops_to_remove;
  for (const auto& entry : sorted_loops) {
    const SubmapId& submap_id = entry.id;
    const auto& connected_nodes = entry.data;
    for (int trajectory_id : connected_nodes.trajectory_ids()) {
      int loop_trimmer_trajectory_id;
      if (loop_trimmer_options_.count(submap_id.trajectory_id)) {
        loop_trimmer_trajectory_id = submap_id.trajectory_id;
      } else {
        loop_trimmer_trajectory_id = trajectory_id;
      }
      const proto::LoopTrimmerOptions& loop_trimmer_options =
          loop_trimmer_options_.at(loop_trimmer_trajectory_id);
      if (!loop_trimmer_options.trim_loops_in_window()) {
        continue;
      }
      int window_first_node_index =
          connected_nodes.BeginOfTrajectory(trajectory_id)->id.node_index;
      NodeId node_id_to_retain(-1, -1);
      float max_score = std::numeric_limits<float>::min();
      for (const auto& connected_node : connected_nodes.trajectory(trajectory_id)) {
        const NodeId& node_id = connected_node.id;
        float score = connected_node.data;
        if (node_id.node_index >= window_first_node_index + loop_trimmer_options.window_size_per_submap()) {
          loops_to_remove.erase(std::make_pair(submap_id, node_id_to_retain));
          window_first_node_index = node_id.node_index;
          max_score = std::numeric_limits<float>::min();
        }
        if (score > max_score) {
          node_id_to_retain = node_id;
          max_score = score;
        }
        loops_to_remove.emplace(submap_id, node_id);
      }
      loops_to_remove.erase(std::make_pair(submap_id, node_id_to_retain));
    }
  }

  int num_after = 0;
  std::vector<PoseGraphInterface::Constraint> trimmed_constraints;
  trimmed_constraints.reserve(data_.constraints.size());
  for (const auto& constraint : data_.constraints) {
    if (constraint.tag == Constraint::INTRA_SUBMAP) {
      trimmed_constraints.push_back(constraint);
    } else if (loops_to_remove.count(std::make_pair(constraint.submap_id, constraint.node_id)) == 0 ||
          data_.trajectory_connectivity_state.LastConnectionTime(
              constraint.submap_id.trajectory_id, constraint.node_id.trajectory_id) ==
          GetLatestNodeTime(constraint.node_id, constraint.submap_id)) {
      trimmed_constraints.push_back(constraint);
      num_after++;
    }
  }
  data_.constraints = std::move(trimmed_constraints);

  if (options_.log_number_of_trimmed_loops()) {
    LOG(INFO) << "Trimmed " << num_before - num_after <<
        " (" << num_before << " -> " << num_after << ") loops using search window";
  }
}

std::vector<PoseGraphInterface::Constraint>
PoseGraph3D::TrimLoops(const std::vector<PoseGraphInterface::Constraint>& new_loops) {
  MEASURE_BLOCK_TIME(trim_loops);
  auto true_detected_loops = TrimFalseDetectedLoops(new_loops);
  TrimLoopsInWindow();
  return true_detected_loops;
}

void PoseGraph3D::HandleWorkQueue(
    const constraints::ConstraintBuilder3D::Result& result) {
  absl::MutexLock queue_locker(&executing_work_item_mutex_);
  if (options_.log_constraints()) {
    for (const Constraint& constraint : result) {
      bool local, global;
      std::tie(local, global) =
          CheckIfConstraintCanBeAdded(constraint.node_id, constraint.submap_id);
      CHECK(local != global);
      std::ostringstream info;
      if (global) {
        info << "Global. ";
      }
      absl::MutexLock locker(&mutex_);
      CHECK(data_.trajectory_nodes.at(constraint.node_id).submap_ids.size());
      SubmapId submap_id_for_node(
          data_.trajectory_nodes.at(constraint.node_id).submap_ids.back());
      info << "Node from " << submap_id_for_node <<
          ", submap " << constraint.submap_id <<
          ", score " << std::setprecision(3) << constraint.score;
      LOG(INFO) << info.str();
    }
  }
  constraints::ConstraintBuilder3D::Result true_detected_loops = TrimLoops(result);
  {
    absl::MutexLock locker(&mutex_);
    data_.constraints.insert(data_.constraints.end(), true_detected_loops.begin(),
                             true_detected_loops.end());
  }

  MEASURE_TIME_FROM_HERE(optimization);
  RunOptimization();
  STOP_TIME_MEASUREMENT(optimization);

  {
    absl::MutexLock locker(&mutex_);
    for (const Constraint& constraint : true_detected_loops) {
      UpdateTrajectoryConnectivity(constraint);
    }
    DeleteTrajectoriesIfNeeded();
    TrimmingHandle trimming_handle(this);
    for (auto& trimmer : trimmers_) {
      trimmer->Trim(&trimming_handle);
    }
    trimmers_.erase(std::remove_if(trimmers_.begin(), trimmers_.end(),
        [](std::unique_ptr<PoseGraphTrimmer>& trimmer) {
          return trimmer->IsFinished();
        }), trimmers_.end());

    num_nodes_since_last_loop_closure_ = 0;

    // Update the gauges that count the current number of constraints.
    double inter_constraints_same_trajectory = 0;
    double inter_constraints_different_trajectory = 0;
    for (const auto& constraint : data_.constraints) {
      if (constraint.tag ==
          cartographer::mapping::PoseGraph::Constraint::INTRA_SUBMAP) {
        continue;
      }
      if (constraint.node_id.trajectory_id ==
          constraint.submap_id.trajectory_id) {
        ++inter_constraints_same_trajectory;
      } else {
        ++inter_constraints_different_trajectory;
      }
    }
    kConstraintsSameTrajectoryMetric->Set(inter_constraints_same_trajectory);
    kConstraintsDifferentTrajectoryMetric->Set(
        inter_constraints_different_trajectory);
  }

  if (global_slam_optimization_callback_) {
    std::map<int, NodeId> trajectory_id_to_last_optimized_node_id;
    std::map<int, SubmapId> trajectory_id_to_last_optimized_submap_id;
    const auto& submap_data = optimization_problem_->submap_data();
    const auto& node_data = optimization_problem_->node_data();
    for (const int trajectory_id : node_data.trajectory_ids()) {
      if (node_data.SizeOfTrajectoryOrZero(trajectory_id) == 0 ||
          submap_data.SizeOfTrajectoryOrZero(trajectory_id) == 0) {
        continue;
      }
      trajectory_id_to_last_optimized_node_id.emplace(
          trajectory_id,
          std::prev(node_data.EndOfTrajectory(trajectory_id))->id);
      trajectory_id_to_last_optimized_submap_id.emplace(
          trajectory_id,
          std::prev(submap_data.EndOfTrajectory(trajectory_id))->id);
    }
    global_slam_optimization_callback_(
        trajectory_id_to_last_optimized_submap_id,
        trajectory_id_to_last_optimized_node_id);
  }
}

void PoseGraph3D::DrainWorkQueue() {
  bool process_work_queue = true;
  size_t work_queue_size;

  static auto last_log_time = std::chrono::steady_clock::now();
  static long unsigned int max_queue_size = 0;
  {
    absl::MutexLock locker(&work_queue_mutex_);
    max_queue_size = max_queue_size < work_queue_->size() ? work_queue_->size() : max_queue_size;
  }
  auto now_time = std::chrono::steady_clock::now();
  if (common::ToSeconds(now_time - last_log_time) > 3) {
    static std::unique_ptr<std::vector<long unsigned int>, std::function<void(std::vector<long unsigned int>*)>> max_queue_size_list_ptr(
      new std::vector<long unsigned int>, 
      [](std::vector<long unsigned int>* list_ptr){
        std::string log_string;
        log_string += "Max queue size list:\n";
        log_string += "    ";
        for (auto max_queue_size : *list_ptr) {
          log_string += std::to_string(max_queue_size) + " ";
        }
        log_string += "\n";
        std::cout << log_string;
        delete list_ptr;
      });
    max_queue_size_list_ptr->push_back(max_queue_size);
    last_log_time = now_time;
    if (options_.log_work_queue_size()) {
      LOG(INFO) << "Work items in queue: " << max_queue_size;
    }
    max_queue_size = 0;
  }

  static time_measurer::TimeMeasurer work_item_processing_latency("work_item_processing_latency", true);
  while (process_work_queue) {
    std::function<WorkItem::Result()> work_item;
    {
      absl::MutexLock locker(&work_queue_mutex_);
      if (work_queue_->empty()) {
        work_queue_.reset();
        return;
      }
      auto add_time = work_queue_->front().time;
      auto now_time = std::chrono::steady_clock::now();
      work_item_processing_latency.AddMeasurement(common::ToSeconds(now_time - add_time));
      work_item = work_queue_->front().task;
      work_queue_->pop_front();
      work_queue_size = work_queue_->size();
      kWorkQueueSizeMetric->Set(work_queue_size);
    }
    process_work_queue = work_item() == WorkItem::Result::kDoNotRunOptimization;
  }
  // LOG(INFO) << "Optimization requested.";
  // We have to optimize again.
  constraint_builder_.WhenDone(
      [this](const constraints::ConstraintBuilder3D::Result& result) {
        HandleWorkQueue(result);
        DrainWorkQueue();
      });
}

void PoseGraph3D::RunOptimization() {
  if (optimization_problem_->submap_data().empty()) {
    return;
  }

  // No other thread is accessing the optimization_problem_, data_.constraints,
  // data_.frozen_trajectories and data_.landmark_nodes when executing the
  // Solve. Solve is time consuming, so not taking the mutex before Solve to
  // avoid blocking foreground processing.
  optimization_problem_->Solve(data_.constraints, GetTrajectoryStates(),
                               data_.landmark_nodes);


  absl::MutexLock locker(&mutex_);
  const auto& submap_data = optimization_problem_->submap_data();
  const auto& node_data = optimization_problem_->node_data();
  for (const int trajectory_id : node_data.trajectory_ids()) {
    for (const auto& node : node_data.trajectory(trajectory_id)) {
      data_.trajectory_nodes.at(node.id).global_pose = node.data.global_pose;
    }

    // Extrapolate all point cloud poses that were not included in the
    // 'optimization_problem_' yet.
    const auto local_to_new_global =
        ComputeLocalToGlobalTransform(submap_data, trajectory_id);
    const auto local_to_old_global = ComputeLocalToGlobalTransform(
        data_.global_submap_poses_3d, trajectory_id);
    const transform::Rigid3d old_global_to_new_global =
        local_to_new_global * local_to_old_global.inverse();

    const NodeId last_optimized_node_id =
        std::prev(node_data.EndOfTrajectory(trajectory_id))->id;
    auto node_it =
        std::next(data_.trajectory_nodes.find(last_optimized_node_id));
    for (; node_it != data_.trajectory_nodes.EndOfTrajectory(trajectory_id);
         ++node_it) {
      auto& mutable_trajectory_node = data_.trajectory_nodes.at(node_it->id);
      mutable_trajectory_node.global_pose =
          old_global_to_new_global * mutable_trajectory_node.global_pose;
    }
  }
  for (const auto& landmark : optimization_problem_->landmark_data()) {
    data_.landmark_nodes[landmark.first].global_landmark_pose = landmark.second;
  }
  data_.global_submap_poses_3d = submap_data;

  // Log the histograms for the pose residuals.
  if (options_.log_residual_histograms()) {
    LogResidualHistograms();
  }
}

void PoseGraph3D::RunFinalOptimization() {
  AddWorkItem([this]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    optimization_problem_->SetMaxNumIterations(
        options_.max_num_final_iterations());
    return WorkItem::Result::kRunOptimization;
  });
  AddWorkItem([this]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    optimization_problem_->SetMaxNumIterations(
        options_.optimization_problem_options()
            .ceres_solver_options().max_num_iterations());
    return WorkItem::Result::kDoNotRunOptimization;
  });
  WaitForAllComputations();
}

void PoseGraph3D::WaitForAllComputations() {
  int num_trajectory_nodes;
  {
    absl::MutexLock locker(&mutex_);
    num_trajectory_nodes = data_.num_trajectory_nodes;
  }

  const int num_finished_nodes_at_start =
      constraint_builder_.GetNumFinishedNodes();

  auto report_progress = [this, num_trajectory_nodes,
                          num_finished_nodes_at_start]() {
    // Log progress on nodes only when we are actually processing nodes.
    if (num_trajectory_nodes != num_finished_nodes_at_start) {
      std::ostringstream progress_info;
      progress_info << "Optimizing: " << std::fixed << std::setprecision(1)
                    << 100. *
                           (constraint_builder_.GetNumFinishedNodes() -
                            num_finished_nodes_at_start) /
                           (num_trajectory_nodes - num_finished_nodes_at_start)
                    << "%...";
      std::cout << "\r\x1b[K" << progress_info.str() << std::flush;
    }
  };

  // First wait for the work queue to drain so that it's safe to schedule
  // a WhenDone() callback.
  {
    const auto predicate = [this]() {
      return work_queue_ == nullptr;
    };
    absl::MutexLock locker(&work_queue_mutex_);
    while (!work_queue_mutex_.AwaitWithTimeout(
        absl::Condition(&predicate),
        absl::FromChrono(common::FromSeconds(1.)))) {
      report_progress();
    }
  }

  // Now wait for any pending constraint computations to finish.
  absl::MutexLock locker(&mutex_);
  bool notification = false;
  constraint_builder_.WhenDone(
      [this,
       &notification](const constraints::ConstraintBuilder3D::Result& result) {
            absl::MutexLock locker(&mutex_);
            data_.constraints.insert(data_.constraints.end(), result.begin(),
                                     result.end());
            notification = true;
          });
  const auto predicate = [&notification]() {
    return notification;
  };
  while (!mutex_.AwaitWithTimeout(absl::Condition(&predicate),
                                  absl::FromChrono(common::FromSeconds(1.)))) {
    report_progress();
  }
  CHECK_EQ(constraint_builder_.GetNumFinishedNodes(), num_trajectory_nodes);
  std::cout << "\r\x1b[KOptimizing: Done.     " << std::endl;
}

void PoseGraph3D::DeleteTrajectory(int trajectory_id) {
  {
    absl::MutexLock locker(&mutex_);
    auto it = data_.trajectories_state.find(trajectory_id);
    if (it == data_.trajectories_state.end()) {
      LOG(WARNING) << "Skipping request to delete non-existing trajectory_id: "
                   << trajectory_id;
      return;
    }
    it->second.deletion_state =
        InternalTrajectoryState::DeletionState::SCHEDULED_FOR_DELETION;
  }
  AddWorkItem([this, trajectory_id]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    CHECK(data_.trajectories_state.at(trajectory_id).state !=
          TrajectoryState::ACTIVE);
    CHECK(data_.trajectories_state.at(trajectory_id).state !=
          TrajectoryState::DELETED);
    CHECK(data_.trajectories_state.at(trajectory_id).deletion_state ==
          InternalTrajectoryState::DeletionState::SCHEDULED_FOR_DELETION);
    data_.trajectories_state.at(trajectory_id).deletion_state =
        InternalTrajectoryState::DeletionState::WAIT_FOR_DELETION;
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::FinishTrajectory(int trajectory_id) {
  AddWorkItem([this, trajectory_id]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    CHECK(!IsTrajectoryFinishedUnderLock(trajectory_id));
    data_.trajectories_state[trajectory_id].state = TrajectoryState::FINISHED;

    for (const auto& submap : data_.submap_data.trajectory(trajectory_id)) {
      data_.submap_data.at(submap.id).state = SubmapState::kFinished;
    }
    return WorkItem::Result::kRunOptimization;
  });
}

bool PoseGraph3D::IsTrajectoryFinished(int trajectory_id) const {
  absl::MutexLock locker(&mutex_);
  return IsTrajectoryFinishedUnderLock(trajectory_id);
}

bool PoseGraph3D::IsTrajectoryFinishedUnderLock(int trajectory_id) const {
  return data_.trajectories_state.count(trajectory_id) != 0 &&
         data_.trajectories_state.at(trajectory_id).state ==
             TrajectoryState::FINISHED;
}

void PoseGraph3D::FreezeTrajectory(int trajectory_id) {
  {
    absl::MutexLock locker(&mutex_);
    data_.trajectory_connectivity_state.Add(trajectory_id);
  }
  AddWorkItem([this, trajectory_id]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    CHECK(!IsTrajectoryFrozenUnderLock(trajectory_id));
    // Connect multiple frozen trajectories among each other.
    // This is required for localization against multiple frozen trajectories
    // because we lose inter-trajectory constraints when freezing.
    for (const auto& entry : data_.trajectories_state) {
      const int other_trajectory_id = entry.first;
      if (!IsTrajectoryFrozenUnderLock(other_trajectory_id)) {
        continue;
      }
      if (data_.trajectory_connectivity_state.TransitivelyConnected(
              trajectory_id, other_trajectory_id)) {
        // Already connected, nothing to do.
        continue;
      }
      data_.trajectory_connectivity_state.Connect(
          trajectory_id, other_trajectory_id, common::FromUniversal(0));
    }
    data_.trajectories_state[trajectory_id].state = TrajectoryState::FROZEN;
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

bool PoseGraph3D::IsTrajectoryFrozen(int trajectory_id) const {
  absl::MutexLock locker(&mutex_);
  return IsTrajectoryFrozenUnderLock(trajectory_id);
}

bool PoseGraph3D::IsTrajectoryFrozenUnderLock(int trajectory_id) const {
  return data_.trajectories_state.count(trajectory_id) != 0 &&
         data_.trajectories_state.at(trajectory_id).state ==
             TrajectoryState::FROZEN;
}

void PoseGraph3D::AddSubmapFromProto(
    const transform::Rigid3d& global_submap_pose,
    const proto::Submap& submap) {
  if (!submap.has_submap_3d()) {
    return;
  }

  const SubmapId submap_id = {submap.submap_id().trajectory_id(),
                              submap.submap_id().submap_index()};
  std::shared_ptr<const Submap3D> submap_ptr =
      std::make_shared<const Submap3D>(submap.submap_3d());

  {
    absl::MutexLock locker(&mutex_);
    AddTrajectoryIfNeeded(submap_id.trajectory_id);
    if (!CanAddWorkItemModifying(submap_id.trajectory_id)) return;
    data_.submap_data.Insert(submap_id, InternalSubmapData());
    data_.submap_data.at(submap_id).submap = submap_ptr;
    // Immediately show the submap at the 'global_submap_pose'.
    data_.global_submap_poses_3d.Insert(
        submap_id, optimization::SubmapSpec3D{global_submap_pose});
    // TODO(MichaelGrupp): MapBuilder does freezing before deserializing submaps,
    // so this should be fine.
    if (IsTrajectoryFrozenUnderLock(submap_id.trajectory_id)) {
      kFrozenSubmapsMetric->Increment();
    } else {
      kActiveSubmapsMetric->Increment();
    }
  }

  AddWorkItem([this, submap_id, global_submap_pose]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    data_.submap_data.at(submap_id).state = SubmapState::kFinished;
    optimization_problem_->InsertSubmap(submap_id, global_submap_pose);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddNodeFromProto(
    const transform::Rigid3d& global_pose,
    const proto::Node& node) {
  const NodeId node_id = {node.node_id().trajectory_id(),
                          node.node_id().node_index()};
  std::shared_ptr<const TrajectoryNode::Data> constant_data =
      std::make_shared<const TrajectoryNode::Data>(FromProto(node.node_data()));

  {
    absl::MutexLock locker(&mutex_);
    AddTrajectoryIfNeeded(node_id.trajectory_id);
    if (!CanAddWorkItemModifying(node_id.trajectory_id)) return;
    data_.trajectory_nodes.Insert(node_id,
        TrajectoryNode{constant_data, global_pose});
    data_.trajectory_nodes.at(node_id).submap_ids.reserve(2);
  }

  AddWorkItem([this, node_id, global_pose]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    const auto& constant_data =
        data_.trajectory_nodes.at(node_id).constant_data;
    optimization_problem_->InsertTrajectoryNode(
        node_id,
        optimization::NodeSpec3D{
            constant_data->time, constant_data->local_pose, global_pose});
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::SetTrajectoryDataFromProto(
    const proto::TrajectoryData& data) {
  TrajectoryData trajectory_data;
  trajectory_data.gravity_constant = data.gravity_constant();
  trajectory_data.imu_calibration = {
      {data.imu_calibration().w(), data.imu_calibration().x(),
       data.imu_calibration().y(), data.imu_calibration().z()}};
  if (data.has_fixed_frame_origin_in_map()) {
    trajectory_data.fixed_frame_origin_in_map =
        transform::ToRigid3(data.fixed_frame_origin_in_map());
  }

  const int trajectory_id = data.trajectory_id();
  AddWorkItem([this, trajectory_id, trajectory_data]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      optimization_problem_->SetTrajectoryData(trajectory_id, trajectory_data);
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::SetLandmarkPose(
    const std::string& landmark_id,
    const transform::Rigid3d& global_pose,
    bool frozen) {
  AddWorkItem([=]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    data_.landmark_nodes[landmark_id].global_landmark_pose = global_pose;
    data_.landmark_nodes[landmark_id].frozen = frozen;
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddSerializedConstraints(
    const std::vector<Constraint>& constraints) {
  AddWorkItem([this, constraints]()
        LOCKS_EXCLUDED(mutex_)
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    absl::MutexLock locker(&mutex_);
    for (const auto& constraint : constraints) {
      CHECK(data_.trajectory_nodes.Contains(constraint.node_id));
      CHECK(data_.submap_data.Contains(constraint.submap_id));
      CHECK(data_.trajectory_nodes.at(constraint.node_id).constant_data !=
            nullptr);
      CHECK(data_.submap_data.at(constraint.submap_id).submap != nullptr);
      switch (constraint.tag) {
        case Constraint::Tag::INTRA_SUBMAP:
          {
            bool added = data_.submap_data.at(constraint.submap_id)
                .node_ids.emplace(constraint.node_id).second;
            CHECK(added);
            std::vector<SubmapId>& submap_ids_for_node =
                data_.trajectory_nodes.at(constraint.node_id).submap_ids;
            submap_ids_for_node.emplace_back(constraint.submap_id);
            std::sort(submap_ids_for_node.begin(), submap_ids_for_node.end());
          }
          break;
        case Constraint::Tag::INTER_SUBMAP:
          UpdateTrajectoryConnectivity(constraint);
          break;
      }
      data_.constraints.push_back(constraint);
    }
    LOG(INFO) << "Loaded " << constraints.size() << " constraints.";
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddTrimmer(std::unique_ptr<PoseGraphTrimmer> trimmer) {
  PoseGraphTrimmer* const trimmer_ptr = trimmer.release();
  AddWorkItem([this, trimmer_ptr]()
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    auto* pure_localization_trimmer_ptr =
        dynamic_cast<PureLocalizationTrimmer*>(trimmer_ptr);
    if (pure_localization_trimmer_ptr) {
      pure_localization_trajectory_ids_.insert(
          pure_localization_trimmer_ptr->trajectory_id());
    }
    trimmers_.emplace_back(trimmer_ptr);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddLoopTrimmer(int trajectory_id,
    const proto::LoopTrimmerOptions& loop_trimmer_options) {
  AddWorkItem([this, trajectory_id, loop_trimmer_options]()
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    loop_trimmer_options_.emplace(trajectory_id, loop_trimmer_options);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

bool PoseGraph3D::CanAddWorkItemModifying(int trajectory_id) {
  auto it = data_.trajectories_state.find(trajectory_id);
  if (it == data_.trajectories_state.end()) {
    return true;
  }
  if (it->second.state == TrajectoryState::FINISHED) {
    // TODO(gaschler): Replace all FATAL to WARNING after some testing.
    LOG(FATAL) << "trajectory_id " << trajectory_id
               << " has finished "
                  "but modification is requested, skipping.";
    return false;
  }
  if (it->second.deletion_state !=
      InternalTrajectoryState::DeletionState::NORMAL) {
    LOG(FATAL) << "trajectory_id " << trajectory_id
               << " has been scheduled for deletion "
                  "but modification is requested, skipping.";
    return false;
  }
  if (it->second.state == TrajectoryState::DELETED) {
    LOG(FATAL) << "trajectory_id " << trajectory_id
               << " has been deleted "
                  "but modification is requested, skipping.";
    return false;
  }
  return true;
}

void PoseGraph3D::SetInitialTrajectoryPose(
    int from_trajectory_id, int to_trajectory_id,
    const transform::Rigid3d& pose, common::Time time) {
  absl::MutexLock locker(&mutex_);
  data_.initial_trajectory_poses[from_trajectory_id] =
      InitialTrajectoryPose{to_trajectory_id, pose, time};
}

transform::Rigid3d PoseGraph3D::GetInterpolatedGlobalTrajectoryPose(
    int trajectory_id, const common::Time time) const {
  CHECK_GT(data_.trajectory_nodes.SizeOfTrajectoryOrZero(trajectory_id), 0);
  const auto it = data_.trajectory_nodes.lower_bound(trajectory_id, time);
  if (it == data_.trajectory_nodes.BeginOfTrajectory(trajectory_id)) {
    return data_.trajectory_nodes.BeginOfTrajectory(trajectory_id)
        ->data.global_pose;
  }
  if (it == data_.trajectory_nodes.EndOfTrajectory(trajectory_id)) {
    return std::prev(data_.trajectory_nodes.EndOfTrajectory(trajectory_id))
        ->data.global_pose;
  }
  return transform::Interpolate(
             transform::TimestampedTransform{std::prev(it)->data.time(),
                                             std::prev(it)->data.global_pose},
             transform::TimestampedTransform{it->data.time(),
                                             it->data.global_pose},
             time)
      .transform;
}

transform::Rigid3d PoseGraph3D::ComputeLocalToGlobalTransform(
    const MapById<SubmapId, optimization::SubmapSpec3D>& global_submap_poses,
    int trajectory_id) const {
  auto begin_it = global_submap_poses.BeginOfTrajectory(trajectory_id);
  auto end_it = global_submap_poses.EndOfTrajectory(trajectory_id);
  if (begin_it == end_it) {
    const auto it = data_.initial_trajectory_poses.find(trajectory_id);
    if (it != data_.initial_trajectory_poses.end()) {
      return GetInterpolatedGlobalTrajectoryPose(it->second.to_trajectory_id,
                                                 it->second.time) *
             it->second.relative_pose;
    } else {
      return transform::Rigid3d::Identity();
    }
  }
  const SubmapId last_optimized_submap_id = std::prev(end_it)->id;
  // Accessing 'local_pose' in Submap is okay, since the member is const.
  return global_submap_poses.at(last_optimized_submap_id).global_pose *
         data_.submap_data.at(last_optimized_submap_id)
             .submap->local_pose()
             .inverse();
}

std::vector<std::vector<int>> PoseGraph3D::GetConnectedTrajectories() const {
  absl::MutexLock locker(&mutex_);
  return data_.trajectory_connectivity_state.Components();
}

bool PoseGraph3D::TrajectoriesTransitivelyConnected(
    int trajectory_id_a, int trajectory_id_b) const {
  absl::MutexLock locker(&mutex_);
  return data_.trajectory_connectivity_state.TransitivelyConnected(
      trajectory_id_a, trajectory_id_b);
}

common::Time PoseGraph3D::TrajectoriesLastConnectionTime(
    int trajectory_id_a, int trajectory_id_b) const {
  absl::MutexLock locker(&mutex_);
  return data_.trajectory_connectivity_state.LastConnectionTime(
      trajectory_id_a, trajectory_id_b);
}

transform::Rigid3d PoseGraph3D::GetLocalToGlobalTransform(
    int trajectory_id) const {
  absl::MutexLock locker(&mutex_);
  return ComputeLocalToGlobalTransform(
      data_.global_submap_poses_3d, trajectory_id);
}

PoseGraphInterface::SubmapData PoseGraph3D::GetSubmapData(
    const SubmapId& submap_id) const {
  absl::MutexLock locker(&mutex_);
  return GetSubmapDataUnderLock(submap_id);
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph3D::GetAllSubmapData() const {
  absl::MutexLock locker(&mutex_);
  return GetSubmapDataUnderLock();
}

MapById<SubmapId, PoseGraphInterface::SubmapPose>
PoseGraph3D::GetAllSubmapPoses() const {
  absl::MutexLock locker(&mutex_);
  MapById<SubmapId, SubmapPose> submap_poses;
  for (const auto& submap_id_data : data_.submap_data) {
    auto submap_data = GetSubmapDataUnderLock(submap_id_data.id);
    submap_poses.Insert(
        submap_id_data.id,
        PoseGraphInterface::SubmapPose{submap_data.submap->num_range_data(),
                                       submap_data.pose});
  }
  return submap_poses;
}

PoseGraphInterface::SubmapData PoseGraph3D::GetSubmapDataUnderLock(
    const SubmapId& submap_id) const {
  const auto it = data_.submap_data.find(submap_id);
  if (it == data_.submap_data.end()) {
    return {};
  }
  auto submap = it->data.submap;
  if (data_.global_submap_poses_3d.Contains(submap_id)) {
    // We already have an optimized pose.
    return {submap, data_.global_submap_poses_3d.at(submap_id).global_pose};
  }
  // We have to extrapolate.
  return {submap,
      ComputeLocalToGlobalTransform(
          data_.global_submap_poses_3d,
          submap_id.trajectory_id) * submap->local_pose()};
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph3D::GetSubmapDataUnderLock() const {
  MapById<SubmapId, PoseGraphInterface::SubmapData> submaps;
  for (const auto& submap_id_data : data_.submap_data) {
    submaps.Insert(
        submap_id_data.id,
        GetSubmapDataUnderLock(submap_id_data.id));
  }
  return submaps;
}

MapById<NodeId, TrajectoryNode> PoseGraph3D::GetTrajectoryNodes() const {
  absl::MutexLock locker(&mutex_);
  return data_.trajectory_nodes;
}

MapById<NodeId, TrajectoryNodePose> PoseGraph3D::GetTrajectoryNodePoses() const {
  MapById<NodeId, TrajectoryNodePose> node_poses;
  absl::MutexLock locker(&mutex_);
  for (const auto& node_id_data : data_.trajectory_nodes) {
    absl::optional<TrajectoryNodePose::ConstantPoseData> constant_pose_data;
    if (node_id_data.data.constant_data != nullptr) {
      constant_pose_data = TrajectoryNodePose::ConstantPoseData{
          node_id_data.data.constant_data->time,
          node_id_data.data.constant_data->local_pose};
    }
    node_poses.Insert(
        node_id_data.id,
        TrajectoryNodePose{node_id_data.data.global_pose, constant_pose_data});
  }
  return node_poses;
}

std::map<int, PoseGraphInterface::TrajectoryState>
PoseGraph3D::GetTrajectoryStates() const {
  std::map<int, PoseGraphInterface::TrajectoryState> trajectories_state;
  absl::MutexLock locker(&mutex_);
  for (const auto& it : data_.trajectories_state) {
    trajectories_state[it.first] = it.second.state;
  }
  return trajectories_state;
}

std::map<std::string, transform::Rigid3d> PoseGraph3D::GetLandmarkPoses() const {
  std::map<std::string, transform::Rigid3d> landmark_poses;
  absl::MutexLock locker(&mutex_);
  for (const auto& landmark : data_.landmark_nodes) {
    // Landmark without value has not been optimized yet.
    if (!landmark.second.global_landmark_pose.has_value()) continue;
    landmark_poses[landmark.first] =
        landmark.second.global_landmark_pose.value();
  }
  return landmark_poses;
}

sensor::MapByTime<sensor::ImuData> PoseGraph3D::GetImuData() const {
  absl::MutexLock locker(&mutex_);
  return optimization_problem_->imu_data();
}

sensor::MapByTime<sensor::OdometryData> PoseGraph3D::GetOdometryData() const {
  absl::MutexLock locker(&mutex_);
  return optimization_problem_->odometry_data();
}

sensor::MapByTime<sensor::FixedFramePoseData>
PoseGraph3D::GetFixedFramePoseData() const {
  absl::MutexLock locker(&mutex_);
  return optimization_problem_->fixed_frame_pose_data();
}

std::map<std::string, PoseGraphInterface::LandmarkNode>
PoseGraph3D::GetLandmarkNodes() const {
  absl::MutexLock locker(&mutex_);
  return data_.landmark_nodes;
}

std::map<int, PoseGraphInterface::TrajectoryData>
PoseGraph3D::GetTrajectoryData() const {
  absl::MutexLock locker(&mutex_);
  return optimization_problem_->trajectory_data();
}

std::vector<PoseGraphInterface::Constraint> PoseGraph3D::constraints() const {
  absl::MutexLock locker(&mutex_);
  return data_.constraints;
}

void PoseGraph3D::SetGlobalSlamOptimizationCallback(
    PoseGraphInterface::GlobalSlamOptimizationCallback callback) {
  AddWorkItem([this, callback = std::move(callback)]()
        LOCKS_EXCLUDED(executing_work_item_mutex_) {
    absl::MutexLock queue_locker(&executing_work_item_mutex_);
    global_slam_optimization_callback_ = std::move(callback);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::LogResidualHistograms() const {
  common::Histogram rotational_residual;
  common::Histogram translational_residual;
  for (const Constraint& constraint : data_.constraints) {
    if (constraint.tag == Constraint::Tag::INTRA_SUBMAP) {
      const cartographer::transform::Rigid3d optimized_node_to_map =
          data_.trajectory_nodes.at(constraint.node_id).global_pose;
      const cartographer::transform::Rigid3d node_to_submap_constraint =
          constraint.pose.zbar_ij;
      const cartographer::transform::Rigid3d optimized_submap_to_map =
          data_.global_submap_poses_3d.at(constraint.submap_id).global_pose;
      const cartographer::transform::Rigid3d optimized_node_to_submap =
          optimized_submap_to_map.inverse() * optimized_node_to_map;
      const cartographer::transform::Rigid3d residual =
          node_to_submap_constraint.inverse() * optimized_node_to_submap;
      rotational_residual.Add(
          common::NormalizeAngleDifference(transform::GetAngle(residual)));
      translational_residual.Add(residual.translation().norm());
    }
  }
  LOG(INFO) << "Translational residuals histogram:\n"
            << translational_residual.ToString(10);
  LOG(INFO) << "Rotational residuals histogram:\n"
            << rotational_residual.ToString(10);
}

void PoseGraph3D::RegisterMetrics(metrics::FamilyFactory* family_factory) {
  auto* latency = family_factory->NewGaugeFamily(
      "mapping_3d_pose_graph_work_queue_delay",
      "Age of the oldest entry in the work queue in seconds");
  kWorkQueueDelayMetric = latency->Add({});
  auto* queue_size =
      family_factory->NewGaugeFamily("mapping_3d_pose_graph_work_queue_size",
                                     "Number of items in the work queue");
  kWorkQueueSizeMetric = queue_size->Add({});
  auto* constraints = family_factory->NewGaugeFamily(
      "mapping_3d_pose_graph_constraints",
      "Current number of constraints in the pose graph");
  kConstraintsDifferentTrajectoryMetric =
      constraints->Add({{"tag", "inter_submap"}, {"trajectory", "different"}});
  kConstraintsSameTrajectoryMetric =
      constraints->Add({{"tag", "inter_submap"}, {"trajectory", "same"}});
  auto* submaps = family_factory->NewGaugeFamily(
      "mapping_3d_pose_graph_submaps", "Number of submaps in the pose graph.");
  kActiveSubmapsMetric = submaps->Add({{"state", "active"}});
  kFrozenSubmapsMetric = submaps->Add({{"state", "frozen"}});
  kDeletedSubmapsMetric = submaps->Add({{"state", "deleted"}});
}

PoseGraph3D::TrimmingHandle::TrimmingHandle(PoseGraph3D* const parent)
    : parent_(parent) {}

int PoseGraph3D::TrimmingHandle::num_submaps(int trajectory_id) const {
  const auto& submap_data = parent_->optimization_problem_->submap_data();
  return submap_data.SizeOfTrajectoryOrZero(trajectory_id);
}

std::vector<SubmapId> PoseGraph3D::TrimmingHandle::GetSubmapIds(
    int trajectory_id) const {
  std::vector<SubmapId> submap_ids;
  const auto& submap_data = parent_->optimization_problem_->submap_data();
  for (const auto& it : submap_data.trajectory(trajectory_id)) {
    submap_ids.push_back(it.id);
  }
  return submap_ids;
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph3D::TrimmingHandle::GetOptimizedSubmapData() const {
  MapById<SubmapId, PoseGraphInterface::SubmapData> submaps;
  for (const auto& submap_id_data : parent_->data_.submap_data) {
    if (submap_id_data.data.state != SubmapState::kFinished ||
        !parent_->data_.global_submap_poses_3d.Contains(submap_id_data.id)) {
      continue;
    }
    submaps.Insert(
        submap_id_data.id,
        SubmapData{submap_id_data.data.submap,
                   parent_->data_.global_submap_poses_3d.at(submap_id_data.id)
                       .global_pose});
  }
  return submaps;
}

const MapById<NodeId, TrajectoryNode>&
PoseGraph3D::TrimmingHandle::GetTrajectoryNodes() const {
  return parent_->data_.trajectory_nodes;
}

const std::vector<PoseGraphInterface::Constraint>&
PoseGraph3D::TrimmingHandle::GetConstraints() const {
  return parent_->data_.constraints;
}

bool PoseGraph3D::TrimmingHandle::IsFinished(int trajectory_id) const {
  return parent_->IsTrajectoryFinishedUnderLock(trajectory_id);
}

void PoseGraph3D::TrimmingHandle::SetTrajectoryState(int trajectory_id,
                                                     TrajectoryState state) {
  parent_->data_.trajectories_state[trajectory_id].state = state;
}

void PoseGraph3D::TrimmingHandle::TrimSubmap(const SubmapId& submap_id) {
  // TODO(hrapp): We have to make sure that the trajectory has been finished
  // if we want to delete the last submaps.
  CHECK(parent_->data_.submap_data.at(submap_id).state ==
        SubmapState::kFinished);

  for (const NodeId& node_id : parent_->data_.submap_data.at(submap_id).node_ids) {
    std::vector<SubmapId>& submap_ids_for_node =
        parent_->data_.trajectory_nodes.at(node_id).submap_ids;
    auto it = std::find(submap_ids_for_node.begin(),
        submap_ids_for_node.end(), submap_id);
    CHECK(it != submap_ids_for_node.end());
    submap_ids_for_node.erase(it);
  }

  // Compile all nodes that are still INTRA_SUBMAP constrained to other submaps
  // once the submap with 'submap_id' is gone.
  // We need to use node_ids instead of constraints here to be also compatible
  // with frozen trajectories that don't have intra-constraints.
  std::set<NodeId> nodes_to_retain;
  for (const auto& submap_data : parent_->data_.submap_data) {
    if (submap_data.id != submap_id) {
      nodes_to_retain.insert(submap_data.data.node_ids.begin(),
                             submap_data.data.node_ids.end());
    }
  }

  // Remove all nodes that are exlusively associated to 'submap_id'.
  std::set<NodeId> nodes_to_remove;
  std::set_difference(parent_->data_.submap_data.at(submap_id).node_ids.begin(),
                      parent_->data_.submap_data.at(submap_id).node_ids.end(),
                      nodes_to_retain.begin(), nodes_to_retain.end(),
                      std::inserter(nodes_to_remove, nodes_to_remove.begin()));

  // Remove all 'data_.constraints' related to 'submap_id'.
  {
    std::vector<Constraint> constraints;
    for (const Constraint& constraint : parent_->data_.constraints) {
      if (constraint.submap_id != submap_id) {
        constraints.push_back(constraint);
      } else {
        if (constraint.tag == Constraint::INTER_SUBMAP) {
          const NodeId& first_node_id_for_submap =
              *parent_->data_.submap_data.at(constraint.submap_id).node_ids.begin();
          parent_->trimmed_loops_.push_back(
              TrimmedLoop{constraint.submap_id, constraint.node_id,
                  constraint.score,
                  parent_->data_.trajectory_nodes.at(
                      first_node_id_for_submap).constant_data->travelled_distance,
                  parent_->data_.trajectory_nodes.at(
                      constraint.node_id).constant_data->travelled_distance});
        }
      }
    }
    parent_->data_.constraints = std::move(constraints);
  }

  // Remove all 'data_.constraints' related to 'nodes_to_remove'.
  // If the removal lets other submaps lose all their inter-submap constraints,
  // delete their corresponding constraint submap matchers to save memory.
  {
    std::vector<Constraint> constraints;
    std::set<SubmapId> other_submap_ids_losing_constraints;
    for (const Constraint& constraint : parent_->data_.constraints) {
      if (nodes_to_remove.count(constraint.node_id) == 0) {
        constraints.push_back(constraint);
      } else {
        // A constraint to another submap will be removed, mark it as affected.
        other_submap_ids_losing_constraints.insert(constraint.submap_id);
        if (constraint.tag == Constraint::INTER_SUBMAP) {
          const NodeId& first_node_id_for_submap =
              *parent_->data_.submap_data.at(constraint.submap_id).node_ids.begin();
          parent_->trimmed_loops_.push_back(
              TrimmedLoop{constraint.submap_id, constraint.node_id,
                  constraint.score,
                  parent_->data_.trajectory_nodes.at(
                      first_node_id_for_submap).constant_data->travelled_distance,
                  parent_->data_.trajectory_nodes.at(
                      constraint.node_id).constant_data->travelled_distance});
        }
      }
    }
    parent_->data_.constraints = std::move(constraints);
    // Go through the remaining constraints to ensure we only delete scan
    // matchers of other submaps that have no inter-submap constraints left.
    for (const Constraint& constraint : parent_->data_.constraints) {
      if (constraint.tag == Constraint::Tag::INTRA_SUBMAP) {
        continue;
      } else if (other_submap_ids_losing_constraints.count(
                     constraint.submap_id)) {
        // This submap still has inter-submap constraints - ignore it.
        other_submap_ids_losing_constraints.erase(constraint.submap_id);
      }
    }
    // Delete scan matchers of the submaps that lost all constraints.
    // TODO(wohe): An improvement to this implementation would be to add the
    // caching logic at the constraint builder which could keep around only
    // recently used scan matchers.
    for (const SubmapId& submap_id : other_submap_ids_losing_constraints) {
      parent_->constraint_builder_.DeleteScanMatcher(submap_id);
    }
  }

  // Mark the submap with 'submap_id' as trimmed and remove its data.
  CHECK(parent_->data_.submap_data.at(submap_id).state ==
        SubmapState::kFinished);
  parent_->data_.submap_data.Trim(submap_id);
  parent_->constraint_builder_.DeleteScanMatcher(submap_id);
  parent_->optimization_problem_->TrimSubmap(submap_id);

  // We have one submap less, update the gauge metrics.
  kDeletedSubmapsMetric->Increment();
  if (parent_->IsTrajectoryFrozenUnderLock(submap_id.trajectory_id)) {
    kFrozenSubmapsMetric->Decrement();
  } else {
    kActiveSubmapsMetric->Decrement();
  }

  // Remove the 'nodes_to_remove' from the pose graph and the optimization
  // problem.
  for (const NodeId& node_id : nodes_to_remove) {
    parent_->data_.trajectory_nodes.Trim(node_id);
    parent_->optimization_problem_->TrimTrajectoryNode(node_id);
  }
}

}  // namespace mapping
}  // namespace cartographer
