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
#include <random>
#include <utility>

#include "Eigen/Eigenvalues"
#include "absl/memory/memory.h"
#include "cartographer/common/math.h"
#include "cartographer/mapping/proto/pose_graph/constraint_builder_options.pb.h"
#include "cartographer/sensor/compressed_point_cloud.h"
#include "cartographer/sensor/internal/voxel_filter.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

#include "kas_utils/utils.hpp"
#include "kas_utils/collection.hpp"

namespace cartographer {
namespace mapping {

static auto* kWorkQueueDelayMetric = metrics::Gauge::Null();
static auto* kWorkQueueSizeMetric = metrics::Gauge::Null();
static auto* kConstraintsSameTrajectoryMetric = metrics::Gauge::Null();
static auto* kConstraintsDifferentTrajectoryMetric = metrics::Gauge::Null();
static auto* kActiveSubmapsMetric = metrics::Gauge::Null();
static auto* kFrozenSubmapsMetric = metrics::Gauge::Null();
static auto* kDeletedSubmapsMetric = metrics::Gauge::Null();

using kas_utils::fastErase;

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
      num_global_constraints_to_compute_(0.0),
      loops_trimmer_(options.log_number_of_trimmed_loops()) {}

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

void PoseGraph3D::DeleteTrajectoriesIfNeeded() {
  for (const auto& [trajectory_id, trajectory_state] : trajectory_states_) {
    if (trajectory_states_.IsTrajectoryReadyForDeletion(trajectory_id)) {
      std::vector<SubmapId> submaps_to_trim;
      for (const auto& submap_it : submap_data_.trajectory(trajectory_id)) {
        submaps_to_trim.emplace_back(submap_it.id);
      }
      for (const auto& submap : submaps_to_trim) {
        TrimSubmap(submap);
      }
      CHECK(submap_data_.SizeOfTrajectoryOrZero(trajectory_id) == 0);
      CHECK(trajectory_data_.count(trajectory_id) == 0);
      trajectory_states_.DeleteTrajectory(trajectory_id);
      maps_.DeleteTrajectory(trajectory_id);
    }
  }
}

void PoseGraph3D::AddImuData(
    int trajectory_id,
    const sensor::ImuData& imu_data) {
  AddWorkItem([=]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    CHECK(trajectory_states_.CanModifyTrajectory(trajectory_id));
    imu_data_.Append(trajectory_id, imu_data);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddOdometryData(
    int trajectory_id,
    const sensor::OdometryData& odometry_data) {
  AddWorkItem([=]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    CHECK(trajectory_states_.CanModifyTrajectory(trajectory_id));
    odometry_data_.Append(trajectory_id, odometry_data);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddFixedFramePoseData(
    int trajectory_id,
    const sensor::FixedFramePoseData& fixed_frame_pose_data) {
  AddWorkItem([=]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    CHECK(trajectory_states_.CanModifyTrajectory(trajectory_id));
    fixed_frame_pose_data_.Append(trajectory_id, fixed_frame_pose_data);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddLandmarkData(
    int trajectory_id,
    const sensor::LandmarkData& landmark_data) {
  AddWorkItem([=]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    CHECK(trajectory_states_.CanModifyTrajectory(trajectory_id));
    for (const auto& observation : landmark_data.landmark_observations) {
      landmark_nodes_[observation.id].landmark_observations.emplace_back(
          PoseGraphInterface::LandmarkNode::LandmarkObservation{
              trajectory_id, landmark_data.time,
              observation.landmark_to_tracking_transform,
              observation.translation_weight, observation.rotation_weight});
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

std::pair<bool, bool> PoseGraph3D::CheckIfConstraintCanBeAdded(
    const NodeId& node_id, const SubmapId& submap_id) {
  bool local_constraint_can_be_added = false;
  bool global_constraint_can_be_added = false;

  if (node_id.trajectory_id != submap_id.trajectory_id &&
      !maps_.TrajectoriesBelongToTheSameMap(
          node_id.trajectory_id, submap_id.trajectory_id)) {
    return std::make_pair(false, false);
  }

  CHECK(submap_data_.at(submap_id).state == SubmapState::kFinished);
  if (!submap_data_.at(submap_id).submap->insertion_finished()) {
    // this happens when unfinished submap was loaded from file
    return std::make_pair(false, false);
  }

  common::Time latest_node_time = GetLatestNodeTime(node_id, submap_id);
  common::Time last_connection_time = trajectory_states_.LastConnectionTime(
      node_id.trajectory_id, submap_id.trajectory_id);
  bool connected = trajectory_states_.TransitivelyConnected(
      node_id.trajectory_id, submap_id.trajectory_id);
  common::Duration global_constraint_search_after_n_seconds =
      common::FromSeconds(options_.global_constraint_search_after_n_seconds());
  bool recently_connected = connected &&
      ((global_constraint_search_after_n_seconds < common::FromSeconds(0.)) ||
          (latest_node_time - last_connection_time < global_constraint_search_after_n_seconds));
  if (node_id.trajectory_id == submap_id.trajectory_id || recently_connected) {
    const transform::Rigid3d& global_node_pose =
        trajectory_nodes_.at(node_id).global_pose;
    const transform::Rigid3d& global_submap_pose =
        submap_data_.at(submap_id).global_pose;
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
PoseGraph3D::ComputeCandidatesForConstraints(const NodeId& node_id,
    const std::map<int, SubmapId>& last_connected_submap_ids) {
  bool pure_localization_trajectory =
      pure_localization_trajectory_ids_.count(node_id.trajectory_id);
  std::vector<SubmapId> local_candidates;
  std::vector<SubmapId> global_candidates;
  for (const auto& submap_id_data : submap_data_) {
    const SubmapId& submap_id = submap_id_data.id;
    if (last_connected_submap_ids.at(submap_id.trajectory_id) < submap_id) {
      continue;
    }
    if (pure_localization_trajectory && node_id.trajectory_id == submap_id.trajectory_id) {
      continue;
    }
    if (submap_id_data.data.node_ids.count(node_id)) {
      continue;
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
PoseGraph3D::ComputeCandidatesForConstraints(const SubmapId& submap_id,
    const std::map<int, NodeId>& last_connected_node_ids) {
  bool pure_localization_trajectory =
      pure_localization_trajectory_ids_.count(submap_id.trajectory_id);
  CHECK(!pure_localization_trajectory);

  const std::set<NodeId>& submap_node_ids = submap_data_.at(submap_id).node_ids;
  std::vector<NodeId> local_candidates;
  std::vector<NodeId> global_candidates;
  for (const auto& node_id_data : trajectory_nodes_) {
    const NodeId& node_id = node_id_data.id;
    if (last_connected_node_ids.at(node_id.trajectory_id) < node_id) {
      continue;
    }
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

std::vector<SubmapId>
PoseGraph3D::SelectCandidatesForConstraints(
    const std::vector<SubmapId>& candidates,
    double& num_constraints_to_compute,
    std::set<SubmapId>& submaps_used_for_constraints) {
  static std::mt19937 generator(123);

  // sort candidates before performing std::set_difference
  std::vector<SubmapId> candidates_sorted = candidates;
  std::sort(candidates_sorted.begin(), candidates_sorted.end());

  // remove used submaps from candidates
  std::vector<SubmapId> available_submaps;
  std::set_difference(
      candidates_sorted.begin(), candidates_sorted.end(),
      submaps_used_for_constraints.begin(), submaps_used_for_constraints.end(),
      std::back_inserter(available_submaps));

  // output submaps
  std::vector<SubmapId> submaps_for_constraints;
  bool used_all_available_submaps = false;
  while (num_constraints_to_compute >= 1.0) {
    if (available_submaps.size() == 0) {
      if (used_all_available_submaps) {
        // special case: we drew all the candidates
        num_constraints_to_compute = 1.0;
        CHECK(submaps_for_constraints.size() == candidates.size());
        return submaps_for_constraints;
      }
      // special case: we drew all submaps that weren't in the used submaps list.
      // Clear the used submaps list and draw submaps that were previously discarded.
      std::sort(submaps_for_constraints.begin(), submaps_for_constraints.end());
      std::set_difference(
          candidates_sorted.begin(), candidates_sorted.end(),
          submaps_for_constraints.begin(), submaps_for_constraints.end(),
          std::back_inserter(available_submaps));
      submaps_used_for_constraints.clear();
      used_all_available_submaps = true;
      continue;
    }
    // draw random submap
    int i = generator() % available_submaps.size();
    auto submap_it = available_submaps.begin() + i;
    submaps_for_constraints.emplace_back(*submap_it);
    submaps_used_for_constraints.emplace(*submap_it);
    available_submaps.erase(submap_it);
    num_constraints_to_compute -= 1.0;
  }
  return submaps_for_constraints;
}

std::pair<std::vector<SubmapId>, std::vector<SubmapId>>
PoseGraph3D::SelectCandidatesForConstraints(
    const std::vector<SubmapId>& local_candidates,
    const std::vector<SubmapId>& global_candidates) {
  std::vector<SubmapId> submaps_for_local_constraints =
      SelectCandidatesForConstraints(
        local_candidates, num_local_constraints_to_compute_,
        submaps_used_for_local_constraints_);
  std::vector<SubmapId> submaps_for_global_constraints =
      SelectCandidatesForConstraints(
        global_candidates, num_global_constraints_to_compute_,
        submaps_used_for_global_constraints_);
  return std::make_pair(
      std::move(submaps_for_local_constraints),
      std::move(submaps_for_global_constraints));
}

std::vector<NodeId>
PoseGraph3D::SelectCandidatesForConstraints(
    const std::vector<NodeId>& candidates,
    double& num_constraints_to_compute) {
  std::vector<NodeId> nodes_for_constraints;
  int num_constraints_to_compute_int = std::floor(num_constraints_to_compute);
  if (num_constraints_to_compute_int > (int)candidates.size()) {
    num_constraints_to_compute_int = candidates.size();
  }
  // draw nodes with equal steps
  for (int n = 0; n < num_constraints_to_compute_int; n++) {
    int i = std::lround(1.0 * n / num_constraints_to_compute_int * candidates.size());
    nodes_for_constraints.emplace_back(candidates[i]);
  }
  num_constraints_to_compute -= std::floor(num_constraints_to_compute);
  return nodes_for_constraints;
}

std::pair<std::vector<NodeId>, std::vector<NodeId>>
PoseGraph3D::SelectCandidatesForConstraints(
    const std::vector<NodeId>& local_candidates,
    const std::vector<NodeId>& global_candidates) {
  std::vector<NodeId> nodes_for_local_constraints =
      SelectCandidatesForConstraints(
          local_candidates, num_local_constraints_to_compute_);
  std::vector<NodeId> nodes_for_global_constraints =
      SelectCandidatesForConstraints(
          global_candidates, num_global_constraints_to_compute_);
  return std::make_pair(
      std::move(nodes_for_local_constraints),
      std::move(nodes_for_global_constraints));
}

void PoseGraph3D::MaybeAddConstraints(const NodeId& node_id,
    const std::vector<SubmapId>& local_submap_ids,
    const std::vector<SubmapId>& global_submap_ids) {
  const TrajectoryNode& node_data = trajectory_nodes_.at(node_id);
  const transform::Rigid3d& global_node_pose = node_data.global_pose;
  const TrajectoryNode::Data* constant_data = node_data.constant_data.get();

  for (const SubmapId& submap_id : local_submap_ids) {
    const InternalSubmapData& submap_data = submap_data_.at(submap_id);
    const transform::Rigid3d& global_submap_pose = submap_data.global_pose;
    const Submap3D* submap = static_cast<const Submap3D*>(submap_data.submap.get());
    constraint_builder_.MaybeAddConstraint(
        submap_id, submap, node_id, constant_data,
        global_node_pose, global_submap_pose);
  }

  for (const SubmapId& submap_id : global_submap_ids) {
    const InternalSubmapData& submap_data = submap_data_.at(submap_id);
    const transform::Rigid3d& global_submap_pose = submap_data.global_pose;
    const Submap3D* submap = static_cast<const Submap3D*>(submap_data.submap.get());
    constraint_builder_.MaybeAddGlobalConstraint(
        submap_id, submap, node_id, constant_data,
        global_node_pose.rotation(), global_submap_pose.rotation());
  }
}

void PoseGraph3D::MaybeAddConstraints(const SubmapId& submap_id,
      const std::vector<NodeId>& local_node_ids,
      const std::vector<NodeId>& global_node_ids) {
  const InternalSubmapData& submap_data = submap_data_.at(submap_id);
  const transform::Rigid3d& global_submap_pose = submap_data.global_pose;
  const Submap3D* submap = static_cast<const Submap3D*>(submap_data.submap.get());

  for (const NodeId& node_id : local_node_ids) {
    const TrajectoryNode& node_data = trajectory_nodes_.at(node_id);
    const transform::Rigid3d& global_node_pose = node_data.global_pose;
    const TrajectoryNode::Data* constant_data = node_data.constant_data.get();
    constraint_builder_.MaybeAddConstraint(
        submap_id, submap, node_id, constant_data,
        global_node_pose, global_submap_pose);
  }

  for (const NodeId& node_id : global_node_ids) {
    const TrajectoryNode& node_data = trajectory_nodes_.at(node_id);
    const transform::Rigid3d& global_node_pose = node_data.global_pose;
    const TrajectoryNode::Data* constant_data = node_data.constant_data.get();
    constraint_builder_.MaybeAddGlobalConstraint(
        submap_id, submap, node_id, constant_data,
        global_node_pose.rotation(), global_submap_pose.rotation());
  }
}

void PoseGraph3D::ConnectNodeToGraph(
    const NodeId& node_id,
    const std::vector<SubmapId>& submap_ids,
    bool new_submap_added,
    bool old_submap_finished) {
  absl::MutexLock locker(&mutex_);
  CHECK(trajectory_states_.CanModifyTrajectory(node_id.trajectory_id));
  CHECK(submap_ids.size() == 1 || submap_ids.size() == 2);
  int trajectory_id = node_id.trajectory_id;
    
  const auto& constant_data = trajectory_nodes_.at(node_id).constant_data;
  std::vector<transform::Rigid3d> submap_local_poses;
  submap_local_poses.reserve(submap_ids.size());
  for (const SubmapId& submap_id : submap_ids) {
    submap_local_poses.push_back(submap_data_.at(submap_id).submap->local_pose());
  }
  const transform::Rigid3d& front_submap_global_pose =
      submap_data_.at(submap_ids.front()).global_pose;

  if (new_submap_added) {
    if (submap_ids.size() == 1) {
      if (initial_trajectory_poses_.count(trajectory_id) > 0) {
        trajectory_states_.Connect(trajectory_id,
            initial_trajectory_poses_.at(trajectory_id).to_trajectory_id,
            constant_data->time);
      }
      optimization_problem_->InsertSubmap(submap_ids.front(),
          ComputeGlobalToLocalTransform(submap_data_, trajectory_id) *
              submap_local_poses.front());
    } else {
      optimization_problem_->InsertSubmap(submap_ids.back(),
          front_submap_global_pose *
          submap_local_poses.front().inverse() *
          submap_local_poses.back());
    }
  }

  const transform::Rigid3d& local_pose = constant_data->local_pose;
  const transform::Rigid3d global_pose = front_submap_global_pose *
      submap_local_poses.front().inverse() * local_pose;
  trajectory_data_[submap_ids.front().trajectory_id];
  optimization_problem_->InsertTrajectoryNode(
      node_id, constant_data->time, local_pose, global_pose);

  for (size_t i = 0; i < submap_ids.size(); ++i) {
    const SubmapId submap_id = submap_ids[i];
    CHECK(submap_data_.at(submap_id).state == SubmapState::kNoConstraintSearch);
    submap_data_.at(submap_id).node_ids.emplace(node_id);
    trajectory_nodes_.at(node_id).submap_ids.emplace_back(submap_id);
    const transform::Rigid3d constraint_transform =
        submap_local_poses[i].inverse() * local_pose;
    constraints_.push_back(Constraint{
        submap_id,
        node_id,
        {constraint_transform, options_.matcher_translation_weight(),
            options_.matcher_rotation_weight()},
        Constraint::INTRA_SUBMAP});
  }

  if (old_submap_finished) {
    const SubmapId old_submap_finished_id = submap_ids.front();
    InternalSubmapData& finished_submap_data =
        submap_data_.at(old_submap_finished_id);
    CHECK(finished_submap_data.state == SubmapState::kNoConstraintSearch);
    finished_submap_data.state = SubmapState::kFinished;
  }
}

WorkItem::Result PoseGraph3D::ComputeConstraintsForNode(
    const NodeId& node_id,
    const std::vector<SubmapId>& submap_ids,
    bool old_submap_finished) {
  absl::MutexLock locker(&mutex_);
  std::map<int, SubmapId> last_connected_submap_ids;
  std::map<int, NodeId> last_connected_node_ids;
  std::tie(last_connected_submap_ids, last_connected_node_ids) = optimization_problem_->GetLastIds();
  for (int trajectory_id : submap_data_.trajectory_ids()) {
    if (last_connected_submap_ids.count(trajectory_id) == 0) {
      last_connected_submap_ids.emplace(trajectory_id, SubmapId{trajectory_id, -1});
    }
  }
  for (int trajectory_id : trajectory_nodes_.trajectory_ids()) {
    if (last_connected_node_ids.count(trajectory_id) == 0) {
      last_connected_node_ids.emplace(trajectory_id, NodeId{trajectory_id, -1});
    }
  }

  MEASURE_TIME_FROM_HERE(schedule_constraints_computation_for_node);
  std::vector<SubmapId> submap_candidates_for_local_constraints;
  std::vector<SubmapId> submap_candidates_for_global_constraints;
  std::tie(submap_candidates_for_local_constraints, submap_candidates_for_global_constraints) =
      ComputeCandidatesForConstraints(node_id, last_connected_submap_ids);
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
  STOP_TIME_MEASUREMENT(schedule_constraints_computation_for_node);

  bool pure_localization_trajectory =
      pure_localization_trajectory_ids_.count(node_id.trajectory_id);
  if (old_submap_finished && !pure_localization_trajectory) {
    MEASURE_TIME_FROM_HERE(schedule_constraints_computation_for_submap);
    const SubmapId finished_submap_id = submap_ids.front();
    int finished_submap_num_nodes = submap_data_.at(finished_submap_id).node_ids.size();
    std::vector<NodeId> node_candidates_for_local_constraints;
    std::vector<NodeId> node_candidates_for_global_constraints;
    std::tie(node_candidates_for_local_constraints, node_candidates_for_global_constraints) =
        ComputeCandidatesForConstraints(finished_submap_id, last_connected_node_ids);
    num_local_constraints_to_compute_ +=
        options_.local_constraints_per_node() * finished_submap_num_nodes / 2;
    num_global_constraints_to_compute_ +=
        options_.global_constraints_per_node() * finished_submap_num_nodes / 2;
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
    MaybeAddConstraints(finished_submap_id, nodes_for_local_constraints, nodes_for_global_constraints);
    STOP_TIME_MEASUREMENT(schedule_constraints_computation_for_submap);
  }

  constraint_builder_.NotifyEndOfNode();
  ++num_nodes_since_last_loop_closure_;
  if (options_.optimize_every_n_nodes() > 0 &&
      num_nodes_since_last_loop_closure_ > options_.optimize_every_n_nodes()) {
    return WorkItem::Result::kRunOptimization;
  }
  return WorkItem::Result::kDoNotRunOptimization;
}

std::tuple<NodeId, std::vector<SubmapId>, bool> PoseGraph3D::AppendNode(
    std::shared_ptr<const TrajectoryNode::Data> constant_data,
    int trajectory_id,
    const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps) {
  absl::MutexLock locker(&mutex_);
  if (!trajectory_states_.ContainsTrajectory(trajectory_id)) {
    trajectory_states_.AddTrajectory(trajectory_id);
  }

  const NodeId node_id = trajectory_nodes_.Append(
      trajectory_id, TrajectoryNode{constant_data,
          ComputeGlobalToLocalTransform(submap_data_, trajectory_id) *
              constant_data->local_pose});
  trajectory_nodes_.at(node_id).submap_ids.reserve(2);
  ++num_trajectory_nodes_;

  SubmapId last_submap_id(-1, -1);
  bool new_submap_added;
  if (submap_data_.SizeOfTrajectoryOrZero(trajectory_id) == 0 ||
      std::prev(submap_data_.EndOfTrajectory(trajectory_id))->data.submap !=
          insertion_submaps.back()) {
    transform::Rigid3d global_pose =
        ComputeGlobalToLocalTransform(submap_data_, trajectory_id) *
            insertion_submaps.back()->local_pose();
    last_submap_id = submap_data_.Append(trajectory_id, InternalSubmapData());
    submap_data_.at(last_submap_id).submap = insertion_submaps.back();
    submap_data_.at(last_submap_id).global_pose = global_pose;
    loops_trimmer_.AddSubmap(last_submap_id, node_id);
    new_submap_added = true;
    kActiveSubmapsMetric->Increment();
  } else {
    last_submap_id =  std::prev(submap_data_.EndOfTrajectory(trajectory_id))->id;
    new_submap_added = false;
  }

  std::vector<SubmapId> submap_ids;
  if (submap_data_.SizeOfTrajectoryOrZero(trajectory_id) == 1) {
    submap_ids = {last_submap_id};
  } else {
    submap_ids = {
        SubmapId(last_submap_id.trajectory_id, last_submap_id.submap_index - 1),
        last_submap_id};
  }

  loops_trimmer_.AddNode(node_id, constant_data->accum_rotation, constant_data->travelled_distance);
  if (!maps_.ContainsTrajectory(trajectory_id)) {
    maps_.AddTrajectory("default", trajectory_id);
  }
  return std::make_tuple(node_id, submap_ids, new_submap_added);
}

NodeId PoseGraph3D::AddNode(
    std::shared_ptr<const TrajectoryNode::Data> constant_data, int trajectory_id,
    const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps) {
  const auto [node_id, submap_ids, new_submap_added] = AppendNode(
      constant_data, trajectory_id, insertion_submaps);
  const bool old_submap_finished = insertion_submaps.front()->insertion_finished();

  AddWorkItem([=]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    ConnectNodeToGraph(node_id, submap_ids, new_submap_added, old_submap_finished);
    return ComputeConstraintsForNode(node_id, submap_ids, old_submap_finished);
  });
  return node_id;
}

common::Time PoseGraph3D::GetLatestNodeTime(
    const NodeId& node_id, const SubmapId& submap_id) const {
  common::Time time = trajectory_nodes_.at(node_id).constant_data->time;
  const InternalSubmapData& submap_data = submap_data_.at(submap_id);
  if (!submap_data.node_ids.empty()) {
    const NodeId last_submap_node_id =
        *submap_data_.at(submap_id).node_ids.rbegin();
    time = std::max(
        time,
        trajectory_nodes_.at(last_submap_node_id).constant_data->time);
  }
  return time;
}

void PoseGraph3D::UpdateTrajectoryConnectivity(const Constraint& constraint) {
  CHECK_EQ(constraint.tag, Constraint::INTER_SUBMAP);
  const common::Time time =
      GetLatestNodeTime(constraint.node_id, constraint.submap_id);
  trajectory_states_.Connect(
      constraint.node_id.trajectory_id, constraint.submap_id.trajectory_id,
      time);
}

void PoseGraph3D::TrimSubmap(const SubmapId& submap_id) {
  // TODO(hrapp): We have to make sure that the trajectory has been finished
  // if we want to delete the last submaps.
  CHECK(submap_data_.Contains(submap_id));
  CHECK(submap_data_.at(submap_id).state == SubmapState::kFinished);

  for (const NodeId& node_id : submap_data_.at(submap_id).node_ids) {
    std::vector<SubmapId>& submap_ids_for_node =
        trajectory_nodes_.at(node_id).submap_ids;
    auto it = std::find(submap_ids_for_node.begin(),
        submap_ids_for_node.end(), submap_id);
    CHECK(it != submap_ids_for_node.end());
    submap_ids_for_node.erase(it);
  }

  std::set<NodeId> nodes_to_retain;
  for (const auto& submap_data : submap_data_) {
    if (submap_data.id != submap_id) {
      nodes_to_retain.insert(submap_data.data.node_ids.begin(),
                             submap_data.data.node_ids.end());
    }
  }

  std::set<NodeId> nodes_to_remove;
  std::set_difference(submap_data_.at(submap_id).node_ids.begin(),
                      submap_data_.at(submap_id).node_ids.end(),
                      nodes_to_retain.begin(), nodes_to_retain.end(),
                      std::inserter(nodes_to_remove, nodes_to_remove.begin()));

  std::set<SubmapId> other_submap_ids_losing_constraints;
  for (auto constraint_it = constraints_.begin(); constraint_it != constraints_.end();) {
    const Constraint& constraint = *constraint_it;
    if (constraint.submap_id == submap_id || nodes_to_remove.count(constraint.node_id)) {
      if (constraint.tag == Constraint::INTER_SUBMAP) {
        if (constraint.submap_id != submap_id) {
          other_submap_ids_losing_constraints.insert(constraint.submap_id);
        }
        loops_trimmer_.RemoveLoop(constraint.submap_id, constraint.node_id);
      }
      constraint_it = fastErase(constraints_, constraint_it);
      continue;
    }
    ++constraint_it;
  }

  for (const Constraint& constraint : constraints_) {
    if (constraint.tag == Constraint::Tag::INTER_SUBMAP) {
      other_submap_ids_losing_constraints.erase(constraint.submap_id);
    }
  }

  for (const SubmapId& submap_id : other_submap_ids_losing_constraints) {
    constraint_builder_.DeleteScanMatcher(submap_id);
  }

  submap_data_.Trim(submap_id);
  constraint_builder_.DeleteScanMatcher(submap_id);
  optimization_problem_->TrimSubmap(submap_id);

  for (const NodeId& node_id : nodes_to_remove) {
    imu_data_.Trim(trajectory_nodes_, node_id);
    odometry_data_.Trim(trajectory_nodes_, node_id);
    fixed_frame_pose_data_.Trim(trajectory_nodes_, node_id);
    trajectory_nodes_.Trim(node_id);
    if (trajectory_nodes_.SizeOfTrajectoryOrZero(node_id.trajectory_id) == 0) {
      trajectory_data_.erase(node_id.trajectory_id);
    }
    optimization_problem_->TrimTrajectoryNode(node_id);
  }

  kDeletedSubmapsMetric->Increment();
  if (trajectory_states_.IsTrajectoryFrozen(submap_id.trajectory_id)) {
    kFrozenSubmapsMetric->Decrement();
  } else {
    kActiveSubmapsMetric->Decrement();
  }
}

void PoseGraph3D::ReattachLoop(Constraint& loop, const NodeId& to_node_id) {
  CHECK(loop.tag == Constraint::INTER_SUBMAP);
  const transform::Rigid3d& node_pose = trajectory_nodes_.at(loop.node_id).global_pose;
  const transform::Rigid3d& to_node_pose = trajectory_nodes_.at(to_node_id).global_pose;
  transform::Rigid3d correction = node_pose.inverse() * to_node_pose;
  loop.node_id = to_node_id;
  loop.pose.zbar_ij = loop.pose.zbar_ij * correction;
}

void PoseGraph3D::ReattachLoop(Constraint& loop, const SubmapId& to_submap_id) {
  CHECK(loop.tag == Constraint::INTER_SUBMAP);
  const transform::Rigid3d& submap_pose = submap_data_.at(loop.submap_id).global_pose;
  const transform::Rigid3d& to_submap_pose = submap_data_.at(to_submap_id).global_pose;
  transform::Rigid3d correction = to_submap_pose.inverse() * submap_pose;
  loop.submap_id = to_submap_id;
  loop.pose.zbar_ij = correction * loop.pose.zbar_ij;
}

void PoseGraph3D::TrimNode(const NodeId& node_id) {
  MEASURE_BLOCK_TIME(TrimNode);

  CHECK(trajectory_nodes_.Contains(node_id));
  for (const SubmapId& submap_id : trajectory_nodes_.at(node_id).submap_ids) {
    CHECK(submap_data_.at(submap_id).state == SubmapState::kFinished);
  }

  std::vector<SubmapId> submap_ids;
  for (const auto& submap_data : submap_data_) {
    auto it = submap_data.data.node_ids.find(node_id);
    if (it != submap_data.data.node_ids.end()) {
      if (submap_data.data.node_ids.size() == 1) {
        submap_ids.push_back(submap_data.id);
      } else {
        submap_data_.at(submap_data.id).node_ids.erase(it);
      }
    }
  }
  CHECK(submap_ids.size() <= 2);

  std::vector<Constraint> new_constraints;
  for (auto constraint_it = constraints_.begin(); constraint_it != constraints_.end();) {
    const Constraint& constraint = *constraint_it;
    if (constraint.node_id == node_id ||
        std::find(submap_ids.begin(), submap_ids.end(), constraint.submap_id) != submap_ids.end()) {
      if (constraint.tag == Constraint::INTER_SUBMAP) {
        Constraint new_constraint = constraint;
        bool drop_loop = false;
        if (constraint.node_id == node_id) {
          auto it = trajectory_nodes_.find(new_constraint.node_id);
          CHECK(it != trajectory_nodes_.end());
          if (it != trajectory_nodes_.BeginOfTrajectory(it->id.trajectory_id)) {
            ReattachLoop(new_constraint, std::prev(it)->id);
          } else if (trajectory_nodes_.SizeOfTrajectoryOrZero(it->id.trajectory_id) >= 2) {
            ReattachLoop(new_constraint, std::next(it)->id);
          } else {
            drop_loop = true;
          }
        }
        if (std::find(submap_ids.begin(), submap_ids.end(), constraint.submap_id) != submap_ids.end()) {
          auto it = submap_data_.find(constraint.submap_id);
          CHECK(it != submap_data_.end());
          int back_dist = 0;
          int forth_dist = 0;
          auto back_it = it;
          auto forth_it = it;
          while (std::find(submap_ids.begin(), submap_ids.end(), back_it->id) != submap_ids.end()) {
            if (back_it == submap_data_.BeginOfTrajectory(back_it->id.trajectory_id)) {
              back_dist = std::numeric_limits<int>::max();
              break;
            }
            --back_it;
            back_dist++;
          }
          while (std::find(submap_ids.begin(), submap_ids.end(), forth_it->id) != submap_ids.end()) {
            ++forth_it;
            forth_dist++;
            if (forth_it == submap_data_.EndOfTrajectory(forth_it->id.trajectory_id)) {
              forth_dist = std::numeric_limits<int>::max();
              break;
            }
          }
          CHECK(back_dist != 0);
          CHECK(forth_dist != 0);
          if (back_dist == std::numeric_limits<int>::max() && forth_dist == std::numeric_limits<int>::max()) {
            drop_loop = true;
          } else {
            if (back_dist < forth_dist) {
              ReattachLoop(new_constraint, back_it->id);
            } else {
              ReattachLoop(new_constraint, forth_it->id);
            }
          }
        }

        loops_trimmer_.RemoveLoop(constraint.submap_id, constraint.node_id);
        if (!drop_loop) {
          *constraint_it = new_constraint;
          loops_trimmer_.AddLoop(new_constraint);
          ++constraint_it;
          continue;
        }
      }
      constraint_it = fastErase(constraints_, constraint_it);
      continue;
    }
    ++constraint_it;
  }

  std::set<std::pair<SubmapId, NodeId>> connections;
  for (auto constraint_it = constraints_.begin(); constraint_it != constraints_.end();) {
    const Constraint& constraint = *constraint_it;
    if (constraint.tag == Constraint::INTER_SUBMAP) {
      std::pair<SubmapId, NodeId> connection(constraint.submap_id, constraint.node_id);
      if (connections.count(connection)) {
        loops_trimmer_.RemoveLoop(constraint.submap_id, constraint.node_id);
        constraint_it = fastErase(constraints_, constraint_it);
        continue;
      } else {
        connections.insert(connection);
      }
    }
    ++constraint_it;
  }

  imu_data_.Trim(trajectory_nodes_, node_id);
  odometry_data_.Trim(trajectory_nodes_, node_id);
  fixed_frame_pose_data_.Trim(trajectory_nodes_, node_id);
  trajectory_nodes_.Trim(node_id);
  if (trajectory_nodes_.SizeOfTrajectoryOrZero(node_id.trajectory_id) == 0) {
    trajectory_data_.erase(node_id.trajectory_id);
  }
  optimization_problem_->TrimTrajectoryNode(node_id);
  for (const SubmapId& submap_id : submap_ids) {
    submap_data_.Trim(submap_id);
    constraint_builder_.DeleteScanMatcher(submap_id);
    optimization_problem_->TrimSubmap(submap_id);
  }

  for (const Constraint& constraint : constraints_) {
    CHECK(trajectory_nodes_.Contains(constraint.node_id));
    CHECK(submap_data_.Contains(constraint.submap_id));
  }
}

void PoseGraph3D::TrimPureLocalizationTrajectories() {
  for (auto& [trajectory_id, options] : pure_localization_trimmer_options_) {
    if (!trajectory_states_.ContainsTrajectory(trajectory_id)) {
      continue;
    }
    if (trajectory_states_.IsTrajectoryDeleted(trajectory_id)) {
      continue;
    }
    CHECK(!trajectory_states_.IsTrajectoryReadyForDeletion(trajectory_id));
    if (trajectory_states_.IsTrajectoryFinished(trajectory_id)) {
      options.set_max_submaps_to_keep(0);
    }
    int num_submaps = optimization_problem_->NumSubmapsOrZero(trajectory_id);
    int i = 0;
    std::vector<SubmapId> submaps_to_trim;
    for (const auto& it : submap_data_.trajectory(trajectory_id)) {
      if (i + options.max_submaps_to_keep() >= num_submaps) {
        break;
      }
      submaps_to_trim.push_back(it.id);
      i++;
    }
    for (const auto& submap_id : submaps_to_trim) {
      TrimSubmap(submap_id);
    }
    if (options.max_submaps_to_keep() == 0) {
      if (trajectory_states_.IsTrajectoryTransitionNone(trajectory_id)) {
        trajectory_states_.ScheduleTrajectoryForDeletion(trajectory_id);
      }
      if (trajectory_states_.IsTrajectoryScheduledForDeletion(trajectory_id)) {
        trajectory_states_.PrepareTrajectoryForDeletion(trajectory_id);
      }
      trajectory_states_.DeleteTrajectory(trajectory_id);
    }
  }
  for (auto it = pure_localization_trimmer_options_.begin();
      it != pure_localization_trimmer_options_.end();) {
    if (it->second.max_submaps_to_keep() == 0) {
      it = pure_localization_trimmer_options_.erase(it);
    } else {
      ++it;
    }
  }
}

void PoseGraph3D::TrimScheduledNodes() {
  for (const NodeId& node_to_trim : nodes_scheduled_to_trim_) {
    if (!trajectory_nodes_.Contains(node_to_trim)) {
      continue;
    }

    if (trajectory_states_.IsTrajectoryFrozen(node_to_trim.trajectory_id)) {
      continue;
    }

    bool submaps_for_node_are_not_finished = false;
    for (const SubmapId& submap_id : trajectory_nodes_.at(node_to_trim).submap_ids) {
      if (submap_data_.at(submap_id).state == SubmapState::kNoConstraintSearch) {
        submaps_for_node_are_not_finished = true;
        break;
      }
    }
    if (submaps_for_node_are_not_finished) {
      continue;
    }

    TrimNode(node_to_trim);
  }
  nodes_scheduled_to_trim_.clear();
}

void PoseGraph3D::HandleWorkQueue(
    const constraints::ConstraintBuilder3D::Result& result) {
  if (options_.log_constraints()) {
    absl::MutexLock locker(&mutex_);
    for (const Constraint& constraint : result) {
      bool local, global;
      std::tie(local, global) =
          CheckIfConstraintCanBeAdded(constraint.node_id, constraint.submap_id);
      CHECK(!(local && global));
      std::ostringstream info;
      if (global) {
        info << "Global. ";
      }
      CHECK(trajectory_nodes_.at(constraint.node_id).submap_ids.size());
      SubmapId submap_id_for_node(
          trajectory_nodes_.at(constraint.node_id).submap_ids.back());
      info << "Node from " << submap_id_for_node <<
          ", submap " << constraint.submap_id <<
          ", score " << std::setprecision(3) << constraint.score;
      LOG(INFO) << info.str();
    }
  }

  constraints::ConstraintBuilder3D::Result true_detected_loops;
  {
    absl::MutexLock locker(&mutex_);
    loops_trimmer_.TrimCloseLoops(constraints_);
    true_detected_loops = loops_trimmer_.TrimFalseDetectedLoops(
        result, submap_data_, trajectory_nodes_);
    constraints_.insert(constraints_.end(),
        true_detected_loops.begin(), true_detected_loops.end());
    loops_trimmer_.AddLoops(true_detected_loops);
    DeleteTrajectoriesIfNeeded();
  }

  MEASURE_TIME_FROM_HERE(optimization);
  RunOptimization();
  STOP_TIME_MEASUREMENT(optimization);

  {
    absl::MutexLock locker(&mutex_);
    for (const Constraint& constraint : true_detected_loops) {
      UpdateTrajectoryConnectivity(constraint);
    }

    TrimPureLocalizationTrajectories();
    TrimScheduledNodes();
    num_nodes_since_last_loop_closure_ = 0;

    // Update the gauges that count the current number of constraints.
    double inter_constraints_same_trajectory = 0;
    double inter_constraints_different_trajectory = 0;
    for (const auto& constraint : constraints_) {
      if (constraint.tag ==
          cartographer::mapping::Constraint::INTRA_SUBMAP) {
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

  {
    absl::MutexLock locker(&callback_mutex_);
    if (global_slam_optimization_callback_) {
      std::map<int, SubmapId> last_connected_submap_ids;
      std::map<int, NodeId> last_connected_node_ids;
      std::tie(last_connected_submap_ids, last_connected_node_ids) = optimization_problem_->GetLastIds();
      global_slam_optimization_callback_(last_connected_submap_ids, last_connected_node_ids);
    }
  }
}

static void work_item_processing_latency_print_summary(
    const std::string& name, const std::vector<double>& latencies) {
  if (latencies.size()) {
    double accum_latency = 0.0;
    double max_latency = 0.0;
    for (double latency : latencies) {
      accum_latency += latency;
      max_latency = std::max(max_latency, latency);
    }
    double average_latency = accum_latency / latencies.size();

    std::string log_string;
    log_string += name + ":\n";
    log_string += "    Average latency: " + std::to_string(average_latency) + "\n";
    log_string += "    Max latency: " + std::to_string(max_latency) + "\n";

    std::cout << log_string;
  }
}

static void max_queue_size_print_summary(
    const std::string& name, const std::vector<long unsigned int>& max_queue_sizes) {
  std::string log_string;
  log_string += name + ":\n";
  log_string += "    Queue size: ";
  for (long unsigned int max_queue_size : max_queue_sizes) {
    log_string += std::to_string(max_queue_size) + " ";
  }
  log_string += "\n";

  std::cout << log_string;
}

void PoseGraph3D::DrainWorkQueue() {
  bool process_work_queue = true;
  size_t work_queue_size;

  static auto last_log_time = std::chrono::steady_clock::now();
  static long unsigned int max_queue_size = 0;
  {
    absl::MutexLock locker(&work_queue_mutex_);
    max_queue_size = std::max(max_queue_size, work_queue_->size());
  }
  auto now_time = std::chrono::steady_clock::now();
  if (common::ToSeconds(now_time - last_log_time) > 3) {
    static kas_utils::Collection<long unsigned int> max_queue_size_col(
        "max_queue_size", max_queue_size_print_summary, nullptr, nullptr);
    max_queue_size_col.add(max_queue_size);

    last_log_time = now_time;
    if (options_.log_work_queue_size()) {
      LOG(INFO) << "Work items in queue: " << max_queue_size;
    }
    max_queue_size = 0;
  }

  static kas_utils::Collection<double> work_item_processing_latency_col(
      "work_item_processing_latency", work_item_processing_latency_print_summary,
      nullptr, nullptr);
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
      work_item_processing_latency_col.add(common::ToSeconds(now_time - add_time));
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
  bool problem_is_built;
  {
    absl::MutexLock locker(&mutex_);
    if (optimization_problem_->NumSubmaps() == 0) {
      return;
    }
    problem_is_built = optimization_problem_->BuildProblem(
        constraints_, imu_data_, odometry_data_, fixed_frame_pose_data_,
        trajectory_states_, trajectory_data_, landmark_nodes_);
  }

  if (problem_is_built) {
    optimization_problem_->Solve();
  }

  absl::MutexLock locker(&mutex_);
  const MapById<SubmapId, transform::Rigid3d> new_submap_poses = optimization_problem_->GetSubmapPoses();
  const MapById<NodeId, transform::Rigid3d> new_node_poses = optimization_problem_->GetNodePoses();
  for (const int trajectory_id : new_node_poses.trajectory_ids()) {
    for (const auto& new_node : new_node_poses.trajectory(trajectory_id)) {
      trajectory_nodes_.at(new_node.id).global_pose = new_node.data;
    }

    const transform::Rigid3d old_global_to_local = ComputeGlobalToLocalTransform(
        submap_data_, trajectory_id);
    const transform::Rigid3d new_global_to_local = ComputeGlobalToLocalTransform(
        new_submap_poses, trajectory_id);
    const transform::Rigid3d correction_in_global =
        new_global_to_local * old_global_to_local.inverse();

    const NodeId last_new_node_id =
        std::prev(new_node_poses.EndOfTrajectory(trajectory_id))->id;
    auto node_it = std::next(trajectory_nodes_.find(last_new_node_id));
    for (; node_it != trajectory_nodes_.EndOfTrajectory(trajectory_id); ++node_it) {
      auto& trajectory_node = trajectory_nodes_.at(node_it->id);
      trajectory_node.global_pose = correction_in_global * trajectory_node.global_pose;
    }
  }

  for (const int trajectory_id : new_submap_poses.trajectory_ids()) {
    for (const auto& new_submap : new_submap_poses.trajectory(trajectory_id)) {
      submap_data_.at(new_submap.id).global_pose = new_submap.data;
    }

    const transform::Rigid3d old_global_to_local = ComputeGlobalToLocalTransform(
        submap_data_, trajectory_id);
    const transform::Rigid3d new_global_to_local = ComputeGlobalToLocalTransform(
        new_submap_poses, trajectory_id);
    const transform::Rigid3d correction_in_global =
        new_global_to_local * old_global_to_local.inverse();

    const SubmapId last_new_submap_id =
        std::prev(new_submap_poses.EndOfTrajectory(trajectory_id))->id;
    auto submap_it = std::next(submap_data_.find(last_new_submap_id));
    for (; submap_it != submap_data_.EndOfTrajectory(trajectory_id); ++submap_it) {
      auto& submap_data = submap_data_.at(submap_it->id);
      submap_data.global_pose = correction_in_global * submap_data.global_pose;
    }
  }

  for (const auto& landmark : optimization_problem_->GetLandmarkData()) {
    landmark_nodes_[landmark.first].global_landmark_pose = landmark.second;
  }
  trajectory_data_ = optimization_problem_->GetTrajectoryData();

  // Log the histograms for the pose residuals.
  if (options_.log_residual_histograms()) {
    LogResidualHistograms();
  }
}

void PoseGraph3D::ScheduleFalseConstraintsTrimming(
    double max_rotation_error, double max_translation_error) {
  AddWorkItem([this, max_rotation_error, max_translation_error]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    loops_trimmer_.TrimFalseLoops(constraints_,
        max_rotation_error, max_translation_error,
        submap_data_, trajectory_nodes_);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::RunFinalOptimization() {
  AddWorkItem([this]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    optimization_problem_->SetMaxNumIterations(
        options_.max_num_final_iterations());
    return WorkItem::Result::kRunOptimization;
  });
  AddWorkItem([this]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    optimization_problem_->SetMaxNumIterations(
        options_.optimization_problem_options()
            .ceres_solver_options().max_num_iterations());
    return WorkItem::Result::kDoNotRunOptimization;
  });
  WaitForQueue();
}

void PoseGraph3D::WaitForAllComputations() {
  int num_trajectory_nodes;
  {
    absl::MutexLock locker(&mutex_);
    num_trajectory_nodes = num_trajectory_nodes_;
  }

  const int num_finished_nodes_at_start = constraint_builder_.GetNumFinishedNodes();
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
            constraints_.insert(constraints_.end(),
                result.begin(), result.end());
            loops_trimmer_.AddLoops(result);
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

void PoseGraph3D::WaitForQueue() {
  bool work_queue_reset;
  {
    absl::MutexLock locker(&work_queue_mutex_);
    work_queue_reset = (work_queue_ == nullptr);
  }
  while (!work_queue_reset) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    {
      absl::MutexLock locker(&work_queue_mutex_);
      work_queue_reset = (work_queue_ == nullptr);
    }
  }
}

void PoseGraph3D::DeleteTrajectory(int trajectory_id) {
  {
    absl::MutexLock locker(&mutex_);
    trajectory_states_.ScheduleTrajectoryForDeletion(trajectory_id);
  }

  AddWorkItem([this, trajectory_id]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    trajectory_states_.PrepareTrajectoryForDeletion(trajectory_id);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::FinishTrajectory(int trajectory_id) {
  {
    absl::MutexLock locker(&mutex_);
    trajectory_states_.ScheduleTrajectoryForFinish(trajectory_id);
  }

  AddWorkItem([this, trajectory_id]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    trajectory_states_.FinishTrajectory(trajectory_id);
    for (const auto& submap : submap_data_.trajectory(trajectory_id)) {
      submap_data_.at(submap.id).state = SubmapState::kFinished;
    }
    return WorkItem::Result::kRunOptimization;
  });
}

bool PoseGraph3D::IsTrajectoryFinished(int trajectory_id) const {
  absl::MutexLock locker(&mutex_);
  return trajectory_states_.IsTrajectoryFinished(trajectory_id);
}

void PoseGraph3D::FreezeTrajectory(int trajectory_id) {
  {
    absl::MutexLock locker(&mutex_);
    trajectory_states_.ScheduleTrajectoryForFreezing(trajectory_id);
  }

  AddWorkItem([this, trajectory_id]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    for (const auto& [other_trajectory_id, other_trajectory_state] : trajectory_states_) {
      if (!trajectory_states_.IsTrajectoryFrozen(other_trajectory_id)) {
        continue;
      }
      if (trajectory_states_.TransitivelyConnected(trajectory_id, other_trajectory_id)) {
        continue;
      }
      trajectory_states_.Connect(
          trajectory_id, other_trajectory_id, common::FromUniversal(0));
    }
    trajectory_states_.FreezeTrajectory(trajectory_id);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

bool PoseGraph3D::IsTrajectoryFrozen(int trajectory_id) const {
  absl::MutexLock locker(&mutex_);
  return trajectory_states_.IsTrajectoryFrozen(trajectory_id);
}

void PoseGraph3D::ScheduleNodesToTrim(const std::set<common::Time>& nodes_to_trim) {
  AddWorkItem([this, nodes_to_trim]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    for (const auto& node : trajectory_nodes_) {
      if (nodes_to_trim.count(node.data.constant_data->time)) {
        nodes_scheduled_to_trim_.insert(node.id);
      }
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::ScheduleNodesToTrim(const std::set<NodeId>& nodes_to_trim) {
  AddWorkItem([this, nodes_to_trim]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    std::set<NodeId> merged;
    std::set_union(
      nodes_scheduled_to_trim_.begin(), nodes_scheduled_to_trim_.end(),
      nodes_to_trim.begin(), nodes_to_trim.end(),
      std::inserter(merged, merged.end()));
    nodes_scheduled_to_trim_ = std::move(merged);
    return WorkItem::Result::kDoNotRunOptimization;
  });
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
    if (!trajectory_states_.ContainsTrajectory(submap_id.trajectory_id)) {
      trajectory_states_.AddTrajectory(submap_id.trajectory_id);
    }
    if (!maps_.ContainsTrajectory(submap_id.trajectory_id)) {
      maps_.AddTrajectory("default", submap_id.trajectory_id);
    }
    submap_data_.Insert(submap_id, InternalSubmapData());
    submap_data_.at(submap_id).submap = submap_ptr;
    submap_data_.at(submap_id).global_pose = global_submap_pose;
    if (trajectory_states_.IsTrajectoryFrozen(submap_id.trajectory_id)) {
      kFrozenSubmapsMetric->Increment();
    } else {
      kActiveSubmapsMetric->Increment();
    }
  }

  AddWorkItem([this, submap_id, global_submap_pose]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    submap_data_.at(submap_id).state = SubmapState::kFinished;
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
    if (!trajectory_states_.ContainsTrajectory(node_id.trajectory_id)) {
      trajectory_states_.AddTrajectory(node_id.trajectory_id);
    }
    if (!maps_.ContainsTrajectory(node_id.trajectory_id)) {
      maps_.AddTrajectory("default", node_id.trajectory_id);
    }
    trajectory_nodes_.Insert(node_id,
        TrajectoryNode{constant_data, global_pose});
    trajectory_nodes_.at(node_id).submap_ids.reserve(2);
    loops_trimmer_.AddNode(node_id, constant_data->accum_rotation, constant_data->travelled_distance);
  }

  AddWorkItem([this, node_id, global_pose]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    const auto& constant_data = trajectory_nodes_.at(node_id).constant_data;
    trajectory_data_[node_id.trajectory_id];
    optimization_problem_->InsertTrajectoryNode(
        node_id, constant_data->time, constant_data->local_pose, global_pose);
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
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    trajectory_data_[trajectory_id] = trajectory_data;
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::SetLandmarkPose(
    const std::string& landmark_id,
    const transform::Rigid3d& global_pose,
    bool frozen) {
  AddWorkItem([=]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    landmark_nodes_[landmark_id].global_landmark_pose = global_pose;
    landmark_nodes_[landmark_id].frozen = frozen;
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddSerializedConstraints(
    const std::vector<Constraint>& constraints) {
  AddWorkItem([this, constraints]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    for (const Constraint& constraint : constraints) {
      CHECK(trajectory_nodes_.Contains(constraint.node_id));
      CHECK(submap_data_.Contains(constraint.submap_id));
      CHECK(trajectory_nodes_.at(constraint.node_id).constant_data !=
            nullptr);
      CHECK(submap_data_.at(constraint.submap_id).submap != nullptr);
      switch (constraint.tag) {
        case Constraint::Tag::INTRA_SUBMAP:
          {
            bool emplaced = submap_data_.at(constraint.submap_id)
                .node_ids.emplace(constraint.node_id).second;
            CHECK(emplaced);
            std::vector<SubmapId>& submap_ids_for_node =
                trajectory_nodes_.at(constraint.node_id).submap_ids;
            submap_ids_for_node.emplace_back(constraint.submap_id);
            std::sort(submap_ids_for_node.begin(), submap_ids_for_node.end());
          }
          break;
        case Constraint::Tag::INTER_SUBMAP:
          UpdateTrajectoryConnectivity(constraint);
          break;
      }
      constraints_.push_back(constraint);
    }
    for (const auto& [submap_id, submap_data] : submap_data_) {
      CHECK(submap_data.node_ids.size());
      loops_trimmer_.AddSubmap(submap_id, *submap_data.node_ids.begin());
    }
    for (const Constraint& constraint : constraints_) {
      if (constraint.tag == Constraint::INTER_SUBMAP) {
        loops_trimmer_.AddLoop(constraint);
      }
    }
    LOG(INFO) << "Loaded " << constraints.size() << " constraints.";
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddSerializedMaps(
    const std::map<std::string, std::set<int>>& maps_data) {
  AddWorkItem([this, maps_data]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    maps_.UpdateData(maps_data);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddPureLocalizationTrimmer(int trajectory_id,
    const proto::PureLocalizationTrimmerOptions& pure_localization_trimmer_options) {
  AddWorkItem([this, trajectory_id, pure_localization_trimmer_options]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    pure_localization_trajectory_ids_.insert(trajectory_id);
    pure_localization_trimmer_options_.emplace(
        trajectory_id, pure_localization_trimmer_options);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddLoopTrimmer(int trajectory_id,
    const proto::LoopTrimmerOptions& loop_trimmer_options) {
  AddWorkItem([this, trajectory_id, loop_trimmer_options]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    loops_trimmer_.AddLoopsTrimmerOptions(trajectory_id, loop_trimmer_options);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::SetInitialTrajectoryPose(
    int from_trajectory_id, int to_trajectory_id,
    const transform::Rigid3d& pose, common::Time time) {
  absl::MutexLock locker(&mutex_);
  initial_trajectory_poses_[from_trajectory_id] =
      InitialTrajectoryPose{to_trajectory_id, pose, time};
}

transform::Rigid3d PoseGraph3D::GetInterpolatedGlobalTrajectoryPose(
    int trajectory_id, const common::Time time) const {
  CHECK_GT(trajectory_nodes_.SizeOfTrajectoryOrZero(trajectory_id), 0);
  const auto it = trajectory_nodes_.lower_bound(trajectory_id, time);
  if (it == trajectory_nodes_.BeginOfTrajectory(trajectory_id)) {
    return trajectory_nodes_.BeginOfTrajectory(trajectory_id)
        ->data.global_pose;
  }
  if (it == trajectory_nodes_.EndOfTrajectory(trajectory_id)) {
    return std::prev(trajectory_nodes_.EndOfTrajectory(trajectory_id))
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

transform::Rigid3d PoseGraph3D::ComputeGlobalToLocalTransform(
    const MapById<SubmapId, InternalSubmapData>& global_submap_poses,
    int trajectory_id) const {
  auto begin_it = global_submap_poses.BeginOfTrajectory(trajectory_id);
  auto end_it = global_submap_poses.EndOfTrajectory(trajectory_id);
  if (begin_it == end_it) {
    const auto it = initial_trajectory_poses_.find(trajectory_id);
    if (it != initial_trajectory_poses_.end()) {
      return GetInterpolatedGlobalTrajectoryPose(it->second.to_trajectory_id,
                                                 it->second.time) *
             it->second.relative_pose;
    } else {
      return transform::Rigid3d::Identity();
    }
  }
  const SubmapId last_submap_id = std::prev(end_it)->id;
  return global_submap_poses.at(last_submap_id).global_pose *
      submap_data_.at(last_submap_id).submap->local_pose().inverse();
}

transform::Rigid3d PoseGraph3D::ComputeGlobalToLocalTransform(
    const MapById<SubmapId, transform::Rigid3d>& global_submap_poses,
    int trajectory_id) const {
  auto begin_it = global_submap_poses.BeginOfTrajectory(trajectory_id);
  auto end_it = global_submap_poses.EndOfTrajectory(trajectory_id);
  if (begin_it == end_it) {
    const auto it = initial_trajectory_poses_.find(trajectory_id);
    if (it != initial_trajectory_poses_.end()) {
      return GetInterpolatedGlobalTrajectoryPose(it->second.to_trajectory_id,
                                                 it->second.time) *
             it->second.relative_pose;
    } else {
      return transform::Rigid3d::Identity();
    }
  }
  const SubmapId last_submap_id = std::prev(end_it)->id;
  return global_submap_poses.at(last_submap_id) *
      submap_data_.at(last_submap_id).submap->local_pose().inverse();
}

std::vector<std::vector<int>> PoseGraph3D::GetConnectedTrajectories() const {
  absl::MutexLock locker(&mutex_);
  return trajectory_states_.Components();
}

bool PoseGraph3D::TrajectoriesTransitivelyConnected(
    int trajectory_a, int trajectory_b) const {
  absl::MutexLock locker(&mutex_);
  return trajectory_states_.TransitivelyConnected(trajectory_a, trajectory_b);
}

common::Time PoseGraph3D::TrajectoriesLastConnectionTime(
    int trajectory_a, int trajectory_b) const {
  absl::MutexLock locker(&mutex_);
  return trajectory_states_.LastConnectionTime(trajectory_a, trajectory_b);
}

bool PoseGraph3D::TrajectoriesBelongToTheSameMap(
    int trajectory_a, int trajectory_b) const {
  absl::MutexLock locker(&mutex_);
  return maps_.TrajectoriesBelongToTheSameMap(trajectory_a, trajectory_b);
}

transform::Rigid3d PoseGraph3D::GetGlobalToLocalTransform(
    int trajectory_id) const {
  absl::MutexLock locker(&mutex_);
  return ComputeGlobalToLocalTransform(submap_data_, trajectory_id);
}

void PoseGraph3D::MoveTrajectoryToMap(int trajectory_id, const std::string& map_name) {
  AddWorkItem([this, trajectory_id, map_name]()
        ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    if (trajectory_nodes_.SizeOfTrajectoryOrZero(trajectory_id) &&
        maps_.GetMapName(trajectory_id) != "default") {
    }
    maps_.MoveTrajectory(trajectory_id, map_name);
    return WorkItem::Result::kDoNotRunOptimization;
  });
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
  for (const auto& submap_id_data : submap_data_) {
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
  const auto it = submap_data_.find(submap_id);
  if (it == submap_data_.end()) {
    return {};
  }
  return {it->data.submap, it->data.global_pose};
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph3D::GetSubmapDataUnderLock() const {
  MapById<SubmapId, PoseGraphInterface::SubmapData> submaps;
  for (const auto& submap_id_data : submap_data_) {
    submaps.Insert(
        submap_id_data.id,
        GetSubmapDataUnderLock(submap_id_data.id));
  }
  return submaps;
}

MapById<NodeId, TrajectoryNode> PoseGraph3D::GetTrajectoryNodes() const {
  absl::MutexLock locker(&mutex_);
  return trajectory_nodes_;
}

MapById<NodeId, TrajectoryNodePose> PoseGraph3D::GetTrajectoryNodePoses() const {
  MapById<NodeId, TrajectoryNodePose> node_poses;
  absl::MutexLock locker(&mutex_);
  for (const auto& node_id_data : trajectory_nodes_) {
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

std::map<int, TrajectoryState> PoseGraph3D::GetTrajectoryStates() const {
  std::map<int, TrajectoryState> trajectory_states;
  absl::MutexLock locker(&mutex_);
  for (const auto& [trajectory_id, trajectory_state] : trajectory_states_) {
    trajectory_states[trajectory_id] = trajectory_state;
  }
  return trajectory_states;
}

std::map<std::string, transform::Rigid3d> PoseGraph3D::GetLandmarkPoses() const {
  std::map<std::string, transform::Rigid3d> landmark_poses;
  absl::MutexLock locker(&mutex_);
  for (const auto& landmark : landmark_nodes_) {
    // Landmark without value has not been optimized yet.
    if (!landmark.second.global_landmark_pose.has_value()) continue;
    landmark_poses[landmark.first] =
        landmark.second.global_landmark_pose.value();
  }
  return landmark_poses;
}

sensor::MapByTime<sensor::ImuData> PoseGraph3D::GetImuData() const {
  absl::MutexLock locker(&mutex_);
  return imu_data_;
}

sensor::MapByTime<sensor::OdometryData> PoseGraph3D::GetOdometryData() const {
  absl::MutexLock locker(&mutex_);
  return odometry_data_;
}

sensor::MapByTime<sensor::FixedFramePoseData>
PoseGraph3D::GetFixedFramePoseData() const {
  absl::MutexLock locker(&mutex_);
  return fixed_frame_pose_data_;
}

std::map<std::string, PoseGraphInterface::LandmarkNode>
PoseGraph3D::GetLandmarkNodes() const {
  absl::MutexLock locker(&mutex_);
  return landmark_nodes_;
}

std::map<int, PoseGraphInterface::TrajectoryData>
PoseGraph3D::GetTrajectoryData() const {
  absl::MutexLock locker(&mutex_);
  return trajectory_data_;
}

std::map<std::string, std::set<int>> PoseGraph3D::GetMapsData() const {
  absl::MutexLock locker(&mutex_);
  return maps_.GetData();
}

std::vector<Constraint> PoseGraph3D::constraints() const {
  absl::MutexLock locker(&mutex_);
  return constraints_;
}

void PoseGraph3D::SetGlobalSlamOptimizationCallback(
    PoseGraphInterface::GlobalSlamOptimizationCallback callback) {
  AddWorkItem([this, callback = std::move(callback)]()
        ABSL_LOCKS_EXCLUDED(callback_mutex_) {
    absl::MutexLock locker(&callback_mutex_);
    global_slam_optimization_callback_ = std::move(callback);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::LogResidualHistograms() const {
  common::Histogram rotational_residual;
  common::Histogram translational_residual;
  for (const Constraint& constraint : constraints_) {
    if (constraint.tag == Constraint::Tag::INTRA_SUBMAP) {
      const cartographer::transform::Rigid3d optimized_node_to_map =
          trajectory_nodes_.at(constraint.node_id).global_pose;
      const cartographer::transform::Rigid3d node_to_submap_constraint =
          constraint.pose.zbar_ij;
      const cartographer::transform::Rigid3d optimized_submap_to_map =
          submap_data_.at(constraint.submap_id).global_pose;
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

}  // namespace mapping
}  // namespace cartographer
