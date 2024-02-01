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

#include "cartographer/mapping/internal/optimization/optimization_problem_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/memory/memory.h"
#include "cartographer/common/internal/ceres_solver_options.h"
#include "cartographer/common/math.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/internal/3d/imu_integration.h"
#include "cartographer/mapping/internal/3d/rotation_parameterization.h"
#include "cartographer/mapping/internal/optimization/cost_functions/acceleration_cost_function_3d.h"
#include "cartographer/mapping/internal/optimization/cost_functions/landmark_cost_function_3d.h"
#include "cartographer/mapping/internal/optimization/cost_functions/rotation_cost_function_3d.h"
#include "cartographer/mapping/internal/optimization/cost_functions/spa_cost_function_3d.h"
#include "cartographer/transform/timestamped_transform.h"
#include "cartographer/transform/transform.h"
#include "ceres/ceres.h"
#include "ceres/jet.h"
#include "ceres/rotation.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace optimization {
namespace {

using LandmarkNode = ::cartographer::mapping::PoseGraphInterface::LandmarkNode;
using TrajectoryData =
    ::cartographer::mapping::PoseGraphInterface::TrajectoryData;

// For odometry.
std::unique_ptr<transform::Rigid3d> Interpolate(
    const sensor::MapByTime<sensor::OdometryData>& map_by_time,
    const int trajectory_id, const common::Time time) {
  const auto it = map_by_time.lower_bound(trajectory_id, time);
  if (it == map_by_time.EndOfTrajectory(trajectory_id)) {
    return nullptr;
  }
  if (it == map_by_time.BeginOfTrajectory(trajectory_id)) {
    if (it->time == time) {
      return absl::make_unique<transform::Rigid3d>(it->pose);
    }
    return nullptr;
  }
  const auto prev_it = std::prev(it);
  return absl::make_unique<transform::Rigid3d>(
      Interpolate(transform::TimestampedTransform{prev_it->time, prev_it->pose},
                  transform::TimestampedTransform{it->time, it->pose}, time)
          .transform);
}

// For fixed frame pose.
std::unique_ptr<transform::Rigid3d> Interpolate(
    const sensor::MapByTime<sensor::FixedFramePoseData>& map_by_time,
    const int trajectory_id, const common::Time time) {
  const auto it = map_by_time.lower_bound(trajectory_id, time);
  if (it == map_by_time.EndOfTrajectory(trajectory_id) ||
      !it->pose.has_value()) {
    return nullptr;
  }
  if (it == map_by_time.BeginOfTrajectory(trajectory_id)) {
    if (it->time == time) {
      return absl::make_unique<transform::Rigid3d>(it->pose.value());
    }
    return nullptr;
  }
  const auto prev_it = std::prev(it);
  if (prev_it->pose.has_value()) {
    return absl::make_unique<transform::Rigid3d>(
        Interpolate(transform::TimestampedTransform{prev_it->time,
                                                    prev_it->pose.value()},
                    transform::TimestampedTransform{it->time, it->pose.value()},
                    time)
            .transform);
  }
  return nullptr;
}

// Selects a trajectory node closest in time to the landmark observation and
// applies a relative transform from it.
transform::Rigid3d GetInitialLandmarkPose(
    const LandmarkNode::LandmarkObservation& observation,
    const NodeData& prev_node, const NodeData& next_node,
    const CeresPose& C_prev_node_pose, const CeresPose& C_next_node_pose) {
  const double interpolation_parameter =
      common::ToSeconds(observation.time - prev_node.time) /
      common::ToSeconds(next_node.time - prev_node.time);

  const std::tuple<std::array<double, 4>, std::array<double, 3>>
      rotation_and_translation = InterpolateNodes3D(
          C_prev_node_pose.rotation(), C_prev_node_pose.translation(),
          C_next_node_pose.rotation(), C_next_node_pose.translation(),
          interpolation_parameter);
  return transform::Rigid3d::FromArrays(std::get<0>(rotation_and_translation),
                                        std::get<1>(rotation_and_translation)) *
         observation.landmark_to_tracking_transform;
}

void AddLandmarkCostFunctions(
    const std::map<std::string, LandmarkNode>& landmark_nodes,
    const MapById<NodeId, NodeData>& node_data,
    const std::set<int>& frozen_trajectories,
    std::map<NodeId, CeresPose>* C_node_poses,
    std::map<std::string, CeresPose>* C_landmark_poses, ceres::Problem* problem,
    double huber_scale) {
  for (const auto& landmark_node : landmark_nodes) {
    // Do not use landmarks that were not optimized for localization.
    for (const auto& observation : landmark_node.second.landmark_observations) {
      int trajectory_id = observation.trajectory_id;
      if (frozen_trajectories.count(trajectory_id) != 0) {
        continue;
      }
      const std::string& landmark_id = landmark_node.first;
      const auto& begin_of_trajectory = node_data.BeginOfTrajectory(trajectory_id);
      // The landmark observation was made before the trajectory was created.
      if (observation.time < begin_of_trajectory->data.time) {
        continue;
      }
      // Find the trajectory nodes before and after the landmark observation.
      auto next = node_data.lower_bound(trajectory_id, observation.time);
      // The landmark observation was made, but the next trajectory node has
      // not been added yet.
      if (next == node_data.EndOfTrajectory(trajectory_id)) {
        continue;
      }
      if (next == begin_of_trajectory) {
        next = std::next(next);
      }
      auto prev = std::prev(next);
      // Add parameter blocks for the landmark ID if they were not added before.
      CeresPose& C_prev_node_pose = C_node_poses->at(prev->id);
      CeresPose& C_next_node_pose = C_node_poses->at(next->id);
      if (!C_landmark_poses->count(landmark_id)) {
        transform::Rigid3d starting_point =
            landmark_node.second.global_landmark_pose.has_value()
                ? landmark_node.second.global_landmark_pose.value()
                : GetInitialLandmarkPose(observation, prev->data, next->data,
                    C_prev_node_pose, C_next_node_pose);
        C_landmark_poses->emplace(landmark_id, starting_point);
        problem->AddParameterBlock(C_landmark_poses->at(landmark_id).rotation(), 4, new ceres::QuaternionParameterization());
        problem->AddParameterBlock(C_landmark_poses->at(landmark_id).translation(), 3, nullptr);

        // Set landmark constant if it is frozen.
        if (landmark_node.second.frozen) {
          problem->SetParameterBlockConstant(C_landmark_poses->at(landmark_id).rotation());
          problem->SetParameterBlockConstant(C_landmark_poses->at(landmark_id).translation());
        }
      }
      problem->AddResidualBlock(
          LandmarkCostFunction3D::CreateAutoDiffCostFunction(
              observation, prev->data, next->data),
          new ceres::HuberLoss(huber_scale),
          C_prev_node_pose.rotation(),
          C_prev_node_pose.translation(),
          C_next_node_pose.rotation(),
          C_next_node_pose.translation(),
          C_landmark_poses->at(landmark_id).rotation(),
          C_landmark_poses->at(landmark_id).translation());
    }
  }
}

std::unique_ptr<transform::Rigid3d> CalculateOdometryBetweenNodes(
    const sensor::MapByTime<sensor::OdometryData>& odometry_data,
    const int trajectory_id, const NodeData& first_node_data,
    const NodeData& second_node_data) {
  if (odometry_data.HasTrajectory(trajectory_id)) {
    const std::unique_ptr<transform::Rigid3d> first_node_odometry =
        Interpolate(odometry_data, trajectory_id, first_node_data.time);
    const std::unique_ptr<transform::Rigid3d> second_node_odometry =
        Interpolate(odometry_data, trajectory_id, second_node_data.time);
    if (first_node_odometry != nullptr && second_node_odometry != nullptr) {
      const transform::Rigid3d relative_odometry =
          first_node_odometry->inverse() * (*second_node_odometry);
      return std::make_unique<transform::Rigid3d>(relative_odometry);
    }
  }
  return nullptr;
}

}  // namespace

OptimizationProblem3D::OptimizationProblem3D(
    const optimization::proto::OptimizationProblemOptions& options)
    : options_(options) {}

OptimizationProblem3D::~OptimizationProblem3D() {}

void OptimizationProblem3D::InsertTrajectoryNode(const NodeId& node_id,
    const common::Time& time, const transform::Rigid3d& local_pose,
    const transform::Rigid3d& global_pose) {
  absl::MutexLock locker(&mutex_);
  const auto& [_, emplaced] = C_node_poses_.emplace(node_id, global_pose);
  node_data_.Insert(node_id, NodeData{time, local_pose});
  CHECK(emplaced);

  int trajectory_id = node_id.trajectory_id;
  if (num_nodes_.count(trajectory_id)) {
    num_nodes_.at(trajectory_id) += 1;
  } else {
    num_nodes_[trajectory_id] = 1;
  }
}

void OptimizationProblem3D::TrimTrajectoryNode(const NodeId& node_id) {
  absl::MutexLock locker(&mutex_);
  int num_erased = C_node_poses_.erase(node_id);
  node_data_.Trim(node_id);
  CHECK(num_erased == 1);

  num_nodes_.at(node_id.trajectory_id) -= 1;
}

void OptimizationProblem3D::InsertSubmap(const SubmapId& submap_id,
    const transform::Rigid3d& global_pose) {
  absl::MutexLock locker(&mutex_);
  const auto& [_, emplaced] = C_submap_poses_.emplace(submap_id, global_pose);
  CHECK(emplaced);

  int trajectory_id = submap_id.trajectory_id;
  if (num_submaps_.count(trajectory_id)) {
    num_submaps_.at(trajectory_id) += 1;
  } else {
    num_submaps_[trajectory_id] = 1;
  }
}

void OptimizationProblem3D::TrimSubmap(const SubmapId& submap_id) {
  absl::MutexLock locker(&mutex_);
  int num_erased = C_submap_poses_.erase(submap_id);
  CHECK(num_erased == 1);

  num_submaps_.at(submap_id.trajectory_id) -= 1;
}

void OptimizationProblem3D::SetMaxNumIterations(const int32 max_num_iterations) {
  absl::MutexLock locker(&mutex_);
  options_.mutable_ceres_solver_options()->set_max_num_iterations(max_num_iterations);
}

bool OptimizationProblem3D::BuildProblem(
    const std::vector<Constraint>& constraints,
    const sensor::MapByTime<sensor::ImuData>& imu_data,
    const sensor::MapByTime<sensor::OdometryData>& odometry_data,
    const sensor::MapByTime<sensor::FixedFramePoseData>& fixed_frame_pose_data,
    const PoseGraphTrajectoryStates& trajectory_states,
    const std::map<int, PoseGraphInterface::TrajectoryData>& trajectory_data,
    const std::map<std::string, LandmarkNode>& landmark_nodes) {
  absl::MutexLock locker(&mutex_);
  CHECK(problem_ == nullptr);
  if (C_node_poses_.empty()) {
    return false;
  }

  trajectory_data_ = trajectory_data;

  std::set<int> frozen_trajectories;
  for (const auto& [trajectory_id, trajectory_state] : trajectory_states) {
    if (trajectory_state.state == TrajectoryState::State::FROZEN) {
      frozen_trajectories.insert(trajectory_id);
    }
  }

  ceres::Problem::Options problem_options;
  problem_ = std::make_unique<ceres::Problem>(problem_options);

  const auto translation_parameterization =
      [this]() -> ceres::LocalParameterization* {
    return options_.fix_z_in_3d()
        ? new ceres::SubsetParameterization(3, std::vector<int>{2})
        : nullptr;
  };

  // Set the starting point.
  CHECK(!C_submap_poses_.empty());
  bool first_submap = true;
  for (auto& [submap_id, C_submap_pose] : C_submap_poses_) {
    bool frozen = frozen_trajectories.count(submap_id.trajectory_id);
    if (first_submap) {
      first_submap = false;
      // Fix the first submap of the first trajectory except for allowing
      // gravity alignment.
      problem_->AddParameterBlock(C_submap_pose.rotation(), 4, new ceres::AutoDiffLocalParameterization<ConstantYawQuaternionPlus, 4, 2>());
      problem_->AddParameterBlock(C_submap_pose.translation(), 3, translation_parameterization());
      problem_->SetParameterBlockConstant(C_submap_pose.translation());
    } else {
      problem_->AddParameterBlock(C_submap_pose.rotation(), 4, new ceres::QuaternionParameterization());
      problem_->AddParameterBlock(C_submap_pose.translation(), 3, translation_parameterization());
    }
    if (frozen) {
      problem_->SetParameterBlockConstant(C_submap_pose.rotation());
      problem_->SetParameterBlockConstant(C_submap_pose.translation());
    }
  }

  for (auto& [node_id, C_node_pose] : C_node_poses_) {
    bool frozen = frozen_trajectories.count(node_id.trajectory_id);
    problem_->AddParameterBlock(C_node_pose.rotation(), 4, new ceres::QuaternionParameterization());
    problem_->AddParameterBlock(C_node_pose.translation(), 3, translation_parameterization());
    if (frozen) {
      problem_->SetParameterBlockConstant(C_node_pose.rotation());
      problem_->SetParameterBlockConstant(C_node_pose.translation());
    }
  }

  // Add cost functions for intra- and inter-submap constraints.
  for (const Constraint& constraint : constraints) {
    if (constraint.node_id.trajectory_id == constraint.submap_id.trajectory_id &&
        frozen_trajectories.count(constraint.node_id.trajectory_id)) {
      continue;
    }
    problem_->AddResidualBlock(
        SpaCostFunction3D::CreateAutoDiffCostFunction(constraint.pose),
        // Loop closure constraints should have a loss function.
        constraint.tag == Constraint::INTER_SUBMAP
            ? new ceres::HuberLoss(options_.huber_scale())
            : nullptr /* loss function */,
        C_submap_poses_.at(constraint.submap_id).rotation(),
        C_submap_poses_.at(constraint.submap_id).translation(),
        C_node_poses_.at(constraint.node_id).rotation(),
        C_node_poses_.at(constraint.node_id).translation());
  }

  // Add cost functions for landmarks.
  C_landmark_poses_.clear();
  AddLandmarkCostFunctions(landmark_nodes, node_data_, frozen_trajectories,
      &C_node_poses_, &C_landmark_poses_, &(*problem_), options_.huber_scale());

  // Add constraints based on IMU observations of angular velocities and
  // linear acceleration.
  if (!options_.fix_z_in_3d()) {
    for (auto node_it = node_data_.begin(); node_it != node_data_.end();) {
      const int trajectory_id = node_it->id.trajectory_id;
      const auto trajectory_end = node_data_.EndOfTrajectory(trajectory_id);
      if (frozen_trajectories.count(trajectory_id) != 0) {
        // We skip frozen trajectories.
        node_it = trajectory_end;
        continue;
      }
      if (!imu_data.HasTrajectory(trajectory_id)) {
        // We skip trajectories with no imu data.
        node_it = trajectory_end;
        continue;
      }
      TrajectoryData& trajectory_data_trajectory = trajectory_data_.at(trajectory_id);

      problem_->AddParameterBlock(trajectory_data_trajectory.imu_calibration.data(), 4, new ceres::QuaternionParameterization());
      if (!options_.use_online_imu_extrinsics_in_3d()) {
        problem_->SetParameterBlockConstant(trajectory_data_trajectory.imu_calibration.data());
      }
      const auto imu_data_trajectory = imu_data.trajectory(trajectory_id);
      CHECK(imu_data_trajectory.begin() != imu_data_trajectory.end());

      auto imu_it = imu_data_trajectory.begin();
      auto prev_node_it = node_it;
      for (++node_it; node_it != trajectory_end; ++node_it) {
        const NodeId first_node_id = prev_node_it->id;
        const NodeData& first_node_data = prev_node_it->data;
        prev_node_it = node_it;
        const NodeId second_node_id = node_it->id;
        const NodeData& second_node_data = node_it->data;

        if (second_node_id.node_index != first_node_id.node_index + 1) {
          continue;
        }

        // Skip IMU data before the node.
        while (std::next(imu_it) != imu_data_trajectory.end() &&
               std::next(imu_it)->time <= first_node_data.time) {
          ++imu_it;
        }

        auto imu_it2 = imu_it;
        const IntegrateImuResult<double> result = IntegrateImu(
            imu_data_trajectory, first_node_data.time, second_node_data.time, &imu_it);
        const auto next_node_it = std::next(node_it);
        const common::Time first_time = first_node_data.time;
        const common::Time second_time = second_node_data.time;
        const common::Duration first_duration = second_time - first_time;
        if (next_node_it != trajectory_end &&
            next_node_it->id.node_index == second_node_id.node_index + 1) {
          const NodeId third_node_id = next_node_it->id;
          const NodeData& third_node_data = next_node_it->data;
          const common::Time third_time = third_node_data.time;
          const common::Duration second_duration = third_time - second_time;
          const common::Time first_center = first_time + first_duration / 2;
          const common::Time second_center = second_time + second_duration / 2;
          const IntegrateImuResult<double> result_to_first_center =
              IntegrateImu(imu_data_trajectory, first_time, first_center, &imu_it2);
          const IntegrateImuResult<double> result_center_to_center =
              IntegrateImu(imu_data_trajectory, first_center, second_center, &imu_it2);
          // 'delta_velocity' is the change in velocity from the point in time
          // halfway between the first and second poses to halfway between
          // second and third pose. It is computed from IMU data and still
          // contains a delta due to gravity. The orientation of this vector is
          // in the IMU frame at the second pose.
          const Eigen::Vector3d delta_velocity =
              (result.delta_rotation.inverse() *
               result_to_first_center.delta_rotation) *
              result_center_to_center.delta_velocity;
          problem_->AddResidualBlock(
              AccelerationCostFunction3D::CreateAutoDiffCostFunction(
                  options_.acceleration_weight() /
                      common::ToSeconds(first_duration + second_duration),
                  delta_velocity, common::ToSeconds(first_duration),
                  common::ToSeconds(second_duration)),
              nullptr /* loss function */,
              C_node_poses_.at(second_node_id).rotation(),
              C_node_poses_.at(first_node_id).translation(),
              C_node_poses_.at(second_node_id).translation(),
              C_node_poses_.at(third_node_id).translation(),
              &trajectory_data_trajectory.gravity_constant,
              trajectory_data_trajectory.imu_calibration.data());
        }
        problem_->AddResidualBlock(
            RotationCostFunction3D::CreateAutoDiffCostFunction(
                options_.rotation_weight() / common::ToSeconds(first_duration),
                result.delta_rotation),
            nullptr /* loss function */,
            C_node_poses_.at(first_node_id).rotation(),
            C_node_poses_.at(second_node_id).rotation(),
            trajectory_data_trajectory.imu_calibration.data());
      }

      // Force gravity constant to be positive.
      problem_->SetParameterLowerBound(&trajectory_data_trajectory.gravity_constant, 0, 0.0);
    }
  }

  if (options_.fix_z_in_3d() || options_.add_local_slam_consecutive_node_constraints_in_3d() ||
      options_.add_odometry_consecutive_node_constraints_in_3d()) {
    // Add penalties for violating odometry (if available) and changes between
    // consecutive nodes.
    for (auto node_it = node_data_.begin(); node_it != node_data_.end();) {
      const int trajectory_id = node_it->id.trajectory_id;
      const auto trajectory_end = node_data_.EndOfTrajectory(trajectory_id);
      if (frozen_trajectories.count(trajectory_id) != 0) {
        node_it = trajectory_end;
        continue;
      }

      auto prev_node_it = node_it;
      for (++node_it; node_it != trajectory_end; ++node_it) {
        const NodeId first_node_id = prev_node_it->id;
        const NodeData& first_node_data = prev_node_it->data;
        prev_node_it = node_it;
        const NodeId second_node_id = node_it->id;
        const NodeData& second_node_data = node_it->data;

        if (second_node_id.node_index != first_node_id.node_index + 1) {
          continue;
        }

        if (options_.fix_z_in_3d() || options_.add_odometry_consecutive_node_constraints_in_3d()) {
          // Add a relative pose constraint based on the odometry (if available).
          const std::unique_ptr<transform::Rigid3d> relative_odometry =
              CalculateOdometryBetweenNodes(odometry_data,
                  trajectory_id, first_node_data, second_node_data);
          if (relative_odometry != nullptr) {
            problem_->AddResidualBlock(
                SpaCostFunction3D::CreateAutoDiffCostFunction(Constraint::Pose{
                    *relative_odometry, options_.odometry_translation_weight(),
                    options_.odometry_rotation_weight()}),
                nullptr /* loss function */,
                C_node_poses_.at(first_node_id).rotation(),
                C_node_poses_.at(first_node_id).translation(),
                C_node_poses_.at(second_node_id).rotation(),
                C_node_poses_.at(second_node_id).translation());
          }
        }

        if (options_.fix_z_in_3d() || options_.add_local_slam_consecutive_node_constraints_in_3d()) {
          // Add a relative pose constraint based on consecutive local SLAM poses.
          const transform::Rigid3d relative_local_slam_pose =
              first_node_data.local_pose.inverse() * second_node_data.local_pose;
          problem_->AddResidualBlock(
              SpaCostFunction3D::CreateAutoDiffCostFunction(
                  Constraint::Pose{relative_local_slam_pose,
                      options_.local_slam_pose_translation_weight(),
                      options_.local_slam_pose_rotation_weight()}),
              nullptr /* loss function */,
              C_node_poses_.at(first_node_id).rotation(),
              C_node_poses_.at(first_node_id).translation(),
              C_node_poses_.at(second_node_id).rotation(),
              C_node_poses_.at(second_node_id).translation());
        }
      }
    }
  }

  // Add fixed frame pose constraints.
  C_fixed_frame_poses_.clear();
  for (auto node_it = node_data_.begin(); node_it != node_data_.end();) {
    const int trajectory_id = node_it->id.trajectory_id;
    const auto trajectory_end = node_data_.EndOfTrajectory(trajectory_id);
    if (frozen_trajectories.count(trajectory_id)) {
      node_it = trajectory_end;
      continue;
    }
    if (!fixed_frame_pose_data.HasTrajectory(trajectory_id)) {
      node_it = trajectory_end;
      continue;
    }

    const TrajectoryData& trajectory_data_trajectory = trajectory_data_.at(trajectory_id);
    bool fixed_frame_pose_initialized = false;
    for (; node_it != trajectory_end; ++node_it) {
      const NodeId node_id = node_it->id;
      const NodeData& node_data = node_it->data;

      const std::unique_ptr<transform::Rigid3d> fixed_frame_pose =
          Interpolate(fixed_frame_pose_data, trajectory_id, node_data.time);
      if (fixed_frame_pose == nullptr) {
        continue;
      }

      const Constraint::Pose constraint_pose{
          *fixed_frame_pose, options_.fixed_frame_pose_translation_weight(),
          options_.fixed_frame_pose_rotation_weight()};

      if (!fixed_frame_pose_initialized) {
        transform::Rigid3d fixed_frame_pose_in_map;
        if (trajectory_data_trajectory.fixed_frame_origin_in_map.has_value()) {
          fixed_frame_pose_in_map = trajectory_data_trajectory.fixed_frame_origin_in_map.value();
        } else {
          fixed_frame_pose_in_map =
              C_node_poses_.at(node_id).ToRigid() * constraint_pose.zbar_ij.inverse();
        }
        fixed_frame_pose_in_map = transform::Rigid3d(
            fixed_frame_pose_in_map.translation(),
            Eigen::AngleAxisd(
                transform::GetYaw(fixed_frame_pose_in_map.rotation()),
                Eigen::Vector3d::UnitZ()));
        C_fixed_frame_poses_.emplace(trajectory_id, fixed_frame_pose_in_map);
        problem_->AddParameterBlock(C_fixed_frame_poses_.at(trajectory_id).rotation(), 4, new ceres::AutoDiffLocalParameterization<YawOnlyQuaternionPlus, 4, 1>());
        problem_->AddParameterBlock(C_fixed_frame_poses_.at(trajectory_id).translation(), 3, nullptr);
        fixed_frame_pose_initialized = true;
      }

      problem_->AddResidualBlock(
          SpaCostFunction3D::CreateAutoDiffCostFunction(constraint_pose),
          options_.fixed_frame_pose_use_tolerant_loss() ?
              new ceres::TolerantLoss(
                  options_.fixed_frame_pose_tolerant_loss_param_a(),
                  options_.fixed_frame_pose_tolerant_loss_param_b()) :
              nullptr,
          C_fixed_frame_poses_.at(trajectory_id).rotation(),
          C_fixed_frame_poses_.at(trajectory_id).translation(),
          C_node_poses_.at(node_id).rotation(),
          C_node_poses_.at(node_id).translation());
    }
  }

  return true;
}

void OptimizationProblem3D::Solve() {
  absl::MutexLock locker(&mutex_);
  CHECK(problem_);
  ceres::Solver::Summary summary;
  ceres::Solve(
      common::CreateCeresSolverOptions(options_.ceres_solver_options()),
      &(*problem_), &summary);

  if (options_.log_solver_summary()) {
    LOG(INFO) << summary.FullReport();
    for (const auto& trajectory_id_and_data : trajectory_data_) {
      const int trajectory_id = trajectory_id_and_data.first;
      const TrajectoryData& trajectory_data_trajectory = trajectory_id_and_data.second;
      if (trajectory_id != 0) {
        LOG(INFO) << "Trajectory " << trajectory_id << ":";
      }
      LOG(INFO) << "Gravity was: " << trajectory_data_trajectory.gravity_constant;
      const auto& imu_calibration = trajectory_data_trajectory.imu_calibration;
      LOG(INFO) << "IMU correction was: "
                << common::RadToDeg(2. *
                                    std::acos(std::abs(imu_calibration[0])))
                << " deg (" << imu_calibration[0] << ", " << imu_calibration[1]
                << ", " << imu_calibration[2] << ", " << imu_calibration[3]
                << ")";
    }
  }

  for (const auto& [landmark_id, C_landmark_pose] : C_landmark_poses_) {
    landmark_data_[landmark_id] = C_landmark_pose.ToRigid();
  }
  for (const auto& [fixed_frame_id, C_fixed_frame_pose] : C_fixed_frame_poses_) {
    trajectory_data_.at(fixed_frame_id).fixed_frame_origin_in_map =
        C_fixed_frame_pose.ToRigid();
  }

  problem_.reset();
}

MapById<SubmapId, transform::Rigid3d> OptimizationProblem3D::GetSubmapPoses() const {
  MapById<SubmapId, transform::Rigid3d> submap_poses;
  absl::MutexLock locker(&mutex_);
  for (const auto& [submap_id, C_submap_pose] : C_submap_poses_) {
    submap_poses.Insert(submap_id, C_submap_pose.ToRigid());
  }
  return submap_poses;
}

MapById<NodeId, transform::Rigid3d> OptimizationProblem3D::GetNodePoses() const {
  MapById<NodeId, transform::Rigid3d> node_poses;
  absl::MutexLock locker(&mutex_);
  for (const auto& [node_id, C_node_pose] : C_node_poses_) {
    node_poses.Insert(node_id, C_node_pose.ToRigid());
  }
  return node_poses;
}

std::map<int, PoseGraphInterface::TrajectoryData> OptimizationProblem3D::GetTrajectoryData() const {
  absl::MutexLock locker(&mutex_);
  return trajectory_data_;
}

std::map<std::string, transform::Rigid3d> OptimizationProblem3D::GetLandmarkData() const {
  absl::MutexLock locker(&mutex_);
  return landmark_data_;
}

std::optional<SubmapId> OptimizationProblem3D::GetLastSubmapId(int trajectory_id) const {
  absl::MutexLock locker(&mutex_);
  return GetLastSubmapIdUnderLock(trajectory_id);
}

std::optional<NodeId> OptimizationProblem3D::GetLastNodeId(int trajectory_id) const {
  absl::MutexLock locker(&mutex_);
  return GetLastNodeIdUnderLock(trajectory_id);
}

std::optional<SubmapId> OptimizationProblem3D::GetLastSubmapIdUnderLock(int trajectory_id) const {
  auto next_it = C_submap_poses_.lower_bound(SubmapId(trajectory_id + 1, 0));
  if (next_it == C_submap_poses_.begin()) {
    return std::nullopt;
  }

  auto it = std::prev(next_it);
  if (it->first.trajectory_id != trajectory_id) {
    return std::nullopt;
  }
  return it->first;
}

std::optional<NodeId> OptimizationProblem3D::GetLastNodeIdUnderLock(int trajectory_id) const {
  auto next_it = C_node_poses_.lower_bound(NodeId(trajectory_id + 1, 0));
  if (next_it == C_node_poses_.begin()) {
    return std::nullopt;
  }

  auto it = std::prev(next_it);
  if (it->first.trajectory_id != trajectory_id) {
    return std::nullopt;
  }
  return it->first;
}

std::pair<std::map<int, SubmapId>, std::map<int, NodeId>>
OptimizationProblem3D::GetLastIds() const {
  absl::MutexLock locker(&mutex_);
  std::map<int, SubmapId> trajectory_id_to_last_submap_id;
  std::map<int, NodeId> trajectory_id_to_last_node_id;
  for (const int trajectory_id : node_data_.trajectory_ids()) {
    std::optional<SubmapId> last_submap_id = GetLastSubmapIdUnderLock(trajectory_id);
    std::optional<NodeId> last_node_id = GetLastNodeIdUnderLock(trajectory_id);
    if (!last_submap_id.has_value() || !last_node_id.has_value()) {
      continue;
    }
    trajectory_id_to_last_submap_id.emplace(trajectory_id, *last_submap_id);
    trajectory_id_to_last_node_id.emplace(trajectory_id, *last_node_id);
  }
  return std::make_pair(
      std::move(trajectory_id_to_last_submap_id),
      std::move(trajectory_id_to_last_node_id));
}

int OptimizationProblem3D::NumSubmapsOrZero(int trajectory_id) const {
  absl::MutexLock locker(&mutex_);
  if (num_submaps_.count(trajectory_id)) {
    return num_submaps_.at(trajectory_id);
  }
  return 0;
}

int OptimizationProblem3D::NumNodesOrZero(int trajectory_id) const {
  absl::MutexLock locker(&mutex_);
  if (num_nodes_.count(trajectory_id)) {
    return num_nodes_.at(trajectory_id);
  }
  return 0;
}

int OptimizationProblem3D::NumSubmaps() const {
  absl::MutexLock locker(&mutex_);
  return C_submap_poses_.size();
}

int OptimizationProblem3D::NumNodes() const {
  absl::MutexLock locker(&mutex_);
  return C_node_poses_.size();
}

}  // namespace optimization
}  // namespace mapping
}  // namespace cartographer
