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

#include "cartographer/mapping/internal/2d/pose_graph_2d.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
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

PoseGraph2D::PoseGraph2D(
    const proto::PoseGraphOptions& options,
    std::unique_ptr<optimization::OptimizationProblem2D> optimization_problem,
    common::ThreadPool* thread_pool)
    : options_(options),
      optimization_problem_(std::move(optimization_problem)),
      constraint_builder_(options_.constraint_builder_options(), thread_pool),
      thread_pool_(thread_pool) {
  CHECK(false) << "2d pose graph is not supported";
}

PoseGraph2D::~PoseGraph2D() {
}

std::vector<SubmapId> PoseGraph2D::InitializeGlobalSubmapPoses(
    const int trajectory_id, const common::Time time,
    const std::vector<std::shared_ptr<const Submap2D>>& insertion_submaps) {
  return std::vector<SubmapId>();
}

NodeId PoseGraph2D::AppendNode(
    std::shared_ptr<const TrajectoryNode::Data> constant_data,
    const int trajectory_id,
    const std::vector<std::shared_ptr<const Submap2D>>& insertion_submaps,
    const transform::Rigid3d& optimized_pose) {
  return NodeId(0, 0);
}

NodeId PoseGraph2D::AddNode(
    std::shared_ptr<const TrajectoryNode::Data> constant_data,
    const int trajectory_id,
    const std::vector<std::shared_ptr<const Submap2D>>& insertion_submaps) {
  return NodeId(0, 0);
}

void PoseGraph2D::AddWorkItem(
    const std::function<WorkItem::Result()>& work_item) {
}

void PoseGraph2D::AddTrajectoryIfNeeded(const int trajectory_id) {
}

void PoseGraph2D::AddImuData(const int trajectory_id,
                             const sensor::ImuData& imu_data) {
}

void PoseGraph2D::AddOdometryData(const int trajectory_id,
                                  const sensor::OdometryData& odometry_data) {
}

void PoseGraph2D::AddFixedFramePoseData(
    const int trajectory_id,
    const sensor::FixedFramePoseData& fixed_frame_pose_data) {
}

void PoseGraph2D::AddLandmarkData(int trajectory_id,
                                  const sensor::LandmarkData& landmark_data) {
}

std::pair<bool, bool> PoseGraph2D::CheckIfConstraintCanBeAdded(
    const NodeId& node_id,
    const SubmapId& submap_id,
    bool only_local_constraint /* false */,
    bool only_global_constraint /* false */) {
  return std::pair<bool, bool>();
}

WorkItem::Result PoseGraph2D::ComputeConstraintsForNode(
    const NodeId& node_id,
    std::vector<std::shared_ptr<const Submap2D>> insertion_submaps,
    const bool newly_finished_submap) {
  return WorkItem::Result::kDoNotRunOptimization;
}

common::Time PoseGraph2D::GetLatestNodeTime(const NodeId& node_id,
                                            const SubmapId& submap_id) const {
  return common::Time();
}

void PoseGraph2D::UpdateTrajectoryConnectivity(const Constraint& constraint) {
}

void PoseGraph2D::DeleteTrajectoriesIfNeeded() {
}

void PoseGraph2D::HandleWorkQueue(
    const constraints::ConstraintBuilder2D::Result& result) {
}

void PoseGraph2D::DrainWorkQueue() {
}

void PoseGraph2D::WaitForAllComputations() {
}

void PoseGraph2D::DeleteTrajectory(const int trajectory_id) {
}

void PoseGraph2D::FinishTrajectory(const int trajectory_id) {
}

bool PoseGraph2D::IsTrajectoryFinished(const int trajectory_id) const {
  return bool();
}

void PoseGraph2D::FreezeTrajectory(const int trajectory_id) {
}

bool PoseGraph2D::IsTrajectoryFrozen(const int trajectory_id) const {
  return bool();
}

void PoseGraph2D::AddSubmapFromProto(
    const transform::Rigid3d& global_submap_pose, const proto::Submap& submap) {
}

void PoseGraph2D::AddNodeFromProto(const transform::Rigid3d& global_pose,
                                   const proto::Node& node) {
}

void PoseGraph2D::SetTrajectoryDataFromProto(
    const proto::TrajectoryData& data) {
}

void PoseGraph2D::AddSerializedConstraints(
    const std::vector<Constraint>& constraints) {
}

void PoseGraph2D::AddLoopTrimmer(int trajectory_id,
    const proto::LoopTrimmerOptions& loop_trimmer_options) {
}

void PoseGraph2D::RunFinalOptimization() {
}

void PoseGraph2D::RunOptimization() {
}

bool PoseGraph2D::CanAddWorkItemModifying(int trajectory_id) {
  return bool();
}

MapById<NodeId, TrajectoryNode> PoseGraph2D::GetTrajectoryNodes() const {
  return MapById<NodeId, TrajectoryNode>();
}

MapById<NodeId, TrajectoryNodePose> PoseGraph2D::GetTrajectoryNodePoses()
    const {
  return MapById<NodeId, TrajectoryNodePose>();
}

std::map<int, TrajectoryState>
PoseGraph2D::GetTrajectoryStates() const {
  return std::map<int, TrajectoryState>();
}

std::map<std::string, transform::Rigid3d> PoseGraph2D::GetLandmarkPoses()
    const {
  return std::map<std::string, transform::Rigid3d>();
}

void PoseGraph2D::SetLandmarkPose(const std::string& landmark_id,
                                  const transform::Rigid3d& global_pose,
                                  const bool frozen) {
}

sensor::MapByTime<sensor::ImuData> PoseGraph2D::GetImuData() const {
  return sensor::MapByTime<sensor::ImuData>();
}

sensor::MapByTime<sensor::OdometryData> PoseGraph2D::GetOdometryData() const {
  return sensor::MapByTime<sensor::OdometryData>();
}

std::map<std::string /* landmark ID */, PoseGraphInterface::LandmarkNode>
PoseGraph2D::GetLandmarkNodes() const {
  return std::map<std::string /* landmark ID */, PoseGraphInterface::LandmarkNode>();
}

std::map<int, PoseGraphInterface::TrajectoryData>
PoseGraph2D::GetTrajectoryData() const {
  return std::map<int, PoseGraphInterface::TrajectoryData>();
}

sensor::MapByTime<sensor::FixedFramePoseData>
PoseGraph2D::GetFixedFramePoseData() const {
  return sensor::MapByTime<sensor::FixedFramePoseData>();
}

std::vector<Constraint> PoseGraph2D::constraints() const {
  return std::vector<Constraint>();
}

void PoseGraph2D::SetInitialTrajectoryPose(const int from_trajectory_id,
                                           const int to_trajectory_id,
                                           const transform::Rigid3d& pose,
                                           const common::Time time) {
}

transform::Rigid3d PoseGraph2D::GetInterpolatedGlobalTrajectoryPose(
    const int trajectory_id, const common::Time time) const {
  return transform::Rigid3d();
}

transform::Rigid3d PoseGraph2D::GetLocalToGlobalTransform(
    const int trajectory_id) const {
  return transform::Rigid3d();
}

std::vector<std::vector<int>> PoseGraph2D::GetConnectedTrajectories() const {
  return std::vector<std::vector<int>>();
}

bool PoseGraph2D::TrajectoriesTransitivelyConnected(int trajectory_id_a, int trajectory_id_b) const {
  return bool();
}

common::Time PoseGraph2D::TrajectoriesLastConnectionTime(int trajectory_id_a, int trajectory_id_b) const {
  return common::Time();
}

PoseGraphInterface::SubmapData PoseGraph2D::GetSubmapData(
    const SubmapId& submap_id) const {
  return PoseGraphInterface::SubmapData();
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph2D::GetAllSubmapData() const {
  return MapById<SubmapId, PoseGraphInterface::SubmapData>();
}

MapById<SubmapId, PoseGraphInterface::SubmapPose>
PoseGraph2D::GetAllSubmapPoses() const {
  return MapById<SubmapId, PoseGraphInterface::SubmapPose>();
}

transform::Rigid3d PoseGraph2D::ComputeLocalToGlobalTransform(
    const MapById<SubmapId, optimization::SubmapSpec2D>& global_submap_poses,
    const int trajectory_id) const {
  return transform::Rigid3d();
}

PoseGraphInterface::SubmapData PoseGraph2D::GetSubmapDataUnderLock(
    const SubmapId& submap_id) const {
  return PoseGraphInterface::SubmapData();
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph2D::GetSubmapDataUnderLock() const {
  return MapById<SubmapId, PoseGraphInterface::SubmapData>();
}

void PoseGraph2D::SetGlobalSlamOptimizationCallback(
    PoseGraphInterface::GlobalSlamOptimizationCallback callback) {
}

void PoseGraph2D::RegisterMetrics(metrics::FamilyFactory* family_factory) {
}

}  // namespace mapping
}  // namespace cartographer
