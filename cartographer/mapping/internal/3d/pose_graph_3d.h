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

#ifndef CARTOGRAPHER_MAPPING_INTERNAL_3D_POSE_GRAPH_3D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_3D_POSE_GRAPH_3D_H_

#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "cartographer/common/fixed_ratio_sampler.h"
#include "cartographer/common/thread_pool.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/3d/submap_3d.h"
#include "cartographer/mapping/internal/constraints/constraint_builder_3d.h"
#include "cartographer/mapping/internal/optimization/optimization_problem_3d.h"
#include "cartographer/mapping/internal/trajectory_connectivity_state.h"
#include "cartographer/mapping/internal/pose_graph_data.h"
#include "cartographer/mapping/internal/work_queue.h"
#include "cartographer/mapping/pose_graph.h"
#include "cartographer/mapping/pose_graph_trimmer.h"
#include "cartographer/metrics/family_factory.h"
#include "cartographer/sensor/fixed_frame_pose_data.h"
#include "cartographer/sensor/landmark_data.h"
#include "cartographer/sensor/odometry_data.h"
#include "cartographer/sensor/point_cloud.h"
#include "cartographer/transform/transform.h"

#include "time_measurer/time_measurer.h"

namespace cartographer {
namespace mapping {

class PoseGraph3D : public PoseGraph {
private:
  class TrimmingHandle : public Trimmable {
  public:
    TrimmingHandle(PoseGraph3D* parent);
    ~TrimmingHandle() override {}

    int num_submaps(int trajectory_id) const override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->executing_work_item_mutex_);
    std::vector<SubmapId> GetSubmapIds(int trajectory_id) const override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->executing_work_item_mutex_);
    MapById<SubmapId, SubmapData> GetOptimizedSubmapData() const override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->mutex_)
        EXCLUSIVE_LOCKS_REQUIRED(parent_->executing_work_item_mutex_);
    const MapById<NodeId, TrajectoryNode>& GetTrajectoryNodes() const override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->mutex_)
        EXCLUSIVE_LOCKS_REQUIRED(parent_->executing_work_item_mutex_);
    const std::vector<Constraint>& GetConstraints() const override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->mutex_)
        EXCLUSIVE_LOCKS_REQUIRED(parent_->executing_work_item_mutex_);
    void TrimSubmap(const SubmapId& submap_id) override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->mutex_)
        EXCLUSIVE_LOCKS_REQUIRED(parent_->executing_work_item_mutex_);
    bool IsFinished(int trajectory_id) const override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->mutex_)
        EXCLUSIVE_LOCKS_REQUIRED(parent_->executing_work_item_mutex_);
    void SetTrajectoryState(int trajectory_id, TrajectoryState state) override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->mutex_)
        EXCLUSIVE_LOCKS_REQUIRED(parent_->executing_work_item_mutex_);

  private:
    PoseGraph3D* const parent_;
  };


public:
  PoseGraph3D(
      const proto::PoseGraphOptions& options,
      std::unique_ptr<optimization::OptimizationProblem3D> optimization_problem,
      common::ThreadPool* thread_pool);
  ~PoseGraph3D() override;

  PoseGraph3D(const PoseGraph3D&) = delete;
  PoseGraph3D& operator=(const PoseGraph3D&) = delete;

  NodeId AddNode(
      std::shared_ptr<const TrajectoryNode::Data> constant_data,
      int trajectory_id,
      const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps)
          LOCKS_EXCLUDED(mutex_)
          LOCKS_EXCLUDED(work_queue_mutex_);

  void AddImuData(
      int trajectory_id,
      const sensor::ImuData& imu_data) override
          LOCKS_EXCLUDED(work_queue_mutex_);
  void AddOdometryData(
      int trajectory_id,
      const sensor::OdometryData& odometry_data) override
          LOCKS_EXCLUDED(work_queue_mutex_);
  void AddFixedFramePoseData(
      int trajectory_id,
      const sensor::FixedFramePoseData& fixed_frame_pose_data) override
          LOCKS_EXCLUDED(work_queue_mutex_);
  void AddLandmarkData(
      int trajectory_id,
      const sensor::LandmarkData& landmark_data) override
          LOCKS_EXCLUDED(work_queue_mutex_);

  void DeleteTrajectory(int trajectory_id) override
      LOCKS_EXCLUDED(mutex_)
      LOCKS_EXCLUDED(work_queue_mutex_);
  void FinishTrajectory(int trajectory_id) override
      LOCKS_EXCLUDED(work_queue_mutex_);
  bool IsTrajectoryFinished(int trajectory_id) const override
      LOCKS_EXCLUDED(mutex_);
  void FreezeTrajectory(int trajectory_id) override
      LOCKS_EXCLUDED(mutex_)
      LOCKS_EXCLUDED(work_queue_mutex_);
  bool IsTrajectoryFrozen(int trajectory_id) const override
      LOCKS_EXCLUDED(mutex_);

  void AddSubmapFromProto(
      const transform::Rigid3d& global_submap_pose,
      const proto::Submap& submap) override
          LOCKS_EXCLUDED(mutex_)
          LOCKS_EXCLUDED(work_queue_mutex_);
  void AddNodeFromProto(
      const transform::Rigid3d& global_pose,
      const proto::Node& node) override
          LOCKS_EXCLUDED(mutex_)
          LOCKS_EXCLUDED(work_queue_mutex_);
  void SetTrajectoryDataFromProto(
      const proto::TrajectoryData& data) override
          LOCKS_EXCLUDED(work_queue_mutex_);
  void SetLandmarkPose(
      const std::string& landmark_id,
      const transform::Rigid3d& global_pose,
      bool frozen = false) override
          LOCKS_EXCLUDED(work_queue_mutex_);
  void AddSerializedConstraints(
      const std::vector<Constraint>& constraints) override
          LOCKS_EXCLUDED(work_queue_mutex_);

  void AddTrimmer(std::unique_ptr<PoseGraphTrimmer> trimmer) override
      LOCKS_EXCLUDED(work_queue_mutex_);
  void AddLoopTrimmer(
      int trajectory_id,
      const proto::LoopTrimmerOptions& loop_trimmer_options) override
          LOCKS_EXCLUDED(work_queue_mutex_);

  void RunFinalOptimization() override
      LOCKS_EXCLUDED(mutex_)
      LOCKS_EXCLUDED(work_queue_mutex_);

  std::vector<std::vector<int>> GetConnectedTrajectories() const override
      LOCKS_EXCLUDED(mutex_);
  bool TrajectoriesTransitivelyConnected(
      int trajectory_id_a, int trajectory_id_b) const override
          LOCKS_EXCLUDED(mutex_);
  common::Time TrajectoriesLastConnectionTime(
      int trajectory_id_a, int trajectory_id_b) const override
          LOCKS_EXCLUDED(mutex_);

  transform::Rigid3d GetLocalToGlobalTransform(int trajectory_id) const
      LOCKS_EXCLUDED(mutex_);

  PoseGraph::SubmapData GetSubmapData(const SubmapId& submap_id) const
      LOCKS_EXCLUDED(mutex_);
  MapById<SubmapId, SubmapData> GetAllSubmapData() const
      LOCKS_EXCLUDED(mutex_);
  MapById<SubmapId, SubmapPose> GetAllSubmapPoses() const
      LOCKS_EXCLUDED(mutex_);
  MapById<NodeId, TrajectoryNode> GetTrajectoryNodes() const override
      LOCKS_EXCLUDED(mutex_);
  MapById<NodeId, TrajectoryNodePose> GetTrajectoryNodePoses() const override
      LOCKS_EXCLUDED(mutex_);
  std::map<int, TrajectoryState> GetTrajectoryStates() const override
      LOCKS_EXCLUDED(mutex_);
  std::map<std::string, transform::Rigid3d> GetLandmarkPoses() const override
      LOCKS_EXCLUDED(mutex_);
  sensor::MapByTime<sensor::ImuData> GetImuData() const override
      LOCKS_EXCLUDED(mutex_);
  sensor::MapByTime<sensor::OdometryData> GetOdometryData() const override
      LOCKS_EXCLUDED(mutex_);
  sensor::MapByTime<sensor::FixedFramePoseData> GetFixedFramePoseData() const override
      LOCKS_EXCLUDED(mutex_);
  std::map<std::string, PoseGraph::LandmarkNode> GetLandmarkNodes() const override
      LOCKS_EXCLUDED(mutex_);
  std::map<int, TrajectoryData> GetTrajectoryData() const override
      LOCKS_EXCLUDED(mutex_);
  std::vector<Constraint> constraints() const override
      LOCKS_EXCLUDED(mutex_);

  void SetInitialTrajectoryPose(
      int from_trajectory_id, int to_trajectory_id,
      const transform::Rigid3d& pose, common::Time time) override
          LOCKS_EXCLUDED(mutex_);

  void SetGlobalSlamOptimizationCallback(
      PoseGraphInterface::GlobalSlamOptimizationCallback callback) override
          LOCKS_EXCLUDED(mutex_);

  static void RegisterMetrics(metrics::FamilyFactory* family_factory);

protected:
  void WaitForAllComputations()
      LOCKS_EXCLUDED(mutex_)
      LOCKS_EXCLUDED(work_queue_mutex_);

private:
  void AddWorkItem(const std::function<WorkItem::Result()>& work_item)
      LOCKS_EXCLUDED(work_queue_mutex_);
  void HandleWorkQueue(const constraints::ConstraintBuilder3D::Result& result)
      LOCKS_EXCLUDED(mutex_)
      LOCKS_EXCLUDED(work_queue_mutex_)
      LOCKS_EXCLUDED(executing_work_item_mutex_);
  void DrainWorkQueue()
      LOCKS_EXCLUDED(mutex_)
      LOCKS_EXCLUDED(work_queue_mutex_)
      LOCKS_EXCLUDED(executing_work_item_mutex_);

  void RunOptimization()
      LOCKS_EXCLUDED(mutex_)
      EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);

  double GetTravelledDistanceWithLoopsSameTrajectory(
      NodeId node_1, NodeId node_2, float min_score)
          EXCLUSIVE_LOCKS_REQUIRED(mutex_)
          EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);
  double GetTravelledDistanceWithLoopsDifferentTrajectories(
      NodeId node_1, NodeId node_2, float min_score)
          EXCLUSIVE_LOCKS_REQUIRED(mutex_)
          EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);
  double GetTravelledDistanceWithLoops(
      NodeId node_1, NodeId node_2, float min_score)
          EXCLUSIVE_LOCKS_REQUIRED(mutex_)
          EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);
  std::vector<PoseGraphInterface::Constraint> TrimFalseDetectedLoops(
      const std::vector<PoseGraphInterface::Constraint>& new_loops)
          LOCKS_EXCLUDED(mutex_)
          EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);
  void TrimLoopsInWindow()
      LOCKS_EXCLUDED(mutex_)
      EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);
  std::vector<PoseGraphInterface::Constraint> TrimLoops(
      const std::vector<PoseGraphInterface::Constraint>& new_loops)
          LOCKS_EXCLUDED(mutex_)
          EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);

  std::pair<bool, bool> CheckIfConstraintCanBeAdded(
      const NodeId& node_id, const SubmapId& submap_id)
          LOCKS_EXCLUDED(mutex_)
          EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);

  std::pair<std::vector<SubmapId>, std::vector<SubmapId>>
      ComputeCandidatesForConstraints(const NodeId& node_id)
          LOCKS_EXCLUDED(mutex_)
          EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);
  std::pair<std::vector<NodeId>, std::vector<NodeId>>
      ComputeCandidatesForConstraints(const SubmapId& submap_id)
          LOCKS_EXCLUDED(mutex_)
          EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);

  std::pair<std::vector<SubmapId>, std::vector<SubmapId>>
      SelectCandidatesForConstraints(
          const std::vector<SubmapId>& local_candidates,
          const std::vector<SubmapId>& global_candidates)
              EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);
  std::pair<std::vector<NodeId>, std::vector<NodeId>>
      SelectCandidatesForConstraints(
          const std::vector<NodeId>& local_candidates,
          const std::vector<NodeId>& global_candidates)
              EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);

  void MaybeAddConstraints(const NodeId& node_id,
      const std::vector<SubmapId>& local_submap_ids,
      const std::vector<SubmapId>& global_submap_ids)
          LOCKS_EXCLUDED(mutex_)
          EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);
  void MaybeAddConstraints(const SubmapId& submap_id,
      const std::vector<NodeId>& local_node_ids,
      const std::vector<NodeId>& global_node_ids)
          LOCKS_EXCLUDED(mutex_)
          EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);

  std::vector<SubmapId> InitializeGlobalSubmapPoses(
      int trajectory_id, const common::Time time,
      const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps)
          EXCLUSIVE_LOCKS_REQUIRED(mutex_)
          EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);

  WorkItem::Result ComputeConstraintsForNode(
      const NodeId& node_id,
      std::vector<std::shared_ptr<const Submap3D>> insertion_submaps,
      bool newly_finished_submap)
          LOCKS_EXCLUDED(mutex_)
          LOCKS_EXCLUDED(executing_work_item_mutex_);

  NodeId AppendNode(
      std::shared_ptr<const TrajectoryNode::Data> constant_data,
      int trajectory_id,
      const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps,
      const transform::Rigid3d& optimized_pose)
          LOCKS_EXCLUDED(mutex_);

  transform::Rigid3d GetInterpolatedGlobalTrajectoryPose(
      int trajectory_id, const common::Time time) const
          EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  transform::Rigid3d ComputeLocalToGlobalTransform(
      const MapById<SubmapId, optimization::SubmapSpec3D>& global_submap_poses,
      int trajectory_id) const
          EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  void AddTrajectoryIfNeeded(int trajectory_id)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void DeleteTrajectoriesIfNeeded()
      EXCLUSIVE_LOCKS_REQUIRED(mutex_)
      EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);

  bool CanAddWorkItemModifying(int trajectory_id)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool IsTrajectoryFinishedUnderLock(int trajectory_id) const
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool IsTrajectoryFrozenUnderLock(int trajectory_id) const
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  PoseGraph::SubmapData GetSubmapDataUnderLock(const SubmapId& submap_id) const
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  MapById<SubmapId, SubmapData> GetSubmapDataUnderLock() const
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  common::Time GetLatestNodeTime(
      const NodeId& node_id, const SubmapId& submap_id) const
          EXCLUSIVE_LOCKS_REQUIRED(mutex_)
          EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);
  
  void UpdateTrajectoryConnectivity(const Constraint& constraint)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_)
      EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);

  void LogResidualHistograms() const
      EXCLUSIVE_LOCKS_REQUIRED(mutex_)
      EXCLUSIVE_LOCKS_REQUIRED(executing_work_item_mutex_);

private:
  mutable absl::Mutex mutex_;
  absl::Mutex work_queue_mutex_;
  absl::Mutex executing_work_item_mutex_;

  const proto::PoseGraphOptions options_;
  std::unique_ptr<optimization::OptimizationProblem3D> optimization_problem_;
  constraints::ConstraintBuilder3D constraint_builder_;
  common::ThreadPool* const thread_pool_;

  std::unique_ptr<WorkQueue> work_queue_ GUARDED_BY(work_queue_mutex_);

  int num_nodes_since_last_loop_closure_ GUARDED_BY(executing_work_item_mutex_);

  std::vector<std::unique_ptr<PoseGraphTrimmer>> trimmers_
      GUARDED_BY(executing_work_item_mutex_);
  std::set<int> pure_localization_trajectory_ids_
      GUARDED_BY(executing_work_item_mutex_);
  std::vector<TrimmedLoop> trimmed_loops_
      GUARDED_BY(executing_work_item_mutex_);  // loops trimmed
          // by TrimmingHandle::TrimSubmap()
  std::map<int, proto::LoopTrimmerOptions> loop_trimmer_options_
      GUARDED_BY(executing_work_item_mutex_);

  double num_local_constraints_to_compute_ GUARDED_BY(executing_work_item_mutex_);
  double num_global_constraints_to_compute_ GUARDED_BY(executing_work_item_mutex_);
  std::set<SubmapId> submaps_used_for_local_constraints_
      GUARDED_BY(executing_work_item_mutex_);
  std::set<SubmapId> submaps_used_for_global_constraints_
      GUARDED_BY(executing_work_item_mutex_);

  PoseGraphData data_ GUARDED_BY(mutex_);

  GlobalSlamOptimizationCallback global_slam_optimization_callback_
      GUARDED_BY(executing_work_item_mutex_);
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_3D_POSE_GRAPH_3D_H_
