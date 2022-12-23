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
  // The 'mutex_' of the pose graph is held while this class is used.
  class TrimmingHandle : public Trimmable {
   public:
    TrimmingHandle(PoseGraph3D* parent);
    ~TrimmingHandle() override {}

    int num_submaps(int trajectory_id) const override;
    std::vector<SubmapId> GetSubmapIds(int trajectory_id) const override;
    MapById<SubmapId, SubmapData> GetOptimizedSubmapData() const override;
    const MapById<NodeId, TrajectoryNode>& GetTrajectoryNodes() const override;
    const std::vector<Constraint>& GetConstraints() const override;
    void TrimSubmap(const SubmapId& submap_id);
    bool IsFinished(int trajectory_id) const override;

    void SetTrajectoryState(int trajectory_id, TrajectoryState state) override;

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
      const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps);

  void AddImuData(int trajectory_id, const sensor::ImuData& imu_data) override;
  void AddOdometryData(int trajectory_id,
                       const sensor::OdometryData& odometry_data) override;
  void AddFixedFramePoseData(
      int trajectory_id,
      const sensor::FixedFramePoseData& fixed_frame_pose_data) override;
  void AddLandmarkData(int trajectory_id,
                       const sensor::LandmarkData& landmark_data) override;

  void DeleteTrajectory(int trajectory_id) override;
  void FinishTrajectory(int trajectory_id) override;
  bool IsTrajectoryFinished(int trajectory_id) const override;
  void FreezeTrajectory(int trajectory_id) override;
  bool IsTrajectoryFrozen(int trajectory_id) const override;

  void AddSubmapFromProto(const transform::Rigid3d& global_submap_pose,
                          const proto::Submap& submap) override;
  void AddNodeFromProto(const transform::Rigid3d& global_pose,
                        const proto::Node& node) override;
  void SetTrajectoryDataFromProto(const proto::TrajectoryData& data) override;
  void AddSerializedConstraints(
      const std::vector<Constraint>& constraints) override;
  void AddNodeToSubmap(const NodeId& node_id,
                       const SubmapId& submap_id) override;

  void AddTrimmer(std::unique_ptr<PoseGraphTrimmer> trimmer) override;
  void AddLoopTrimmer(int trajectory_id,
      const proto::LoopTrimmerOptions& loop_trimmer_options) override;

  void RunFinalOptimization() override;

  std::vector<std::vector<int>> GetConnectedTrajectories() const override;
  bool TrajectoriesTransitivelyConnected(int trajectory_id_a, int trajectory_id_b) const override;
  common::Time TrajectoriesLastConnectionTime(int trajectory_id_a, int trajectory_id_b) const override;

  transform::Rigid3d GetInterpolatedGlobalTrajectoryPose(
      int trajectory_id, const common::Time time) const;
  transform::Rigid3d GetLocalToGlobalTransform(int trajectory_id) const;

  PoseGraph::SubmapData GetSubmapData(const SubmapId& submap_id) const;
  MapById<SubmapId, SubmapData> GetAllSubmapData() const;
  MapById<SubmapId, SubmapPose> GetAllSubmapPoses() const;
  MapById<NodeId, TrajectoryNode> GetTrajectoryNodes() const override;
  MapById<NodeId, TrajectoryNodePose> GetTrajectoryNodePoses() const override;
  std::map<int, TrajectoryState> GetTrajectoryStates() const override;
  std::map<std::string, transform::Rigid3d> GetLandmarkPoses() const override;
  void SetLandmarkPose(const std::string& landmark_id,
                       const transform::Rigid3d& global_pose,
                       const bool frozen = false) override;
  sensor::MapByTime<sensor::ImuData> GetImuData() const override;
  sensor::MapByTime<sensor::OdometryData> GetOdometryData() const override;
  sensor::MapByTime<sensor::FixedFramePoseData> GetFixedFramePoseData() const override;
  std::map<std::string, PoseGraph::LandmarkNode> GetLandmarkNodes() const override;
  std::map<int, TrajectoryData> GetTrajectoryData() const override;
  std::vector<Constraint> constraints() const override;

  void SetInitialTrajectoryPose(int from_trajectory_id, int to_trajectory_id,
                                const transform::Rigid3d& pose,
                                const common::Time time) override;

  void SetGlobalSlamOptimizationCallback(
      PoseGraphInterface::GlobalSlamOptimizationCallback callback) override;

  static void RegisterMetrics(metrics::FamilyFactory* family_factory);

 protected:
  void WaitForAllComputations();

 private:
  void AddWorkItem(const std::function<WorkItem::Result()>& work_item);
  void DrainWorkQueue();
  void HandleWorkQueue(const constraints::ConstraintBuilder3D::Result& result);

  std::vector<PoseGraphInterface::Constraint> TrimFalseDetectedLoops(
      const std::vector<PoseGraphInterface::Constraint>& new_loops);
  void TrimLoopsInWindow();
  std::vector<PoseGraphInterface::Constraint> TrimLoops(
      const std::vector<PoseGraphInterface::Constraint>& new_loops);

  void RunOptimization();

  std::pair<bool, bool> CheckIfConstraintCanBeAdded(
      const NodeId& node_id,
      const SubmapId& submap_id);

  std::pair<std::vector<SubmapId>, std::vector<SubmapId>>
      ComputeCandidatesForConstraints(const NodeId& node_id);
  std::pair<std::vector<NodeId>, std::vector<NodeId>>
      ComputeCandidatesForConstraints(const SubmapId& submap_id);

  std::pair<std::vector<SubmapId>, std::vector<SubmapId>>
      SelectCandidatesForConstraints(
          const std::vector<SubmapId>& local_candidates,
          const std::vector<SubmapId>& global_candidates);
  std::pair<std::vector<NodeId>, std::vector<NodeId>>
      SelectCandidatesForConstraints(
          const std::vector<NodeId>& local_candidates,
          const std::vector<NodeId>& global_candidates);

  void MaybeAddConstraints(const NodeId& node_id,
      const std::vector<SubmapId>& local_submap_ids,
      const std::vector<SubmapId>& global_submap_ids);
  void MaybeAddConstraints(const SubmapId& submap_id,
      const std::vector<NodeId>& local_node_ids,
      const std::vector<NodeId>& global_node_ids);

  std::vector<SubmapId> InitializeGlobalSubmapPoses(
      int trajectory_id, const common::Time time,
      const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps);

  WorkItem::Result ComputeConstraintsForNode(
      const NodeId& node_id,
      std::vector<std::shared_ptr<const Submap3D>> insertion_submaps,
      bool newly_finished_submap);

  NodeId AppendNode(
      std::shared_ptr<const TrajectoryNode::Data> constant_data,
      int trajectory_id,
      const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps,
      const transform::Rigid3d& optimized_pose);

  transform::Rigid3d ComputeLocalToGlobalTransform(
      const MapById<SubmapId, optimization::SubmapSpec3D>& global_submap_poses,
      int trajectory_id) const;

  void AddTrajectoryIfNeeded(int trajectory_id);
  void DeleteTrajectoriesIfNeeded();

  bool CanAddWorkItemModifying(int trajectory_id);

  MapById<SubmapId, SubmapData> GetSubmapDataUnderLock() const;
  PoseGraph::SubmapData GetSubmapDataUnderLock(const SubmapId& submap_id) const;

  common::Time GetLatestNodeTime(const NodeId& node_id,
                                 const SubmapId& submap_id) const;
  
  void UpdateTrajectoryConnectivity(const Constraint& constraint);

  void LogResidualHistograms() const;

 private:
  mutable absl::Mutex mutex_;
  absl::Mutex work_queue_mutex_;

  const proto::PoseGraphOptions options_;
  std::unique_ptr<optimization::OptimizationProblem3D> optimization_problem_;
  constraints::ConstraintBuilder3D constraint_builder_;
  common::ThreadPool* const thread_pool_;

  std::unique_ptr<WorkQueue> work_queue_;

  int num_nodes_since_last_loop_closure_ = 0;

  std::vector<std::unique_ptr<PoseGraphTrimmer>> trimmers_;
  std::set<int> pure_localization_trajectory_ids_;
  std::map<int, proto::LoopTrimmerOptions> loop_trimmer_options_;

  double num_local_constraints_to_compute_ = 0.0;
  double num_global_constraints_to_compute_ = 0.0;
  std::set<SubmapId> submaps_used_for_local_constraints_;
  std::set<SubmapId> submaps_used_for_global_constraints_;

  PoseGraphData data_;

  GlobalSlamOptimizationCallback global_slam_optimization_callback_;
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_3D_POSE_GRAPH_3D_H_
