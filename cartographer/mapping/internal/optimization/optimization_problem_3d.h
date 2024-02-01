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

#ifndef CARTOGRAPHER_MAPPING_INTERNAL_OPTIMIZATION_OPTIMIZATION_PROBLEM_3D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_OPTIMIZATION_OPTIMIZATION_PROBLEM_3D_H_

#include <array>
#include <map>
#include <set>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/types/optional.h"
#include "absl/synchronization/mutex.h"
#include "cartographer/common/port.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/id.h"
#include "cartographer/mapping/internal/optimization/ceres_pose.h"
#include "cartographer/mapping/internal/optimization/optimization_problem_interface.h"
#include "cartographer/mapping/pose_graph_interface.h"
#include "cartographer/mapping/proto/pose_graph/optimization_problem_options.pb.h"
#include "cartographer/sensor/fixed_frame_pose_data.h"
#include "cartographer/sensor/imu_data.h"
#include "cartographer/sensor/map_by_time.h"
#include "cartographer/sensor/odometry_data.h"
#include "cartographer/transform/transform_interpolation_buffer.h"

namespace cartographer {
namespace mapping {
namespace optimization {

struct NodeData {
  common::Time time;
  transform::Rigid3d local_pose;
};

class OptimizationProblem3D : public OptimizationProblemInterface<transform::Rigid3d> {
public:
  explicit OptimizationProblem3D(
      const optimization::proto::OptimizationProblemOptions& options);
  ~OptimizationProblem3D();

  OptimizationProblem3D(const OptimizationProblem3D&) = delete;
  OptimizationProblem3D& operator=(const OptimizationProblem3D&) = delete;

  void InsertTrajectoryNode(const NodeId& node_id,
      const common::Time& time, const transform::Rigid3d& local_pose,
      const transform::Rigid3d& global_pose) override
          ABSL_LOCKS_EXCLUDED(mutex_);
  void TrimTrajectoryNode(const NodeId& node_id) override
      ABSL_LOCKS_EXCLUDED(mutex_);
  void InsertSubmap(const SubmapId& submap_id,
      const transform::Rigid3d& global_pose) override
          ABSL_LOCKS_EXCLUDED(mutex_);
  void TrimSubmap(const SubmapId& submap_id) override
      ABSL_LOCKS_EXCLUDED(mutex_);
  void SetMaxNumIterations(int32 max_num_iterations) override
      ABSL_LOCKS_EXCLUDED(mutex_);

  bool BuildProblem(
      const std::vector<Constraint>& constraints,
      const sensor::MapByTime<sensor::ImuData>& imu_data,
      const sensor::MapByTime<sensor::OdometryData>& odometry_data,
      const sensor::MapByTime<sensor::FixedFramePoseData>& fixed_frame_pose_data,
      const PoseGraphTrajectoryStates& trajectory_states,
      const std::map<int, PoseGraphInterface::TrajectoryData>& trajectory_data,
      const std::map<std::string, LandmarkNode>& landmark_nodes) override
          ABSL_LOCKS_EXCLUDED(mutex_);
  void Solve() override
      ABSL_LOCKS_EXCLUDED(mutex_);

  MapById<SubmapId, transform::Rigid3d> GetSubmapPoses() const override
      ABSL_LOCKS_EXCLUDED(mutex_);
  MapById<NodeId, transform::Rigid3d> GetNodePoses() const override
      ABSL_LOCKS_EXCLUDED(mutex_);
  std::map<int, PoseGraphInterface::TrajectoryData> GetTrajectoryData() const
      ABSL_LOCKS_EXCLUDED(mutex_);
  std::map<std::string, transform::Rigid3d> GetLandmarkData() const override
      ABSL_LOCKS_EXCLUDED(mutex_);

  std::optional<SubmapId> GetLastSubmapId(int trajectory_id) const
      ABSL_LOCKS_EXCLUDED(mutex_);
  std::optional<NodeId> GetLastNodeId(int trajectory_id) const
      ABSL_LOCKS_EXCLUDED(mutex_);
  std::pair<std::map<int, SubmapId>, std::map<int, NodeId>> GetLastIds() const
      ABSL_LOCKS_EXCLUDED(mutex_);

  int NumSubmapsOrZero(int trajectory_id) const ABSL_LOCKS_EXCLUDED(mutex_);
  int NumNodesOrZero(int trajectory_id) const ABSL_LOCKS_EXCLUDED(mutex_);

  int NumSubmaps() const ABSL_LOCKS_EXCLUDED(mutex_);
  int NumNodes() const ABSL_LOCKS_EXCLUDED(mutex_);

private:
  std::optional<SubmapId> GetLastSubmapIdUnderLock(int trajectory_id) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  std::optional<NodeId> GetLastNodeIdUnderLock(int trajectory_id) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

private:
  mutable absl::Mutex mutex_;

  optimization::proto::OptimizationProblemOptions options_ ABSL_GUARDED_BY(mutex_);

  std::map<SubmapId, CeresPose> C_submap_poses_ ABSL_GUARDED_BY(mutex_);
  std::map<NodeId, CeresPose> C_node_poses_ ABSL_GUARDED_BY(mutex_);
  MapById<NodeId, NodeData> node_data_ ABSL_GUARDED_BY(mutex_);

  std::map<int, int> num_submaps_ ABSL_GUARDED_BY(mutex_);
  std::map<int, int> num_nodes_ ABSL_GUARDED_BY(mutex_);

  std::map<std::string, transform::Rigid3d> landmark_data_ ABSL_GUARDED_BY(mutex_);
  std::map<int, PoseGraphInterface::TrajectoryData> trajectory_data_ ABSL_GUARDED_BY(mutex_);

  std::unique_ptr<ceres::Problem> problem_ ABSL_GUARDED_BY(mutex_);
  std::map<std::string, CeresPose> C_landmark_poses_ ABSL_GUARDED_BY(mutex_);
  std::map<int, CeresPose> C_fixed_frame_poses_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace optimization
}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_OPTIMIZATION_OPTIMIZATION_PROBLEM_3D_H_
