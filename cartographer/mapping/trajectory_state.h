#pragma once

namespace cartographer {
namespace mapping {

enum class TrajectoryState {
  ACTIVE,
  FINISHED,
  FROZEN,
  DELETED
};

}
}
