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

#ifndef CARTOGRAPHER_COMMON_TIME_H_
#define CARTOGRAPHER_COMMON_TIME_H_

#include <chrono>
#include <iostream>
#include <ratio>
#include <vector>

#include "cartographer/common/port.h"

namespace cartographer {
namespace common {

constexpr int64 kUtsEpochOffsetFromUnixEpochInSeconds =
    (719162ll * 24ll * 60ll * 60ll);

struct UniversalTimeScaleClock {
  using rep = int64;
  using period = std::ratio<1, 10000000>;
  using duration = std::chrono::duration<rep, period>;
  using time_point = std::chrono::time_point<UniversalTimeScaleClock>;
  static constexpr bool is_steady = true;
};

// Represents Universal Time Scale durations and timestamps which are 64-bit
// integers representing the 100 nanosecond ticks since the Epoch which is
// January 1, 1 at the start of day in UTC.
using Duration = UniversalTimeScaleClock::duration;
using Time = UniversalTimeScaleClock::time_point;

// Convenience functions to create common::Durations.
Duration FromSeconds(double seconds);
Duration FromMilliseconds(int64 milliseconds);

// Returns the given duration in seconds.
double ToSeconds(Duration duration);
double ToSeconds(std::chrono::steady_clock::duration duration);

// Creates a time from a Universal Time Scale.
Time FromUniversal(int64 ticks);

// Outputs the Universal Time Scale timestamp for a given Time.
int64 ToUniversal(Time time);

// For logging and unit tests, outputs the timestamp integer.
std::ostream& operator<<(std::ostream& os, Time time);

// CPU time consumed by the thread so far, in seconds.
double GetThreadCpuTimeSeconds();

class TimeMeasurer {
 public:
  TimeMeasurer(std::string name, bool print_results_on_destruction);
  ~TimeMeasurer();

  void StartMeasurement();
  void StopMeasurement();

 private:
  std::string name_;
  bool print_results_on_destruction_;
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
  std::vector<double> time_measurements_;
  bool is_measuring_;
  pthread_t thread_id_;
};

#define MEASURE_TIME_FROM_HERE(name) \
  static cartographer::common::TimeMeasurer (name ## _time_measurer)(#name, true); \
  (name ## _time_measurer).StartMeasurement()

#define STOP_TIME_MESUREMENT(name) \
  (name ## _time_measurer).StopMeasurement()

#define MEASURE_BLOCK_TIME(name) \
  static cartographer::common::TimeMeasurer (name ## _time_measurer)(#name, true); \
  class name ## _time_measurer_stop_trigger_class { \
   public: \
    (name ## _time_measurer_stop_trigger_class)() {}; \
    (~name ## _time_measurer_stop_trigger_class)() {(name ## _time_measurer).StopMeasurement();}; \
  }; \
  name ## _time_measurer_stop_trigger_class    name ## _time_measurer_stop_trigger; \
  (name ## _time_measurer).StartMeasurement()

}  // namespace common
}  // namespace cartographer

#endif  // CARTOGRAPHER_COMMON_TIME_H_
