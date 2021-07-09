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

#include "cartographer/common/time.h"

#include <time.h>

#include <cerrno>
#include <cstring>
#include <string>

#include "glog/logging.h"

namespace cartographer {
namespace common {

Duration FromSeconds(const double seconds) {
  return std::chrono::duration_cast<Duration>(
      std::chrono::duration<double>(seconds));
}

double ToSeconds(const Duration duration) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(duration)
      .count();
}

double ToSeconds(const std::chrono::steady_clock::duration duration) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(duration)
      .count();
}

Time FromUniversal(const int64 ticks) { return Time(Duration(ticks)); }

int64 ToUniversal(const Time time) { return time.time_since_epoch().count(); }

std::ostream& operator<<(std::ostream& os, const Time time) {
  os << std::to_string(ToUniversal(time));
  return os;
}

common::Duration FromMilliseconds(const int64 milliseconds) {
  return std::chrono::duration_cast<Duration>(
      std::chrono::milliseconds(milliseconds));
}

double GetThreadCpuTimeSeconds() {
#ifndef WIN32
  struct timespec thread_cpu_time;
  CHECK(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &thread_cpu_time) == 0)
      << std::strerror(errno);
  return thread_cpu_time.tv_sec + 1e-9 * thread_cpu_time.tv_nsec;
#else
  return 0.;
#endif
}

TimeMeasurer::TimeMeasurer(std::string name="Time measurer", bool print_results_on_destruction=false)
    : name_(name),
      print_results_on_destruction_(print_results_on_destruction),
      is_measuring_(false),
      thread_id_(0) {}

void TimeMeasurer::StartMeasurement() {
  if (thread_id_ == 0) {
    thread_id_ = pthread_self();
  } else {
    CHECK_EQ(thread_id_, pthread_self());
  }
  CHECK(!is_measuring_);
  is_measuring_ = true;
  start_time_ = std::chrono::steady_clock::now();
}

void TimeMeasurer::StopMeasurement() {
  auto stop_time = std::chrono::steady_clock::now();
  CHECK_EQ(thread_id_, pthread_self());
  CHECK(is_measuring_);
  double measured_time = ToSeconds(stop_time - start_time_);
  time_measurements_.push_back(measured_time);
  is_measuring_ = false;
}

TimeMeasurer::~TimeMeasurer() {
  if (print_results_on_destruction_) {
    double total_measured_time = 0.;
    double avarage_time = 0.;
    for (auto measured_time : time_measurements_) {
      total_measured_time += measured_time;
    }
    avarage_time = total_measured_time / time_measurements_.size();

    std::string log_string;
    log_string += name_ + ":\n";
    log_string += "    Number of measurements: " + std::to_string(time_measurements_.size()) + "\n";
    log_string += "    Total measured time: " + std::to_string(total_measured_time) + "\n";
    log_string += "    Average time: " + std::to_string(avarage_time) + "\n";

    std::cout << log_string;
  }
}

}  // namespace common
}  // namespace cartographer
