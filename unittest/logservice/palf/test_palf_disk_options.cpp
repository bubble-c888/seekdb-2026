/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include "logservice/palf/palf_env_impl.h"

namespace oceanbase
{
namespace palf
{
using namespace common;

class TestPalfDiskOptions : public ::testing::Test
{
protected:
  using Status = PalfDiskOptionsWrapper::Status;
  static constexpr int64_t GIB = 1024LL * 1024 * 1024;

  static PalfDiskOptions options(const bool second, const bool shrink = false)
  {
    PalfDiskOptions result;
    result.log_disk_usage_limit_size_ = second && shrink ? 4 * GIB : 8 * GIB;
    result.log_disk_utilization_threshold_ = second ? 60 : 50;
    result.log_disk_utilization_limit_threshold_ = second ? 95 : 80;
    result.log_disk_throttling_percentage_ = second ? 70 : 60;
    result.log_disk_throttling_maximum_duration_ = second ? 3000000 : 2000000;
    return result;
  }

  static PalfDiskOptions shrinking_stop(const PalfDiskOptions &old_opts,
                                        const PalfDiskOptions &new_opts)
  {
    PalfDiskOptions result = old_opts;
    result.log_disk_throttling_percentage_ = new_opts.log_disk_throttling_percentage_;
    result.log_disk_throttling_maximum_duration_ = new_opts.log_disk_throttling_maximum_duration_;
    return result;
  }

  static void expect_options(const PalfDiskOptions &expected, const PalfDiskOptions &actual)
  {
    EXPECT_EQ(expected.log_disk_usage_limit_size_, actual.log_disk_usage_limit_size_);
    EXPECT_EQ(expected.log_disk_utilization_threshold_, actual.log_disk_utilization_threshold_);
    EXPECT_EQ(expected.log_disk_utilization_limit_threshold_, actual.log_disk_utilization_limit_threshold_);
    EXPECT_EQ(expected.log_disk_throttling_percentage_, actual.log_disk_throttling_percentage_);
    EXPECT_EQ(expected.log_disk_throttling_maximum_duration_, actual.log_disk_throttling_maximum_duration_);
  }

  void expect_state(PalfDiskOptionsWrapper &wrapper, const PalfDiskOptions &stop,
                    const PalfDiskOptions &recycle, const Status status, const int64_t sequence)
  {
    PalfDiskOptions actual_stop;
    PalfDiskOptions actual_recycle;
    Status actual_status;
    int64_t actual_sequence = -1;
    wrapper.get_disk_opts(actual_stop, actual_recycle, actual_status, actual_sequence);
    expect_options(stop, actual_stop);
    expect_options(recycle, actual_recycle);
    EXPECT_EQ(status, actual_status);
    EXPECT_EQ(sequence, actual_sequence);
    EXPECT_EQ(status == Status::SHRINKING_STATUS, wrapper.is_shrinking());
  }

  void concurrent_snapshots(const bool shrink)
  {
    const PalfDiskOptions first = options(false);
    const PalfDiskOptions second = options(true, shrink);
    const PalfDiskOptions intermediate = shrinking_stop(first, second);
    ASSERT_TRUE(first.is_valid());
    ASSERT_TRUE(second.is_valid());
    ASSERT_TRUE(intermediate.is_valid());
    PalfDiskOptionsWrapper wrapper;
    ASSERT_EQ(OB_SUCCESS, wrapper.init(first));

    constexpr int READERS = 3;
    constexpr int ITERATIONS = 10000;
    std::mutex mutex;
    std::condition_variable condition;
    int ready = 0;
    bool start = false;
    auto wait_for_start = [&]() {
      std::unique_lock<std::mutex> guard(mutex);
      ++ready;
      condition.notify_all();
      condition.wait(guard, [&]() { return start; });
    };
    int update_error = OB_SUCCESS;
    int bad_iteration[READERS] = {-1, -1, -1};
    int bad_stage[READERS] = {0, 0, 0};
    PalfDiskOptions bad_options[READERS];
    std::thread readers[READERS];
    for (int i = 0; i < READERS; ++i) {
      readers[i] = std::thread([&, i]() {
        wait_for_start();
        int64_t previous_sequence = 0;
        for (int j = 0; j < ITERATIONS; ++j) {
          const PalfDiskOptions stop = wrapper.get_disk_opts_for_stopping_writing();
          const PalfDiskOptions recycle = wrapper.get_disk_opts_for_recycling_blocks();
          // Separate getter calls need not refer to the same generation.
          if (!stop.is_valid() || !(stop == first || stop == second || (shrink && stop == intermediate))) {
            bad_iteration[i] = j;
            bad_stage[i] = 1;
            bad_options[i] = stop;
            break;
          } else if (!recycle.is_valid() || !(recycle == first || recycle == second)) {
            bad_iteration[i] = j;
            bad_stage[i] = 2;
            bad_options[i] = recycle;
            break;
          }
          PalfDiskOptions pair_stop;
          PalfDiskOptions pair_recycle;
          Status status;
          int64_t sequence = -1;
          wrapper.get_disk_opts(pair_stop, pair_recycle, status, sequence);
          const bool normal = status == Status::NORMAL_STATUS
              && pair_stop == pair_recycle && (pair_stop == first || pair_stop == second);
          const bool shrinking = shrink && status == Status::SHRINKING_STATUS
              && pair_stop == intermediate && pair_recycle == second;
          const bool matching_generation = sequence >= 0 && sequence <= ITERATIONS
              && pair_recycle == (sequence % 2 == 0 ? first : second)
              && (sequence % 2 != 0 || status == Status::NORMAL_STATUS);
          if ((!normal && !shrinking) || !matching_generation || sequence < previous_sequence) {
            bad_iteration[i] = j;
            bad_stage[i] = 3;
            bad_options[i] = pair_stop;
            break;
          }
          previous_sequence = sequence;
        }
      });
    }
    std::thread writer([&]() {
      wait_for_start();
      for (int i = 0; i < ITERATIONS; ++i) {
        const PalfDiskOptions &next = i % 2 == 0 ? second : first;
        const int ret = wrapper.update_disk_options(next);
        if (ret != OB_SUCCESS) {
          update_error = ret;
          break;
        }
        if (shrink && i % 2 == 0) {
          PalfDiskOptions stop;
          PalfDiskOptions recycle;
          Status status;
          int64_t sequence = -1;
          wrapper.get_disk_opts(stop, recycle, status, sequence);
          wrapper.change_to_normal(sequence);
        }
      }
    });
    {
      std::unique_lock<std::mutex> guard(mutex);
      condition.wait(guard, [&]() { return ready == READERS + 1; });
      start = true;
    }
    condition.notify_all();
    writer.join();
    for (int i = 0; i < READERS; ++i) {
      readers[i].join();
      EXPECT_EQ(-1, bad_iteration[i]) << "reader=" << i << " stage=" << bad_stage[i]
          << " size=" << bad_options[i].log_disk_usage_limit_size_
          << " recycle=" << bad_options[i].log_disk_utilization_threshold_
          << " stop=" << bad_options[i].log_disk_utilization_limit_threshold_
          << " throttle=" << bad_options[i].log_disk_throttling_percentage_
          << " duration=" << bad_options[i].log_disk_throttling_maximum_duration_;
    }
    EXPECT_EQ(OB_SUCCESS, update_error);
    expect_state(wrapper, first, first, Status::NORMAL_STATUS, ITERATIONS);
  }

  void snapshots_survive_writer_handoff(const bool shrink)
  {
    const PalfDiskOptions first = options(false);
    const PalfDiskOptions second = options(true, shrink);
    PalfDiskOptionsWrapper wrapper;
    ASSERT_EQ(OB_SUCCESS, wrapper.init(first));
    std::mutex mutex;
    std::condition_variable condition;
    bool snapshots_taken = false;
    int update_error = OB_SUCCESS;
    std::thread writer([&]() {
      {
        std::unique_lock<std::mutex> guard(mutex);
        condition.wait(guard, [&]() { return snapshots_taken; });
      }
      update_error = wrapper.update_disk_options(second);
    });
    const PalfDiskOptions &stop = wrapper.get_disk_opts_for_stopping_writing();
    const PalfDiskOptions &recycle = wrapper.get_disk_opts_for_recycling_blocks();
    {
      std::lock_guard<std::mutex> guard(mutex);
      snapshots_taken = true;
    }
    condition.notify_one();
    writer.join();
    ASSERT_EQ(OB_SUCCESS, update_error);
    // The join also makes this a race-free regression test with the old API.
    expect_options(first, stop);
    expect_options(first, recycle);
    expect_state(wrapper, shrink ? shrinking_stop(first, second) : second, second,
                 shrink ? Status::SHRINKING_STATUS : Status::NORMAL_STATUS, 1);
    if (shrink) {
      wrapper.change_to_normal(1);
      expect_state(wrapper, second, second, Status::NORMAL_STATUS, 1);
      expect_options(first, stop);
      expect_options(first, recycle);
    }
  }
};

TEST_F(TestPalfDiskOptions, default_state)
{
  PalfDiskOptionsWrapper wrapper;
  PalfDiskOptions invalid;
  expect_state(wrapper, invalid, invalid, Status::INVALID_STATUS, -1);
  EXPECT_FALSE(wrapper.get_disk_opts_for_stopping_writing().is_valid());
  EXPECT_FALSE(wrapper.get_disk_opts_for_recycling_blocks().is_valid());
}

TEST_F(TestPalfDiskOptions, normal_initialization)
{
  PalfDiskOptionsWrapper wrapper;
  const PalfDiskOptions initial = options(false);
  ASSERT_TRUE(initial.is_valid());
  ASSERT_EQ(OB_SUCCESS, wrapper.init(initial));
  expect_options(initial, wrapper.get_disk_opts_for_stopping_writing());
  expect_options(initial, wrapper.get_disk_opts_for_recycling_blocks());
  expect_state(wrapper, initial, initial, Status::NORMAL_STATUS, 0);
  RecordProperty("disk_options_size", static_cast<int>(sizeof(PalfDiskOptions)));
}

TEST_F(TestPalfDiskOptions, invalid_initialization)
{
  PalfDiskOptionsWrapper wrapper;
  PalfDiskOptions invalid;
  ASSERT_FALSE(invalid.is_valid());
  EXPECT_EQ(OB_INVALID_ARGUMENT, wrapper.init(invalid));
  expect_state(wrapper, invalid, invalid, Status::INVALID_STATUS, -1);
}

TEST_F(TestPalfDiskOptions, getters_return_values)
{
  using Wrapper = const PalfDiskOptionsWrapper &;
  EXPECT_TRUE((std::is_same<PalfDiskOptions,
      decltype(std::declval<Wrapper>().get_disk_opts_for_stopping_writing())>::value));
  EXPECT_TRUE((std::is_same<PalfDiskOptions,
      decltype(std::declval<Wrapper>().get_disk_opts_for_recycling_blocks())>::value));
}

TEST_F(TestPalfDiskOptions, normal_update_and_noop)
{
  PalfDiskOptionsWrapper wrapper;
  const PalfDiskOptions first = options(false);
  const PalfDiskOptions second = options(true);
  ASSERT_TRUE(second.is_valid());
  ASSERT_EQ(OB_SUCCESS, wrapper.init(first));
  ASSERT_EQ(OB_SUCCESS, wrapper.update_disk_options(second));
  expect_state(wrapper, second, second, Status::NORMAL_STATUS, 1);
  ASSERT_EQ(OB_SUCCESS, wrapper.update_disk_options(second));
  expect_state(wrapper, second, second, Status::NORMAL_STATUS, 1);
}

TEST_F(TestPalfDiskOptions, invalid_update_preserves_state)
{
  PalfDiskOptionsWrapper wrapper;
  const PalfDiskOptions first = options(false);
  ASSERT_EQ(OB_SUCCESS, wrapper.init(first));
  PalfDiskOptions invalid = first;
  invalid.log_disk_utilization_limit_threshold_ = invalid.log_disk_utilization_threshold_;
  ASSERT_FALSE(invalid.is_valid());
  EXPECT_EQ(OB_INVALID_ARGUMENT, wrapper.update_disk_options(invalid));
  expect_state(wrapper, first, first, Status::NORMAL_STATUS, 0);
}

TEST_F(TestPalfDiskOptions, owned_snapshots_survive_updates)
{
  PalfDiskOptionsWrapper wrapper;
  const PalfDiskOptions first = options(false);
  const PalfDiskOptions second = options(true);
  ASSERT_EQ(OB_SUCCESS, wrapper.init(first));
  const PalfDiskOptions stop = wrapper.get_disk_opts_for_stopping_writing();
  const PalfDiskOptions recycle = wrapper.get_disk_opts_for_recycling_blocks();
  ASSERT_EQ(OB_SUCCESS, wrapper.update_disk_options(second));
  expect_options(first, stop);
  expect_options(first, recycle);
  expect_options(second, wrapper.get_disk_opts_for_stopping_writing());
  expect_options(second, wrapper.get_disk_opts_for_recycling_blocks());
}

TEST_F(TestPalfDiskOptions, stopping_reference_binds_an_independent_snapshot)
{
  PalfDiskOptionsWrapper wrapper;
  const PalfDiskOptions first = options(false);
  ASSERT_EQ(OB_SUCCESS, wrapper.init(first));
  const PalfDiskOptions &snapshot = wrapper.get_disk_opts_for_stopping_writing();
  ASSERT_EQ(OB_SUCCESS, wrapper.update_disk_options(options(true)));
  expect_options(first, snapshot);
}

TEST_F(TestPalfDiskOptions, recycling_reference_binds_an_independent_snapshot)
{
  PalfDiskOptionsWrapper wrapper;
  const PalfDiskOptions first = options(false);
  ASSERT_EQ(OB_SUCCESS, wrapper.init(first));
  const PalfDiskOptions &snapshot = wrapper.get_disk_opts_for_recycling_blocks();
  ASSERT_EQ(OB_SUCCESS, wrapper.update_disk_options(options(true)));
  expect_options(first, snapshot);
}

TEST_F(TestPalfDiskOptions, auto_reference_binds_an_independent_snapshot)
{
  PalfDiskOptionsWrapper wrapper;
  const PalfDiskOptions first = options(false);
  ASSERT_EQ(OB_SUCCESS, wrapper.init(first));
  auto &&snapshot = wrapper.get_disk_opts_for_recycling_blocks();
  ASSERT_EQ(OB_SUCCESS, wrapper.update_disk_options(options(true)));
  expect_options(first, snapshot);
}

TEST_F(TestPalfDiskOptions, shrinking_and_normal_transition)
{
  PalfDiskOptionsWrapper wrapper;
  const PalfDiskOptions first = options(false);
  const PalfDiskOptions second = options(true, true);
  ASSERT_TRUE(second.is_valid());
  ASSERT_EQ(OB_SUCCESS, wrapper.init(first));
  ASSERT_EQ(OB_SUCCESS, wrapper.update_disk_options(second));
  const PalfDiskOptions expected_stop = shrinking_stop(first, second);
  expect_state(wrapper, expected_stop, second, Status::SHRINKING_STATUS, 1);
  expect_options(expected_stop, wrapper.get_disk_opts_for_stopping_writing());
  expect_options(second, wrapper.get_disk_opts_for_recycling_blocks());
  const PalfDiskOptions &before_normal = wrapper.get_disk_opts_for_stopping_writing();
  wrapper.change_to_normal(1);
  expect_state(wrapper, second, second, Status::NORMAL_STATUS, 1);
  expect_options(expected_stop, before_normal);
}

TEST_F(TestPalfDiskOptions, stale_sequence_cannot_complete_a_new_shrink)
{
  PalfDiskOptionsWrapper wrapper;
  const PalfDiskOptions first = options(false);
  const PalfDiskOptions second = options(true, true);
  PalfDiskOptions third = second;
  third.log_disk_usage_limit_size_ = 2 * GIB;
  ASSERT_TRUE(third.is_valid());
  ASSERT_EQ(OB_SUCCESS, wrapper.init(first));
  ASSERT_EQ(OB_SUCCESS, wrapper.update_disk_options(second));
  ASSERT_EQ(OB_SUCCESS, wrapper.update_disk_options(third));
  wrapper.change_to_normal(1);
  expect_state(wrapper, shrinking_stop(first, third), third, Status::SHRINKING_STATUS, 2);
  wrapper.change_to_normal(2);
  expect_state(wrapper, third, third, Status::NORMAL_STATUS, 2);
}

TEST_F(TestPalfDiskOptions, explicit_stop_options_preserve_recycling_state)
{
  PalfDiskOptionsWrapper wrapper;
  const PalfDiskOptions first = options(false);
  const PalfDiskOptions second = options(true);
  ASSERT_EQ(OB_SUCCESS, wrapper.init(first));
  wrapper.set_stop_writing_disk_options_with_status(second, Status::SHRINKING_STATUS);
  expect_state(wrapper, second, first, Status::SHRINKING_STATUS, 0);
  expect_options(second, wrapper.get_disk_opts_for_stopping_writing());
  expect_options(first, wrapper.get_disk_opts_for_recycling_blocks());
}

TEST_F(TestPalfDiskOptions, reset_and_reinitialize)
{
  PalfDiskOptionsWrapper wrapper;
  const PalfDiskOptions first = options(false);
  ASSERT_EQ(OB_SUCCESS, wrapper.init(first));
  const PalfDiskOptions &snapshot = wrapper.get_disk_opts_for_stopping_writing();
  wrapper.reset();
  PalfDiskOptions invalid;
  expect_state(wrapper, invalid, invalid, Status::INVALID_STATUS, -1);
  expect_options(first, snapshot);
  const PalfDiskOptions second = options(true);
  ASSERT_EQ(OB_SUCCESS, wrapper.init(second));
  expect_state(wrapper, second, second, Status::NORMAL_STATUS, 0);
}

TEST_F(TestPalfDiskOptions, concurrent_normal_snapshots)
{
  concurrent_snapshots(false);
}

TEST_F(TestPalfDiskOptions, snapshots_survive_normal_writer_handoff)
{
  snapshots_survive_writer_handoff(false);
}

TEST_F(TestPalfDiskOptions, snapshots_survive_shrinking_writer_handoff)
{
  snapshots_survive_writer_handoff(true);
}

TEST_F(TestPalfDiskOptions, concurrent_shrinking_snapshots)
{
  concurrent_snapshots(true);
}

} // namespace palf
} // namespace oceanbase

int main(int argc, char **argv)
{
  OB_LOGGER.set_log_level("WARN");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
