/*
 * Copyright (C) 2021 MediaTek Inc., this file is modified on 02/26/2021
 * by MediaTek Inc. based on MIT License .
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the ""Software""), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED ""AS IS"", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_PERF_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_PERF_H_

#include <chrono>
#include <cstdlib>
#include <vector>

#include "neuron/neuron_delegate.h"

namespace tflite {
namespace neuron {

class PerformanceCompilation {
 public:
  static const int kDefaultDuration = 3000;

 public:
  PerformanceCompilation() {}

  virtual ~PerformanceCompilation();

  bool AcquirePerfLock();

 private:
  int handle_;
};

class PerformanceExecution {
 public:
  explicit PerformanceExecution(ExecutionPreference preference)
      : kPreference_(preference) {}

  virtual ~PerformanceExecution();

  bool AcquirePerfLock(uint32_t duration);

 private:
  int handle_;
  std::chrono::steady_clock::time_point fireTime_;
  const ExecutionPreference kPreference_;

  const std::vector<int32_t>& GetParams();
};

}  // namespace neuron
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_PERF_H_
