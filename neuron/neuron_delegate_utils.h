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
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED ""AS IS"", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_UTILS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_UTILS_H_

#include "neuron/neuron_types.h"

#include <string>

#include <dlfcn.h>
#include <fcntl.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"

namespace tflite {
namespace neuron {

#define RETURN_TFLITE_ERROR_IF_NEURON_ERROR(context, code, call_desc)         \
  do {                                                                        \
    const auto _code = (code);                                                \
    const auto _call_desc = (call_desc);                                      \
    if (_code != NEURON_NO_ERROR) {                                           \
      const auto error_desc = NeuronErrorDescription(_code);                  \
      context->ReportError(context,                                           \
                           "Neuron returned error %s at line %d while %s.\n", \
                           error_desc.c_str(), __LINE__, _call_desc);         \
      return kTfLiteError;                                                    \
    }                                                                         \
  } while (0)

inline std::string NeuronErrorDescription(int error_code) {
  switch (error_code) {
    case NEURON_NO_ERROR:
      return "NEURON_NO_ERROR";
    case NEURON_OUT_OF_MEMORY:
      return "NEURON_OUT_OF_MEMORY";
    case NEURON_INCOMPLETE:
      return "NEURON_INCOMPLETE";
    case NEURON_UNEXPECTED_NULL:
      return "NEURON_UNEXPECTED_NULL";
    case NEURON_BAD_DATA:
      return "NEURON_BAD_DATA";
    case NEURON_OP_FAILED:
      return "NEURON_OP_FAILED";
    case NEURON_BAD_STATE:
      return "NEURON_BAD_STATE";
    case NEURON_UNMAPPABLE:
      return "NEURON_UNMAPPABLE";
    case NEURON_OUTPUT_INSUFFICIENT_SIZE:
      return "NEURON_OUTPUT_INSUFFICIENT_SIZE";
    case NEURON_UNAVAILABLE_DEVICE:
      return "NEURON_UNAVAILABLE_DEVICE";
    default:
      return "Unknown Neuron error code: " + std::to_string(error_code);
  }
}

inline bool HasZeroes(TfLiteIntArrayView array) {
  for (auto value : array) {
    if (value == 0) {
      return true;
    }
  }
  return false;
}

inline bool IsFloat(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return true;
    default:
      return false;
  }
}

inline bool IsFloatOrUInt8(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
    case kTfLiteUInt8:
      return true;
    default:
      return false;
  }
}

inline bool IsQuantized(TfLiteType type) {
  switch (type) {
    case kTfLiteUInt8:
    case kTfLiteInt8:
      return true;
    default:
      // kTfLiteInt16 isn't supported as quantized type yet.
      return false;
  }
}

typedef int (*perf_lock_acq)(int, int, int[], int);
typedef int (*perf_lock_rel)(int);

inline int acquirePerf() {
  int (*perfLockAcq)(int, int, int[], int) = NULL;
  int (*perfLockRel)(int) = NULL;

  void *handle, *func;
  handle = dlopen("libmtkperf_client.so", RTLD_NOW);
  func = dlsym(handle, "perf_lock_acq");
  perfLockAcq = reinterpret_cast<perf_lock_acq>(func);
  func = dlsym(handle, "perf_lock_rel");
  perfLockRel = reinterpret_cast<perf_lock_rel>(func);

  int perf_lock_rsc[] = {0x01000000, 0};

  int hdl = perfLockAcq(0, 3000, perf_lock_rsc, 2);
  return hdl;
}

inline void releasePerf(int handle) {
}

}  // namespace neuron
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_UTILS_H_
