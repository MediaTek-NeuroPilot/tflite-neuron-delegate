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

#include "neuron/neuron_implementation.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <string>

#ifdef __ANDROID__
#include <android/dlext.h>
#include <android/log.h>
#include <sys/system_properties.h>
#endif  // __ANDROID__

#define NEURONAPI_LOG(format, ...) Log(format, ##__VA_ARGS__);
#define RETURN_NEURON_ERROR(code, call_desc)                       \
  do {                                                             \
    const auto _code = (code);                                     \
    const auto _call_desc = (call_desc);                           \
    if (_code != 0) {                                              \
      NEURONAPI_LOG("Neuron returned error %d while %s.\n", _code, \
                    _call_desc);                                   \
    }                                                              \
  } while (0)

void Log(const char* format, ...) {
  va_list args;
  va_start(args, format);

#ifdef __ANDROID__
  // First log to Android's explicit log(cat) API.
  va_list args_copy;
  va_copy(args_copy, args);
  __android_log_vprint(ANDROID_LOG_INFO, "tflite", format, args_copy);
  va_end(args_copy);
#endif

  // Also print to stderr for standard console applications.
  fprintf(stderr, "%s: ", "INFO");
  va_copy(args_copy, args);
  vfprintf(stderr, format, args_copy);
  va_end(args_copy);
  fputc('\n', stderr);

  va_end(args);
}

/// M: NeuroPilot {@
static int32_t GetAndroidSdkVersion() {
#ifdef __ANDROID__
  const char* debugSdkProp = "debug.mtk.build.version.sdk";
  const char* sdkProp = "ro.build.version.sdk";
  char sdkVersion[PROP_VALUE_MAX];
  int length = __system_property_get(debugSdkProp, sdkVersion);
  if (length != 0) {
    return 29;
  }
  length = __system_property_get(sdkProp, sdkVersion);
  if (length != 0) {
    for (int i = 0; i < length; ++i) {
      int digit = sdkVersion[i] - '0';
      if (digit < 0 || digit > 9) {
        // Non-numeric SDK version, assume it's higher then expected;
        return 0xFFFF;
      }
    }
    return atoi(sdkVersion);
  }
  NEURONAPI_LOG("No %s prop", sdkProp);
  return 0;
#endif  // __ANDROID__
  return 0;
}
/// M: NeuroPilot @}

void* LoadFunction(void* handle, const char* name, bool optional) {
  if (handle == nullptr) {
    return nullptr;
  }
  void* fn = dlsym(handle, name);
  if (fn == nullptr && !optional) {
    NEURONAPI_LOG("neuron error: unable to open function %s", name);
  }
  return fn;
}

#ifndef __ANDROID__
// Add /dev/shm implementation of shared memory for non-Android platforms
int ASharedMemory_create(const char* name, size_t size) {
  int fd = shm_open(name, O_RDWR | O_CREAT, 0644);
  if (fd < 0) {
    return fd;
  }
  int result = ftruncate(fd, size);
  if (result < 0) {
    close(fd);
    return -1;
  }
  return fd;
}
#endif  // __ANDROID__

#define LOAD_FUNCTION(handle, name, neuronapi_obj)  \
  neuronapi_obj.name = reinterpret_cast<name##_fn>( \
      LoadFunction(handle, #name, /*optional*/ false));

#define LOAD_FUNCTION_OPTIONAL(handle, name, neuronapi_obj) \
  neuronapi_obj.name = reinterpret_cast<name##_fn>(         \
      LoadFunction(handle, #name, /*optional*/ true));

#define LOAD_FUNCTION_RENAME(handle, name, symbol, neuronapi_obj) \
  neuronapi_obj.name = reinterpret_cast<name##_fn>(               \
      LoadFunction(handle, symbol, /*optional*/ false));

const NeuronApi LoadNeuronApi() {
  static const std::string libraries[] = {
      "libneuronusdk_adapter.mtk.so.5", "libneuronusdk_adapter.mtk.so",
      /*"libneuron_adapter_mgvi.so", "libneuron_adapter.so"*/};
  NeuronApi neuron_api = {};
  neuron_api.android_sdk_version = GetAndroidSdkVersion();
  NEURONAPI_LOG("Android SDK version: %d", neuron_api.android_sdk_version);
  if (neuron_api.android_sdk_version <= kMinSdkVersionForNeuron13) {
    neuron_api.neuron_exists = false;
    neuron_api.handle = nullptr;
    return neuron_api;
  }

  void* libneuron_adapter = nullptr;
  for (const auto& lib : libraries) {
    libneuron_adapter = dlopen(lib.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (libneuron_adapter) {
      NEURONAPI_LOG("dlopen %s", lib.c_str());
      break;
    }
  }

  neuron_api.handle = libneuron_adapter;
  neuron_api.neuron_exists = libneuron_adapter != nullptr;

  LOAD_FUNCTION(libneuron_adapter, Neuron_getVersion, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, Neuron_getL1MemorySizeKb, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_create, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_free, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_finish, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_getSupportedOperations,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_addOperand, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_setOperandValue, neuron_api);
  LOAD_FUNCTION(libneuron_adapter,
                NeuronModel_setOperandSymmPerChannelQuantParams, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_addOperation, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_identifyInputsAndOutputs,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_relaxComputationFloat32toFloat16,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_create, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_create, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_setPreference, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_setPriority, neuron_api);
  LOAD_FUNCTION_OPTIONAL(libneuron_adapter,
                         NeuronCompilation_getCompiledNetworkSize, neuron_api);
  LOAD_FUNCTION_OPTIONAL(libneuron_adapter,
                         NeuronCompilation_setOptimizationHint, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_free, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_finish, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_setCaching, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_setL1MemorySizeKb,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_free, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_setInput, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_setInputFromMemory,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_setOutput, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_setOutputFromMemory,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_compute, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_startComputeWithDependencies,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_setBoostHint, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_getOutputOperandRank,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_getOutputOperandDimensions,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_setLoopTimeout, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_setOptimizationString,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, Neuron_getDeviceCount, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, Neuron_getDevice, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronDevice_getName, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronCompilation_createForDevices,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronEvent_free, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronEvent_wait, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronEvent_createFromSyncFenceFd,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronEvent_getSyncFenceFd, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronMemory_createFromAHardwareBuffer,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronMemory_free, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronDevice_getExtensionSupport,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_getExtensionOperandType,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_getExtensionOperationType,
                neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronModel_setOperandExtensionData,
                neuron_api);

  uint32_t num_devices = 0;
  if (libneuron_adapter != nullptr) {
    neuron_api.Neuron_getDeviceCount(&num_devices);
  }
  if (num_devices == 0) {
    neuron_api.neuron_exists = false;
  } else if (neuron_api.neuron_exists) {
    // check if mdla/dsp supported
    neuron_api.neuron_exists = false;
    for (uint32_t i = 0; i < num_devices; ++i) {
      const char* name = nullptr;
      NeuronDevice* device = nullptr;
      RETURN_NEURON_ERROR(neuron_api.Neuron_getDevice(i, &device),
                          "getting device");
      RETURN_NEURON_ERROR(neuron_api.NeuronDevice_getName(device, &name),
                          "getting device name");
      // TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Got device name: %s", name);
      if (name != nullptr && (strncmp(name, "mtk-dsp", 7) == 0 ||
                              strncmp(name, "mtk-mdla", 8) == 0)) {
        neuron_api.neuron_exists = true;
        break;
      }
    }
  }
  if (!neuron_api.neuron_exists) {
    NEURONAPI_LOG("neuron error: Neuron adapter API exists: %d",
                  neuron_api.neuron_exists);
  }

  // ASharedMemory_create has different implementations in Android depending on
  // the partition. Generally it can be loaded from libandroid.so but in vendor
  // partition (e.g. if a HAL wants to use Neuron) it is only accessible through
  // libcutils.
#ifdef __ANDROID__
  void* libandroid = nullptr;
  libandroid = dlopen("libandroid.so", RTLD_LAZY | RTLD_LOCAL);
  if (libandroid != nullptr) {
    LOAD_FUNCTION(libandroid, ASharedMemory_create, neuron_api);
  } else {
    void* cutils_handle = dlopen("libcutils.so", RTLD_LAZY | RTLD_LOCAL);
    if (cutils_handle != nullptr) {
      LOAD_FUNCTION_RENAME(cutils_handle, ASharedMemory_create,
                           "ashmem_create_region", neuron_api);
    } else {
      NEURONAPI_LOG("neuron error: unable to open neither libraries %s and %s",
                    "libandroid.so", "libcutils.so");
    }
  }
#else
  if (libneuron_adapter != nullptr) {
    neuron_api.ASharedMemory_create = ASharedMemory_create;
  }
#endif  // __ANDROID__

  return neuron_api;
}

const NeuronApi* NeuronApiImplementation() {
  static const NeuronApi neuron_api = LoadNeuronApi();
  if (neuron_api.neuron_exists) {
    NeuronRuntimeVersion version;
    if (neuron_api.Neuron_getVersion(&version) == NEURON_NO_ERROR) {
      NEURONAPI_LOG(
          "NeuronApi version: %d.%d.%d", static_cast<int>(version.major),
          static_cast<int>(version.minor), static_cast<int>(version.patch));
    } else {
      NEURONAPI_LOG("NeuronApi error: fail to get version by %s",
                    "Neuron_getVersion");
    }
  }

  return &neuron_api;
}
