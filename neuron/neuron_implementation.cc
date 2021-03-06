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

#include "neuron/neuron_implementation.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>

#ifdef __ANDROID__
#include <android/dlext.h>
#include <sys/system_properties.h>
#endif  // __ANDROID__

#define NEURONAPI_LOG(format, ...) fprintf(stderr, format "\n", __VA_ARGS__);

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
  NeuronApi neuron_api = {};
  neuron_api.neuron_sdk_version = 0;

  void* libneuron_adapter = nullptr;
#ifndef __ANDROID__
  libneuron_adapter =
      dlopen("libneuron_adapter.mtk.so", RTLD_LAZY | RTLD_LOCAL);
  if (libneuron_adapter == nullptr) {
    // Try to dlopen Neuron universal SDK
    libneuron_adapter =
        dlopen("libneuronusdk_adapter.mtk.so", RTLD_LAZY | RTLD_LOCAL);
    if (libneuron_adapter == nullptr) {
      NEURONAPI_LOG("NeuronApi error: unable to open library %s/%s",
                    "libneuron_adapter.so", "libneuronusdk_adapter.mtk.so");
    }
#else
  libneuron_adapter =
      dlopen("libneuron_adapter.mtk.so", RTLD_LAZY | RTLD_LOCAL);
  if (libneuron_adapter == nullptr) {
    // Try to dlopen Neuron universal SDK
    libneuron_adapter =
        dlopen("libneuronusdk_adapter.mtk.so", RTLD_LAZY | RTLD_LOCAL);
    if (libneuron_adapter == nullptr) {
      NEURONAPI_LOG("NeuronApi error: unable to open library %s/%s",
                    "libneuron_adapter.so", "libneuronusdk_adapter.mtk.so");
    }
#endif
  }
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
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_setOutput, neuron_api);
  LOAD_FUNCTION(libneuron_adapter, NeuronExecution_compute, neuron_api);

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
  return &neuron_api;
}
