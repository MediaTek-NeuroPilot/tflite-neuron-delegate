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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_IMPLEMENTATION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_IMPLEMENTATION_H_

#include <android/hardware_buffer.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "neuron/neuron_types.h"

constexpr int32_t kMinSdkVersionForNeuron13 = 30;

struct NeuronApi {
  void* handle;
  bool neuron_exists;
  int32_t android_sdk_version;

  ~NeuronApi() {
    dlclose(handle);
    handle = nullptr;
  }

  // Neuron adapter api function types

  // Get the version of Neuron runtime library.
  int (*Neuron_getVersion)(NeuronRuntimeVersion* version);

  // Get the size of L1 memory in APU.
  int (*Neuron_getL1MemorySizeKb)(uint32_t* sizeKb);

  // Creates a shared memory object from a file descriptor.
  // The shared memory is backed by a file descriptor via mmap.
  int (*NeuronMemory_createFromFd)(size_t size, int protect, int fd,
                                   size_t offset, NeuronMemory** memory);

  // Delete a memory object.
  void (*NeuronMemory_free)(NeuronMemory* memory);

  // Create an empty NeuronModel. The model should be constructed with calls to
  // NeuronModel_addOperation and NeuronModel_addOperand.
  int (*NeuronModel_create)(NeuronModel** model);

  // Destroy a model. The model need not have been finished by a call to
  // NeuronModel_free.
  void (*NeuronModel_free)(NeuronModel* model);

  // Indicate that we have finished modifying a model.
  // Required before calling NeuronCompilation_compile.
  int (*NeuronModel_finish)(NeuronModel* model);

  // Gets the supported operations in a model.
  // This function must be called after calling NeuronModel_finish
  int (*NeuronModel_getSupportedOperations)(NeuronModel* model, bool* supported,
                                            uint32_t operationCount);

  // Add an operand to a model. The order in which the operands are added is
  // important. The first one added to a model will have the index value 0, the
  // second 1, etc. These indexes are used as operand identifiers in
  // NeuronModel_addOperation.
  int (*NeuronModel_addOperand)(NeuronModel* model,
                                const NeuronOperandType* type);

  // Sets an operand to a constant value.
  // For scalar values, the content of buffer is copied into the model.
  // For tensor values, a pointer to the buffer is stored within the model.
  int (*NeuronModel_setOperandValue)(NeuronModel* model, int32_t index,
                                     const void* buffer, size_t length);

  // Sets an operand's per channel quantization parameters
  // Sets parameters required by a tensor of type
  // NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL This function must be called for
  // every tensor of type NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL before calling
  // NeuronModel_finish
  int (*NeuronModel_setOperandSymmPerChannelQuantParams)(
      NeuronModel* model, int32_t index,
      const NeuronSymmPerChannelQuantParams* channelQuant);

  // Add an operation to a model.
  // The operands specified by inputs and outputs must have been previously
  // added by calls to NeuronModel_addOperand.
  int (*NeuronModel_addOperation)(NeuronModel* model, NeuronOperationType type,
                                  uint32_t inputCount, const uint32_t* inputs,
                                  uint32_t outputCount,
                                  const uint32_t* outputs);

  // Specfifies which operands will be the model's inputs and outputs.
  // An operand cannot be used for both input and output. Doing so will return
  // an error.
  int (*NeuronModel_identifyInputsAndOutputs)(NeuronModel* model,
                                              uint32_t inputCount,
                                              const uint32_t* inputs,
                                              uint32_t outputCount,
                                              const uint32_t* outputs);

  // Specifies whether NEURON_TENSOR_FLOAT32 is allowed to be calculated with
  // range and/or precision as low as that of the IEEE 754 16-bit floating-point
  // format. By default, NEURON_TENSOR_FLOAT32 must be calculated using at least
  // the range and precision of the IEEE 754 32-bit floating-point format.
  int (*NeuronModel_relaxComputationFloat32toFloat16)(NeuronModel* model,
                                                      bool allow);

  // Create a NeuronCompilation to compile the given model.
  int (*NeuronCompilation_create)(NeuronModel* model,
                                  NeuronCompilation** compilation);

  // Sets the execution preference associated with this compilation.
  int (*NeuronCompilation_setPreference)(NeuronCompilation* compilation,
                                         int32_t preference);

  // Sets the execution priority associated with this compilation.
  int (*NeuronCompilation_setPriority)(NeuronCompilation* compilation,
                                       int priority);

  // Get the compiled network size of the compilation.
  int (*NeuronCompilation_getCompiledNetworkSize)(
      NeuronCompilation* compilation, size_t* size);

  // Sets compiler optimization hint.
  int (*NeuronCompilation_setOptimizationHint)(NeuronCompilation* compilation,
                                               uint32_t optimizationCode);

  // Destroy a compilation.
  void (*NeuronCompilation_free)(NeuronCompilation* compilation);

  // Compilation is finished once NeuronCompilation_finish is invoked.
  int (*NeuronCompilation_finish)(NeuronCompilation* compilation);

  // Provides optional caching information for faster re-compilation..
  int (*NeuronCompilation_setCaching)(NeuronCompilation* compilation,
                                      const char* cacheDir,
                                      const uint8_t* token);

  // Hint compiler with the size of L1 memory, this value should not be larger
  // than real platform's settings. The user can get the platform's L1 memory
  // size in KB by calling Neuron_getL1MemorySizeKb.
  int (*NeuronCompilation_setL1MemorySizeKb)(NeuronCompilation* compilation,
                                             uint32_t sizeKb);

  // Create a new execution instance by calling the NeuronExecution_create
  // function.
  int (*NeuronExecution_create)(NeuronCompilation* compilation,
                                NeuronExecution** execution);

  // Destroy an execution.
  void (*NeuronExecution_free)(NeuronExecution* execution);

  // Associate a user buffer with an input of the model of the NeuronExecution.
  int (*NeuronExecution_setInput)(NeuronExecution* execution, int32_t index,
                                  const NeuronOperandType* type,
                                  const void* buffer, size_t length);

  // Associate a user buffer with an output of the model of the NeuronExecution.
  int (*NeuronExecution_setOutput)(NeuronExecution* execution, int32_t index,
                                   const NeuronOperandType* type, void* buffer,
                                   size_t length);

  // Associate a user buffer with an input of the model of the NeuronExecution.
  int (*NeuronExecution_setInputFromMemory)(NeuronExecution* execution,
                                            uint32_t index,
                                            const NeuronOperandType* type,
                                            const NeuronMemory* memory,
                                            size_t offset, size_t length);

  // Associate a user buffer with an output of the model of the NeuronExecution.
  int (*NeuronExecution_setOutputFromMemory)(NeuronExecution* execution,
                                             uint32_t index,
                                             const NeuronOperandType* type,
                                             const NeuronMemory* memory,
                                             size_t offset, size_t length);

  // Schedule synchronous evaluation of the execution.
  // Returns once the execution has completed and the outputs are ready to be
  // consumed.
  int (*NeuronExecution_compute)(NeuronExecution* execution);

  int (*NeuronExecution_startComputeWithDependencies)(
      NeuronExecution* execution, const NeuronEvent* const* dependencies,
      uint32_t num_dependencies, uint64_t duration, NeuronEvent** event);

  int (*NeuronExecution_setBoostHint)(NeuronExecution* execution,
                                      uint8_t boostValue);

  int (*NeuronExecution_getOutputOperandRank)(NeuronExecution* execution,
                                              int32_t index, uint32_t* rank);

  int (*NeuronExecution_getOutputOperandDimensions)(NeuronExecution* execution,
                                                    int32_t index,
                                                    uint32_t* dimensions);

  int (*NeuronExecution_setLoopTimeout)(NeuronExecution* execution,
                                        uint64_t duration);

  int (*NeuronCompilation_setOptimizationString)(
      NeuronCompilation* compilation, const char* optimizationString);

  int (*Neuron_getDeviceCount)(uint32_t* numDevices);

  int (*Neuron_getDevice)(uint32_t devIndex, NeuronDevice** device);

  int (*NeuronDevice_getName)(const NeuronDevice* device, const char** name);

  int (*NeuronCompilation_createForDevices)(NeuronModel* model,
                                            const NeuronDevice* const* devices,
                                            uint32_t numDevices,
                                            NeuronCompilation** compilation);

  void (*NeuronEvent_free)(NeuronEvent* event);

  int (*NeuronEvent_wait)(NeuronEvent* event);

  int (*NeuronEvent_createFromSyncFenceFd)(int sync_fence_fd,
                                           NeuronEvent** event);

  int (*NeuronEvent_getSyncFenceFd)(const NeuronEvent* event,
                                    int* sync_fence_fd);

  int (*NeuronDevice_getExtensionSupport)(const char* extensionName,
                                          bool* isExtensionSupported);

  int (*NeuronModel_getExtensionOperandType)(
      NeuronModel* model, const char* extensionName,
      uint16_t operandCodeWithinExtension, int32_t* type);

  int (*NeuronModel_getExtensionOperationType)(
      NeuronModel* model, const char* extensionName,
      uint16_t operationCodeWithinExtension, int32_t* type);

  int (*NeuronModel_setOperandExtensionData)(NeuronModel* model, int32_t index,
                                             const void* data, size_t length);

  // Create a shared memory region
  int (*ASharedMemory_create)(const char* name, size_t size);
  int (*NeuronMemory_createFromAHardwareBuffer)(const AHardwareBuffer* ahwb,
                                                NeuronMemory** memory);
};

/**
 * Load the Neuron implementation from the shared libraries.
 * The NeuronApi structure is filled with all the pointers. If one function
 * doesn't exist, a null pointer is stored.
 */
const NeuronApi* NeuronApiImplementation();

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_IMPLEMENTATION_H_
