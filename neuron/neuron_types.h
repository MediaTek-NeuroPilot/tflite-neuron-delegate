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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_TYPES_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_TYPES_H_

#include <android/hardware_buffer.h>
#include <stdint.h>
#include <stdio.h>

// NeuronModel is an opaque type that contains a description of the mathematical
// operations that constitute the model.
typedef struct NeuronModel NeuronModel;

// NeuronCompilation is an opaque type that can be used to compile a machine
// learning model.
typedef struct NeuronCompilation NeuronCompilation;

// NeuronExecution is an opaque type that can be used to apply a
// machine learning model to a set of inputs.
typedef struct NeuronExecution NeuronExecution;

// NeuronDevice is an opaque type that represents a device. This type is used to
// query basic properties and supported operations of the corresponding device,
// and control which device(s) a model is to be run on.
typedef struct NeuronDevice NeuronDevice;

// NeuronMemory is an opaque type that represents memory. This type is used to
// represent shared memory, memory mapped files, and similar memories. It is the
// application's responsibility to ensure that there are no uses of the memory
// after calling NeuronMemory_free. This includes the execution which references
// this memory because of a call to NeuronExecution_setInputFromMemory or
// NeuronExecution_setOutputFromMemory.
typedef struct NeuronMemory NeuronMemory;

// NeuronEvent is an opaque type that represents an event that will be signaled
// once an execution completes.
typedef struct NeuronEvent NeuronEvent;

// Result codes.
typedef enum {
  NEURON_NO_ERROR = 0,
  NEURON_OUT_OF_MEMORY = 1,
  NEURON_INCOMPLETE = 2,
  NEURON_UNEXPECTED_NULL = 3,
  NEURON_BAD_DATA = 4,
  NEURON_OP_FAILED = 5,
  NEURON_UNMAPPABLE = 6,
  NEURON_BAD_STATE = 7,
  NEURON_BAD_VERSION = 8,
  NEURON_OUTPUT_INSUFFICIENT_SIZE = 9,
  NEURON_UNAVAILABLE_DEVICE = 10,
} ResultCode;

enum { NEURON_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES = 128 };

// Operand types. The type of operands that can be added to a model.
//
// Although we define many types, most operators accept just a few
// types. Most used are NEURON_TENSOR_FLOAT32,
// NEURON_TENSOR_QUANT8_ASYMM, and NEURON_INT32.
enum {
  // The following entries are used to declare scalars.

  // A 32 bit floating point scalar value.
  NEURON_FLOAT32 = 0,
  // A signed 32 bit integer scalar value.
  NEURON_INT32 = 1,
  // An unsigned 32 bit integer scalar value.
  NEURON_UINT32 = 2,

  // The following entries are used to declare tensors.

  // A tensor of 32 bit floating point values.
  NEURON_TENSOR_FLOAT32 = 3,
  // A tensor of 32 bit integer values.
  NEURON_TENSOR_INT32 = 4,
  // A tensor of 8 bit integers that represent real numbers.
  //
  // Attached to this tensor are two numbers that can be used to convert
  // the 8 bit integer to the real value and vice versa.  These two numbers are:
  // - scale: a 32 bit floating point value
  // - zero_value: an 32 bit integer
  //
  // The formula is:
  // real_value = (integer_value - zero_value) * scale.
  //
  NEURON_TENSOR_QUANT8_ASYMM = 5,
  NEURON_BOOL = 6,
  NEURON_TENSOR_QUANT16_SYMM = 7,
  NEURON_TENSOR_FLOAT16 = 8,
  NEURON_TENSOR_BOOL8 = 9,
  NEURON_FLOAT16 = 10,
  NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL = 11,
  NEURON_TENSOR_QUANT16_ASYMM = 12,
  NEURON_TENSOR_QUANT8_SYMM = 13,
  NEURON_TENSOR_QUANT8_ASYMM_SIGNED = 14,
};

// NeuronOperandType describes the type of an operand.
// This structure is used to describe both scalars and tensors.
typedef struct NeuronOperandType {
  // The data type, e.g NEURON_INT8.
  int32_t type;
  // The number of dimensions. It should be 0 for scalars.
  uint32_t dimensionCount;
  // The dimensions of the tensor. It should be nullptr for scalars.
  const uint32_t* dimensions;
  // These two fields are only used for quantized tensors.
  // They should be zero for scalars and non-fixed point tensors.
  // The dequantized value of each entry is (value - zeroPoint) * scale.
  //
  float scale;
  int32_t zeroPoint;
} NeuronOperandType;

/**
 * Parameters for NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL operand.
 */
typedef struct NeuronSymmPerChannelQuantParams {
  /* The index of the channel dimension. */
  uint32_t channelDim;
  /** The size of the scale array. Should be equal to dimension[channelDim] of
   * the Operand. */
  uint32_t scaleCount;
  /** The array of scaling values for each channel. Each value must be greater
   * than zero. */
  const float* scales;
} NeuronSymmPerChannelQuantParams;

// This enum must be consistent with tflite/schema/schema_generated.h.
typedef enum {
  NEURON_ADD = 0,
  NEURON_AVERAGE_POOL_2D = 1,
  NEURON_CONCATENATION = 2,
  NEURON_CONV_2D = 3,
  NEURON_DEPTHWISE_CONV_2D = 4,
  NEURON_DEPTH_TO_SPACE = 5,
  NEURON_DEQUANTIZE = 6,
  NEURON_EMBEDDING_LOOKUP = 7,
  NEURON_FLOOR = 8,
  NEURON_FULLY_CONNECTED = 9,
  NEURON_HASHTABLE_LOOKUP = 10,
  NEURON_L2_NORMALIZATION = 11,
  NEURON_L2_POOL_2D = 12,
  NEURON_LOCAL_RESPONSE_NORMALIZATION = 13,
  NEURON_LOGISTIC = 14,
  NEURON_LSH_PROJECTION = 15,
  NEURON_LSTM = 16,
  NEURON_MAX_POOL_2D = 17,
  NEURON_MUL = 18,
  NEURON_RELU = 19,
  NEURON_RELU1 = 20,
  NEURON_RELU6 = 21,
  NEURON_RESHAPE = 22,
  NEURON_RESIZE_BILINEAR = 23,
  NEURON_RNN = 24,
  NEURON_SOFTMAX = 25,
  NEURON_SPACE_TO_DEPTH = 26,
  NEURON_SVDF = 27,
  NEURON_TANH = 28,
  NEURON_BATCH_TO_SPACE_ND = 29,
  NEURON_DIV = 30,
  NEURON_MEAN = 31,
  NEURON_PAD = 32,
  NEURON_SPACE_TO_BATCH_ND = 33,
  NEURON_SQUEEZE = 34,
  NEURON_STRIDED_SLICE = 35,
  NEURON_SUB = 36,
  NEURON_TRANSPOSE = 37,
  NEURON_ABS = 38,
  NEURON_ARGMAX = 39,
  NEURON_ARGMIN = 40,
  NEURON_AXIS_ALIGNED_BBOX_TRANSFORM = 41,
  NEURON_BIDIRECTIONAL_SEQUENCE_LSTM = 42,
  NEURON_BIDIRECTIONAL_SEQUENCE_RNN = 43,
  NEURON_BOX_WITH_NMS_LIMIT = 44,
  NEURON_CAST = 45,
  NEURON_CHANNEL_SHUFFLE = 46,
  NEURON_DETECTION_POSTPROCESSING = 47,
  NEURON_EQUAL = 48,
  NEURON_EXP = 49,
  NEURON_EXPAND_DIMS = 50,
  NEURON_GATHER = 51,
  NEURON_GENERATE_PROPOSALS = 52,
  NEURON_GREATER = 53,
  NEURON_GREATER_EQUAL = 54,
  NEURON_GROUPED_CONV_2D = 55,
  NEURON_HEATMAP_MAX_KEYPOINT = 56,
  NEURON_INSTANCE_NORMALIZATION = 57,
  NEURON_LESS = 58,
  NEURON_LESS_EQUAL = 59,
  NEURON_LOG = 60,
  NEURON_LOGICAL_AND = 61,
  NEURON_LOGICAL_NOT = 62,
  NEURON_LOGICAL_OR = 63,
  NEURON_LOG_SOFTMAX = 64,
  NEURON_MAXIMUM = 65,
  NEURON_MINIMUM = 66,
  NEURON_NEG = 67,
  NEURON_NOT_EQUAL = 68,
  NEURON_PAD_V2 = 69,
  NEURON_POW = 70,
  NEURON_PRELU = 71,
  NEURON_QUANTIZE = 72,
  NEURON_QUANTIZED_16BIT_LSTM = 73,
  NEURON_RANDOM_MULTINOMIAL = 74,
  NEURON_REDUCE_ALL = 75,
  NEURON_REDUCE_ANY = 76,
  NEURON_REDUCE_MAX = 77,
  NEURON_REDUCE_MIN = 78,
  NEURON_REDUCE_PROD = 79,
  NEURON_REDUCE_SUM = 80,
  NEURON_ROI_ALIGN = 81,
  NEURON_ROI_POOLING = 82,
  NEURON_RSQRT = 83,
  NEURON_SELECT = 84,
  NEURON_SIN = 85,
  NEURON_SLICE = 86,
  NEURON_SPLIT = 87,
  NEURON_SQRT = 88,
  NEURON_TILE = 89,
  NEURON_TOPK_V2 = 90,
  NEURON_TRANSPOSE_CONV_2D = 91,
  NEURON_UNIDIRECTIONAL_SEQUENCE_LSTM = 92,
  NEURON_UNIDIRECTIONAL_SEQUENCE_RNN = 93,
  NEURON_RESIZE_NEAREST_NEIGHBOR = 94,
  NEURON_HARD_SWISH = 99,
  NEURON_NUMBER_OF_OPERATIONS,
  NEURON_OEM_OPERATION = 10000,
} NeuronOperationType;

// Fused activation function types.
typedef enum {
  // NO fused activation function.
  NEURON_FUSED_NONE = 0,
  // Fused ReLU activation function.
  NEURON_FUSED_RELU = 1,
  // Fused ReLU1 activation function.
  NEURON_FUSED_RELU1 = 2,
  // Fused ReLU6 activation function.
  NEURON_FUSED_RELU6 = 3,
} FuseCode;

typedef enum {
  NEURON_PRIORITY_LOW = 90,
  NEURON_PRIORITY_MEDIUM = 100,
  NEURON_PRIORITY_HIGH = 110,
  NEURON_PRIORITY_DEFAULT = NEURON_PRIORITY_MEDIUM,
} PriorityCode;

/**
 * The structure to represent the neuron version.
 */
typedef struct {
  uint8_t major;  ///< major version
  uint8_t minor;  ///< minor version
  uint8_t patch;  ///< patch version
} NeuronRuntimeVersion;

// Neuron adapter api function types

// Get the version of Neuron runtime library.
typedef int (*Neuron_getVersion_fn)(NeuronRuntimeVersion* version);

// Get the size of L1 memory in APU.
typedef int (*Neuron_getL1MemorySizeKb_fn)(uint32_t* sizeKb);

// Creates a shared memory object from a file descriptor.
// The shared memory is backed by a file descriptor via mmap.
typedef int (*NeuronMemory_createFromFd_fn)(size_t size, int protect, int fd,
                                            size_t offset,
                                            NeuronMemory** memory);

// Delete a memory object.
typedef void (*NeuronMemory_free_fn)(NeuronMemory* memory);

// Create an empty NeuronModel. The model should be constructed with calls to
// NeuronModel_addOperation and NeuronModel_addOperand.
typedef int (*NeuronModel_create_fn)(NeuronModel** model);

// Destroy a model. The model need not have been finished by a call to
// NeuronModel_free.
typedef void (*NeuronModel_free_fn)(NeuronModel* model);

// Indicate that we have finished modifying a model.
// Required before calling NeuronCompilation_compile.
typedef int (*NeuronModel_finish_fn)(NeuronModel* model);

// Gets the supported operations in a model.
// This function must be called after calling NeuronModel_finish
typedef int (*NeuronModel_getSupportedOperations_fn)(NeuronModel* model,
                                                     bool* supported,
                                                     uint32_t operationCount);

// Add an operand to a model. The order in which the operands are added is
// important. The first one added to a model will have the index value 0, the
// second 1, etc. These indexes are used as operand identifiers in
// NeuronModel_addOperation.
typedef int (*NeuronModel_addOperand_fn)(NeuronModel* model,
                                         const NeuronOperandType* type);

// Sets an operand to a constant value.
// For scalar values, the content of buffer is copied into the model.
// For tensor values, a pointer to the buffer is stored within the model.
typedef int (*NeuronModel_setOperandValue_fn)(NeuronModel* model, int32_t index,
                                              const void* buffer,
                                              size_t length);

// Sets an operand's per channel quantization parameters
// Sets parameters required by a tensor of type
// NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL This function must be called for
// every tensor of type NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL before calling
// NeuronModel_finish
typedef int (*NeuronModel_setOperandSymmPerChannelQuantParams_fn)(
    NeuronModel* model, int32_t index,
    const NeuronSymmPerChannelQuantParams* channelQuant);

// Add an operation to a model.
// The operands specified by inputs and outputs must have been previously
// added by calls to NeuronModel_addOperand.
typedef int (*NeuronModel_addOperation_fn)(
    NeuronModel* model, NeuronOperationType type, uint32_t inputCount,
    const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs);

// Specifies whether NEURON_TENSOR_FLOAT32 is allowed to be calculated with
// range and/or precision as low as that of the IEEE 754 16-bit floating-point
// format. By default, NEURON_TENSOR_FLOAT32 must be calculated using at least
// the range and precision of the IEEE 754 32-bit floating-point format.
typedef int (*NeuronModel_relaxComputationFloat32toFloat16_fn)(
    NeuronModel* model, bool allow);

// Specfifies which operands will be the model's inputs and outputs.
// An operand cannot be used for both input and output. Doing so will return
// an error.
typedef int (*NeuronModel_identifyInputsAndOutputs_fn)(NeuronModel* model,
                                                       uint32_t inputCount,
                                                       const uint32_t* inputs,
                                                       uint32_t outputCount,
                                                       const uint32_t* outputs);

// Create a NeuronCompilation to compile the given model.
typedef int (*NeuronCompilation_create_fn)(NeuronModel* model,
                                           NeuronCompilation** compilation);

// Sets the execution preference associated with this compilation.
typedef int (*NeuronCompilation_setPreference_fn)(
    NeuronCompilation* compilation, int32_t preference);

// Sets the execution priority associated with this compilation.
typedef int (*NeuronCompilation_setPriority_fn)(NeuronCompilation* compilation,
                                                int32_t priority);
// Get the compiled network size of the compilation.
typedef int (*NeuronCompilation_getCompiledNetworkSize_fn)(
    NeuronCompilation* compilation, size_t* size);

// Sets compiler optimization hint.
typedef int (*NeuronCompilation_setOptimizationHint_fn)(
    NeuronCompilation* compilation, uint32_t optimizationCode);

// Destroy a compilation.
typedef void (*NeuronCompilation_free_fn)(NeuronCompilation* compilation);

// Compilation is finished once NeuronCompilation_finish is invoked.
typedef int (*NeuronCompilation_finish_fn)(NeuronCompilation* compilation);

// Provides optional caching information for faster re-compilation..
typedef int (*NeuronCompilation_setCaching_fn)(NeuronCompilation* compilation,
                                               const char* cacheDir,
                                               const uint8_t* token);

// Hint compiler with the size of L1 memory, this value should not be larger
// than real platform's settings. The user can get the platform's L1 memory size
// in KB by calling Neuron_getL1MemorySizeKb.
typedef int (*NeuronCompilation_setL1MemorySizeKb_fn)(
    NeuronCompilation* compilation, uint32_t sizeKb);

// Create a new execution instance by calling the NeuronExecution_create
// function.
typedef int (*NeuronExecution_create_fn)(NeuronCompilation* compilation,
                                         NeuronExecution** execution);

// Destroy an execution.
typedef void (*NeuronExecution_free_fn)(NeuronExecution* execution);

// Associate a user buffer with an input of the model of the NeuronExecution.
typedef int (*NeuronExecution_setInput_fn)(NeuronExecution* execution,
                                           int32_t index,
                                           const NeuronOperandType* type,
                                           const void* buffer, size_t length);

// Associate a user buffer with an output of the model of the NeuronExecution.
typedef int (*NeuronExecution_setOutput_fn)(NeuronExecution* execution,
                                            int32_t index,
                                            const NeuronOperandType* type,
                                            void* buffer, size_t length);

// Associate part of a memory object with an input of the model of the
// NeuronExecution.
typedef int (*NeuronExecution_setInputFromMemory_fn)(
    NeuronExecution* execution, uint32_t index, const NeuronOperandType* type,
    const NeuronMemory* memory, size_t offset, size_t length);

// Associate part of a memory object with an input of the model of the
// NeuronExecution.
typedef int (*NeuronExecution_setOutputFromMemory_fn)(
    NeuronExecution* execution, uint32_t index, const NeuronOperandType* type,
    const NeuronMemory* memory, size_t offset, size_t length);

// Schedule synchronous evaluation of the execution.
// Returns once the execution has completed and the outputs are ready to be
// consumed.
typedef int (*NeuronExecution_compute_fn)(NeuronExecution* execution);

typedef int (*NeuronExecution_startComputeWithDependencies_fn)(
    NeuronExecution* execution, const NeuronEvent* const* dependencies,
    uint32_t num_dependencies, uint64_t duration, NeuronEvent** event);

typedef int (*NeuronExecution_setBoostHint_fn)(NeuronExecution* execution,
                                               uint8_t boostValue);

typedef int (*NeuronExecution_getOutputOperandRank_fn)(
    NeuronExecution* execution, int32_t index, uint32_t* rank);

typedef int (*NeuronExecution_getOutputOperandDimensions_fn)(
    NeuronExecution* execution, int32_t index, uint32_t* dimensions);

typedef int (*NeuronExecution_setLoopTimeout_fn)(NeuronExecution* execution,
                                                 uint64_t duration);

typedef int (*NeuronCompilation_setOptimizationString_fn)(
    NeuronCompilation* compilation, const char* optimizationString);

typedef int (*Neuron_getDeviceCount_fn)(uint32_t* numDevices);

typedef int (*Neuron_getDevice_fn)(uint32_t devIndex, NeuronDevice** device);

typedef int (*NeuronDevice_getName_fn)(const NeuronDevice* device,
                                       const char** name);

typedef int (*NeuronCompilation_createForDevices_fn)(
    NeuronModel* model, const NeuronDevice* const* devices, uint32_t numDevices,
    NeuronCompilation** compilation);

typedef void (*NeuronEvent_free_fn)(NeuronEvent* event);

typedef int (*NeuronEvent_wait_fn)(NeuronEvent* event);

typedef int (*NeuronEvent_createFromSyncFenceFd_fn)(int sync_fence_fd,
                                                    NeuronEvent** event);

typedef int (*NeuronEvent_getSyncFenceFd_fn)(const NeuronEvent* event,
                                             int* sync_fence_fd);

typedef int (*NeuronDevice_getExtensionSupport_fn)(const char* extensionName,
                                                   bool* isExtensionSupported);

typedef int (*NeuronModel_getExtensionOperandType_fn)(
    NeuronModel* model, const char* extensionName,
    uint16_t operandCodeWithinExtension, int32_t* type);

typedef int (*NeuronModel_getExtensionOperationType_fn)(
    NeuronModel* model, const char* extensionName,
    uint16_t operationCodeWithinExtension, int32_t* type);

typedef int (*NeuronModel_setOperandExtensionData_fn)(NeuronModel* model,
                                                      int32_t index,
                                                      const void* data,
                                                      size_t length);

// Create a shared memory region
typedef int (*ASharedMemory_create_fn)(const char* name, size_t size);

typedef int (*NeuronMemory_createFromAHardwareBuffer_fn)(
    const AHardwareBuffer* ahwb, NeuronMemory** memory);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_TYPES_H_
