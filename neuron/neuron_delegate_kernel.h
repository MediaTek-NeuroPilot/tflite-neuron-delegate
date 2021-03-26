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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_KERNEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_KERNEL_H_

#include "neuron/APUWareUtilsApi.h"
#include "neuron/neuron_delegate.h"
#include "neuron/neuron_delegate_builder.h"
#include "neuron/neuron_delegate_utils.h"
#include "neuron/neuron_delegate_validation.h"
#include "neuron/neuron_implementation.h"

#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "tensorflow/lite/delegates/utils/simple_delegate.h"

namespace tflite {
namespace neuron {

// The kernel that represents the node sub set of TFLite being run on Neuron
// API.
struct NeuronOpMappingArgs {
  TfLiteContext* context;
  NeuronOpBuilder* builder;
  TfLiteNode* node;
  std::vector<int>* model_state_outputs;
  std::vector<int>* model_state_tfl_inputs;
  std::vector<std::tuple<int, int>>* feedback_loops;
  int* neuron_errno;
};

// Manage Neuron shared memory handle
class NNMemory {
 public:
  NNMemory(const NeuronApi* neuronapi, const char* name, size_t size);

  ~NNMemory();

  uint8_t* get_data_ptr() { return data_ptr_; }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
  int fd_ = 0;
  size_t byte_size_ = 0;
  uint8_t* data_ptr_ = nullptr;
};

// RAII Neuron Model Destructor for use with std::unique_ptr
class NNFreeModel {
 public:
  explicit NNFreeModel(const NeuronApi* neuronapi) : neuronapi_(neuronapi) {}
  void operator()(NeuronModel* model) { neuronapi_->NeuronModel_free(model); }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
};

// RAII Neuron Compilation Destructor for use with std::unique_ptr
class NNFreeCompilation {
 public:
  explicit NNFreeCompilation(const NeuronApi* neuronapi)
      : neuronapi_(neuronapi) {}
  void operator()(NeuronCompilation* compilation) {
    neuronapi_->NeuronCompilation_free(compilation);
  }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
};

// RAII Neuron Execution Destructor for use with std::unique_ptr
class NNFreeExecution {
 public:
  explicit NNFreeExecution(const NeuronApi* neuronapi)
      : neuronapi_(neuronapi) {}
  void operator()(NeuronExecution* execution) {
    neuronapi_->NeuronExecution_free(execution);
  }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
};

// Neuron delegate kernel.
class NeuronDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  static TfLiteStatus Map(TfLiteContext* context, int builtin_code, int version,
                          int neuron_platform_version,
                          const NeuronOpMappingArgs& mapping_args,
                          NeuronOperationType* nn_op_type);

  explicit NeuronDelegateKernel(const NeuronApi* neuronapi,
                                const NeuronDelegateOptions& options)
      : neuronapi_(neuronapi),
        options_(options),
        nn_model_(nullptr, NNFreeModel(neuronapi_)),
        nn_compilation_(nullptr, NNFreeCompilation(neuronapi_)) {}

  ~NeuronDelegateKernel() { releasePerformanceLock(perf_handle_); }

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override;

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override;

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override;

 private:
  const NeuronApi* neuronapi_;
  const NeuronDelegateOptions options_;
  // True if initialization has been completed successfully
  bool initialised_;
  std::unique_ptr<NeuronModel, NNFreeModel> nn_model_;
  std::unique_ptr<NeuronCompilation, NNFreeCompilation> nn_compilation_;
  // Node indices that this delegate is responsible for. Indices here
  // indexes into the nodes array in the TfLiteContext.
  std::vector<int> nodes_;
  // Track indices we use
  OperandMapping operand_mapping_;

  std::vector<int> model_state_outputs_;
  std::vector<int> model_state_tfl_inputs_;
  // This is the equivalent of the pair model_state_outputs_,
  // model_state_tfl_inputs_ for all tensors where we have to keep the output
  // data available for TFLite model users
  std::vector<std::tuple<int, int>> feedback_loops_;

  std::unique_ptr<NNMemory> nn_input_memory_;
  std::unique_ptr<NNMemory> nn_output_memory_;

  void AddDequantizeOperatorsWhereNeeded(const TfLiteContext* context,
                                         int builtin_code,
                                         const TfLiteNode* node,
                                         NeuronOpBuilder* builder);

  TfLiteStatus AddOpsAndTensors(TfLiteContext* context);

  TfLiteStatus BuildGraph(TfLiteContext* context,
                          const TfLiteIntArray* input_tensors,
                          const TfLiteIntArray* output_tensors);
  int perf_handle_ = 0;
};

}  // namespace neuron
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_KERNEL_H_
