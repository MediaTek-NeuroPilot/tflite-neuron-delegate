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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NNAPI_KERNEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NNAPI_KERNEL_H_

#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_kernel.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace neuron {

class NeuronNNAPIDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit NeuronNNAPIDelegateKernel(StatefulNnApiDelegate::Options options =
                                         StatefulNnApiDelegate::Options()) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                    "NeuronNNAPIDelegateKernel ctor()");
    nnapi_delegate_.reset(new StatefulNnApiDelegate(options));
    nnapi_delegate_kernel_.reset(
        new NNAPIDelegateKernel(NnApiImplementation()));
  }

  ~NeuronNNAPIDelegateKernel() {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                    "NeuronNNAPIDelegateKernel dtor()");
    nnapi_delegate_.reset(nullptr);
    nnapi_delegate_kernel_.reset(nullptr);
  }

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    int nnapi_error = 0;
    TfLiteDelegateParams new_params;
    // Replace the delegate with StatefulNnApiDelegate
    new_params.delegate = nnapi_delegate_.get();
    new_params.nodes_to_replace = params->nodes_to_replace;
    new_params.input_tensors = params->input_tensors;
    new_params.output_tensors = params->output_tensors;
    return nnapi_delegate_kernel_->Init(context, &new_params, &nnapi_error);
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    int nnapi_error = 0;
    // Replace the delegate with StatefulNnApiDelegate
    node->delegate = nnapi_delegate_.get();
    return nnapi_delegate_kernel_->Prepare(context, node, &nnapi_error);
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    int nnapi_error = 0;
    // Replace the delegate with StatefulNnApiDelegate
    node->delegate = nnapi_delegate_.get();
    return nnapi_delegate_kernel_->Invoke(context, node, &nnapi_error);
  }

 private:
  std::unique_ptr<StatefulNnApiDelegate> nnapi_delegate_;
  std::unique_ptr<delegate::nnapi::NNAPIDelegateKernel> nnapi_delegate_kernel_;
};

}  // namespace neuron
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NNAPI_KERNEL_H_
