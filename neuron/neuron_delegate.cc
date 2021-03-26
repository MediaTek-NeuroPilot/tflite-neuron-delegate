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

#include "neuron/neuron_delegate.h"

#include <utility>

#include "neuron/neuron_delegate_kernel.h"
#include "neuron/neuron_delegate_validation.h"
#include "neuron/neuron_implementation.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace neuron {

// NeuronDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class NeuronDelegate : public SimpleDelegateInterface {
 public:
  explicit NeuronDelegate(const NeuronDelegateOptions& options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    if (!neuron_->neuron_exists) {
      return false;
    }
    std::vector<NeuronValidationFailure> failure;
    bool supported = Validate(registration, node, context, &failure);
    if (!supported) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, "OP %d is not supported(%s)",
                      registration->builtin_code, failure[0].message.c_str());
    }
    return supported;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override {
    return neuron_->neuron_exists ? kTfLiteOk : kTfLiteError;
  }

  const char* Name() const override {
    static constexpr char kName[] = "TFLiteNeuronDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<NeuronDelegateKernel>(neuron_, options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const NeuronApi* neuron_ = NeuronApiImplementation();
  const NeuronDelegateOptions options_;
};

}  // namespace neuron
}  // namespace tflite

NeuronDelegateOptions TfLiteNeuronDelegateOptionsDefault() {
  NeuronDelegateOptions options = {
      // execution_preference = kFastSingleAnswer
      kFastSingleAnswer,
      // Default execution_priority = kPriorityHigh
      kPriorityHigh,
      // Default optimization_hint = kOptimizationDefault
      0,
      // Default allow_fp16 = false
      false,
      // Default boost_duration = 0
      0,
      // const char* cache_dir;
      nullptr,
      // const char* model_token;
      nullptr,
  };

  // Return default options
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteNeuronDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteNeuronDelegateCreate(
    const NeuronDelegateOptions* options) {
  std::unique_ptr<tflite::neuron::NeuronDelegate> neuron(
      new tflite::neuron::NeuronDelegate(
          options ? *options : TfLiteNeuronDelegateOptionsDefault()));

  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(neuron));
}

// Destroys a delegate created with `TfLiteNeuronDelegateCreate` call.
void TfLiteNeuronDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
