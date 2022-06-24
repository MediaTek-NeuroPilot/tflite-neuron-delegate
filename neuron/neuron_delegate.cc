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

#include "neuron/neuron_delegate.h"

#include <farmhash.h>

#include <utility>
#include <vector>

#include "neuron/neuron_delegate_kernel.h"
#include "neuron/neuron_delegate_validation.h"
#include "neuron/neuron_implementation.h"
#include "neuron/neuron_nnapi_delegate_kernel.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace neuron {

// NeuronDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class NeuronDelegate : public SimpleDelegateInterface {
 public:
  explicit NeuronDelegate(const NeuronDelegateOptions &options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration *registration,
                                 const TfLiteNode *node,
                                 TfLiteContext *context) const override {
    std::vector<NeuronValidationFailure> failure;
    bool supported = Validate(registration, node, context, &failure);
    if (!supported) {
      TFLITE_LOG_PROD(
          tflite::TFLITE_LOG_ERROR, "OP %s (v%d) is not supported (%s)",
          tflite::EnumNameBuiltinOperator(
              static_cast<BuiltinOperator>(registration->builtin_code)),
          registration->version,
          failure.size() > 0 ? failure[0].message.c_str() : "");
    }
    return supported;
  }

  TfLiteStatus Initialize(TfLiteContext *context) override {
    if (GenModelToken(context) != kTfLiteOk) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                      "Fail to gen neuron cache token.");
    }
    redirect_nnapi_ = !neuron_->neuron_exists || RedirectNnApi();
    return kTfLiteOk;
  }

  const char *Name() const override {
    static constexpr char kName[] = "TFLiteNeuronDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    if (redirect_nnapi_) {
      StatefulNnApiDelegate::Options options = StatefulNnApiDelegate::Options();
      options.allow_fp16 = options_.allow_fp16;
      options.execution_preference =
          (StatefulNnApiDelegate::Options::ExecutionPreference)
              options_.execution_preference;
      if (options_.execution_preference == kTurboBoost) {
        options.execution_preference =
            (StatefulNnApiDelegate::Options::ExecutionPreference)
                kFastSingleAnswer;
      }
      return std::make_unique<NeuronNNAPIDelegateKernel>(options);
    }
    return std::make_unique<NeuronDelegateKernel>(neuron_, options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  // Compute the hash of a TfLiteIntArray.
  uint64_t GetHash(const TfLiteIntArray *int_array, uint64_t combine_with = 0) {
    constexpr auto kHashConst = 0x9e3779b97f4a7800ULL;
    uint64_t result = combine_with;
    for (auto i : TfLiteIntArrayView(int_array)) {
      result = result ^ (i + kHashConst + (result << 10) + (result >> 4));
    }
    return result;
  }

  TfLiteStatus GenModelToken(TfLiteContext *context) {
    // gen model token
    TfLiteIntArray *execution_plan = nullptr;
    if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, "Fail to get exexution plan");
      return kTfLiteError;
    }
    TfLiteNode *input_node, *output_node;
    TfLiteRegistration *registration;
    if (context->GetNodeAndRegistration(context, 0, &input_node,
                                        &registration) != kTfLiteOk ||
        context->GetNodeAndRegistration(context, execution_plan->size - 1,
                                        &output_node,
                                        &registration) != kTfLiteOk) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                      "Fail to get Node and Registration.");
      return kTfLiteError;
    }

    std::string bytes_count_str = std::to_string(context->tensors_size);
    const char *model_token = bytes_count_str.c_str();
    uint64_t token_parts[4];
    token_parts[0] =
        ::util::Fingerprint64(model_token, std::strlen(model_token));
    token_parts[1] = GetHash(execution_plan);
    token_parts[2] = GetHash(input_node->inputs);
    token_parts[3] = GetHash(output_node->outputs);
    std::vector<uint8_t> nnapi_cache_token(32, 0);
    uint8_t *p = reinterpret_cast<uint8_t *>(token_parts);
    for (int i = 0; i < 4 * sizeof(uint64_t); i++) {
      nnapi_cache_token[i] = p[i];
    }

    // std::string test = "{";
    // for (auto i:nnapi_cache_token) {
    //       test += std::to_string(i);
    //       test += ",";
    //}
    // test += "};";

    neuron_cache_token_ = nnapi_cache_token;
    return kTfLiteOk;
  }

  bool RedirectNnApi() {
#if defined(__ANDROID__)
    constexpr char kRedirectProp[] = "debug.tflite.neuron.redirect_nnapi";
    char redirect[PROP_VALUE_MAX] = "";
    int length = __system_property_get(kRedirectProp, redirect);
    if (length == 1 && redirect[0] == '1') {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                      "Redirect to NNAPI by system property");
      return true;
    }
#endif
    return false;
  }

  const NeuronApi *neuron_ = NeuronApiImplementation();
  const NeuronDelegateOptions options_;
  // If true, redirect the graph to NNAPI delegate
  bool redirect_nnapi_ = false;
  std::vector<uint8_t> neuron_cache_token_;
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
      kOptimizationLowLatency,
      // Default allow_fp16 = false
      false,
      // Default boost_duration = 0
      0,
      // const char* cache_dir;
      nullptr,
      // const char* model_token;
      nullptr,
      // use_ahwb;
      false,
      // use_cacheable_buffer;
      true,
      // Default compile options
      "",
      // accelerator_name
      "",
  };

  // Return default options
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteNeuronDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate *TfLiteNeuronDelegateCreate(
    const NeuronDelegateOptions *options) {
  std::unique_ptr<tflite::neuron::NeuronDelegate> neuron(
      new tflite::neuron::NeuronDelegate(
          options ? *options : TfLiteNeuronDelegateOptionsDefault()));

  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(neuron));
}

// Destroys a delegate created with `TfLiteNeuronDelegateCreate` call.
void TfLiteNeuronDelegateDelete(TfLiteDelegate *delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
