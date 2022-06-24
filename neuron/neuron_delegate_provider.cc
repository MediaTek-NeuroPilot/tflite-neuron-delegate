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

#include <string>

#include "neuron/neuron_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class NeuronDelegateProvider : public DelegateProvider {
 public:
  NeuronDelegateProvider() {
    default_params_.AddParam("use_neuron", ToolParam::Create<bool>(false));
    default_params_.AddParam("neuron_use_ahwb", ToolParam::Create<bool>(false));
    default_params_.AddParam("neuron_use_cacheable_buffer",
                             ToolParam::Create<bool>(true));
    default_params_.AddParam("neuron_execution_preference",
                             ToolParam::Create<std::string>(""));
    default_params_.AddParam("neuron_allow_fp16",
                             ToolParam::Create<bool>(false));
    default_params_.AddParam("neuron_optimization_hint",
                             ToolParam::Create<int32_t>(1));
    default_params_.AddParam("neuron_compile_options",
                             ToolParam::Create<std::string>(""));
    default_params_.AddParam("neuron_accelerator_name",
                             ToolParam::Create<std::string>(""));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "NeuronDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(NeuronDelegateProvider);

std::vector<Flag> NeuronDelegateProvider::CreateFlags(
    ToolParams* params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_neuron", params, "use the Neuron delegate."),
      CreateFlag<bool>("neuron_use_ahwb", params, "use AhardwareBuffer"),
      CreateFlag<bool>("neuron_use_cacheable_buffer", params,
                       "use cacheable buffer"),
      CreateFlag<bool>("neuron_allow_fp16", params, "allow fp16 precision."),
      CreateFlag<std::string>(
          "neuron_execution_preference", params,
          "execution preference for neuron delegate. Should "
          "be one of the following: turbo_boost, fast_single_answer, "
          "sustained_speed, low_power, undefined"),
      CreateFlag<int32_t>("neuron_optimization_hint", params,
                          "set optimization hint. Currently, bit 0: latency, "
                          "bit 1: deep fusion, bit 2: batch processing "),
      CreateFlag<std::string>("neuron_compile_options", params,
                              "Add compile options"),
      CreateFlag<std::string>("neuron_accelerator_name", params,
                              "Set target devices. (seperated by ',') "
                              "e.g. mtk-mdla,mtk-dsp,mtk-gpu")};

  return flags;
}

void NeuronDelegateProvider::LogParams(const ToolParams& params,
                                       bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_neuron", "Use neuron", verbose);
  if (!params.Get<bool>("use_neuron")) return;

  LOG_TOOL_PARAM(params, bool, "neuron_use_ahwb", "Use AhardwareBuffer",
                 verbose);
  LOG_TOOL_PARAM(params, bool, "neuron_use_cacheable_buffer",
                 "Use cacheable buffer", verbose);
  LOG_TOOL_PARAM(params, bool, "neuron_allow_fp16", "Neuron allow FP16",
                 verbose);
  LOG_TOOL_PARAM(params, std::string, "neuron_execution_preference",
                 "Neuron execution preference", verbose);
  LOG_TOOL_PARAM(params, int32_t, "neuron_optimization_hint",
                 "Neuron optimization hint", verbose);
  LOG_TOOL_PARAM(params, std::string, "neuron_compile_options",
                 "Add compile options", verbose);
  LOG_TOOL_PARAM(params, std::string, "neuron_accelerator_name",
                 "Neuron accelerator name", verbose);
}

TfLiteDelegatePtr NeuronDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  LogParams(params, true);
  if (params.Get<bool>("use_neuron")) {
    auto default_options = TfLiteNeuronDelegateOptionsDefault();

    if (params.Get<bool>("neuron_allow_fp16")) {
      default_options.allow_fp16 = true;
    }

    if (params.Get<bool>("neuron_use_ahwb")) {
      default_options.use_ahwb = true;
    }
    if (!params.Get<bool>("neuron_use_cacheable_buffer")) {
      default_options.use_cacheable_buffer = false;
    }

    std::string string_execution_preference =
        params.Get<std::string>("neuron_execution_preference");
    if (!string_execution_preference.empty()) {
      if (string_execution_preference == "low_power") {
        default_options.execution_preference = ExecutionPreference::kLowPower;
      } else if (string_execution_preference == "sustained_speed") {
        default_options.execution_preference =
            ExecutionPreference::kSustainedSpeed;
      } else if (string_execution_preference == "fast_single_answer") {
        default_options.execution_preference =
            ExecutionPreference::kFastSingleAnswer;
      } else if (string_execution_preference == "turbo_boost") {
        default_options.execution_preference = ExecutionPreference::kTurboBoost;
      } else if (string_execution_preference == "undefined") {
        default_options.execution_preference = ExecutionPreference::kUndefined;
      } else {
        TFLITE_LOG(WARN) << "The provided value ("
                         << string_execution_preference
                         << ") is not a valid neuron execution preference.";
      }
    }

    uint32_t hints = params.Get<int32_t>("neuron_optimization_hint");
    default_options.optimization_hint = hints;
    if (hints >> 3)
      TFLITE_LOG(WARN) << "unsupported hints: " << std::hex << hints;

    std::string string_compile_options =
        params.Get<std::string>("neuron_compile_options");
    if (!string_compile_options.empty()) {
      strcpy(default_options.compile_options, string_compile_options.c_str());
    }

    std::string string_accelerator_name =
        params.Get<std::string>("neuron_accelerator_name");
    if (!string_accelerator_name.empty()) {
      strcpy(default_options.accelerator_name, string_accelerator_name.c_str());
    }

    return TfLiteNeuronDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
NeuronDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr), params.GetPosition<bool>("use_neuron"));
}

}  // namespace tools
}  // namespace tflite
