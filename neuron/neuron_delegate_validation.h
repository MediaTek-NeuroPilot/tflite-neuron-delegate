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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_VALIDATION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_VALIDATION_H_

#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace neuron {

enum class NeuronValidationFailureType : int {
  // The operator is not supported by either Neuron or the Neuron
  // Delegate.
  kUnsupportedOperator = 0,
  // The given operation or operands are not supported on the specified
  // Android SDK version. The min supported version is specified in the
  // validation failure message.
  kUnsupportedAndroidVersion = 1,
  // The version of the operator (value of TfLiteRegistration::version)
  // for the given op is not supported. The max supported version
  // is specified in the validation failure message.
  // For more details on each operator version see
  // the GetBuiltinOperatorVersion function in
  // third_party/tensorflow/lite/tools/versioning/op_version.cc.
  kUnsupportedOperatorVersion = 2,
  // The given input operand type is not supported for the current combination
  // of operator type and sdk version.
  kUnsupportedInputType = 3,
  // When using Neuron version 1.0 or 1.1, the condition
  //   input_scale * filter_scale < output_scale
  // must be true for quantized versions of the following ops:
  // * CONV_2D
  // * DEPTHWISE_CONV_2D
  // * FULLY_CONNECTED (where filter actually stands for weights)
  // The condition is relaxed and no longer required since version 1.2.
  kNotRestrictedScaleCompliant = 4,
  // The given output operand type is not supported for the current combination
  // of operator type and sdk version.
  kUnsupportedOutputType = 5,
  // The size of the operand tensor is too large.
  kUnsupportedOperandSize = 6,
  // The value of one of the operands or of a combination of operands is
  // not supported. Details are provided in the failure message.
  kUnsupportedOperandValue = 7,
  // The combination of float inputs and quantized weights or filters
  // is not supported
  kUnsupportedHybridOperator = 8,
  // The quantization type (for example per-channel quantization) is not
  // supported.
  kUnsupportedQuantizationType = 9,
  // The accelerated version of operation requires a specific operand to be
  // specified.
  kMissingRequiredOperand = 10,
  // The rank of the operand is not supported. Details in the failure message.
  kUnsupportedOperandRank = 11,
  // The input tensor cannot be dynamically-sized.
  kInputTensorShouldHaveConstantShape = 12,
  // The operator has a different number of inputs of the one or ones that
  // are supported by Neuron.
  kUnsupportedOperatorVariant = 13,
  // The accelerated version of the operator cannot specify an activation
  // function.
  kNoActivationExpected = 14,
  // Quantization scale and/or zero point are not in the supported value(s)
  // for the accelerated operation.
  kUnsupportedQuantizationParameters = 15,
};

struct NeuronValidationFailure {
  NeuronValidationFailureType type;
  std::string message;

  NeuronValidationFailure(NeuronValidationFailureType type, const char* message)
      : type(type), message(message) {}
};

bool Validate(const TfLiteRegistration* registration, const TfLiteNode* node,
              TfLiteContext* context,
              std::vector<NeuronValidationFailure>* map_failures = nullptr);

}  // namespace neuron
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_VALIDATION_H_
