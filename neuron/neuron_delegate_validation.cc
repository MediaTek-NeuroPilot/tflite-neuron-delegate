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

#include "neuron/neuron_delegate_validation.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "neuron/neuron_delegate_utils.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace neuron {
namespace {

struct OpValidationContext {
  bool is_valid;
  std::vector<NeuronValidationFailure>* validation_failures;
};

#define EXPECT_INPUT_TYPE_IN(actual_type, ...)                     \
  ExpectTypeIn(actual_type, {__VA_ARGS__},                         \
               NeuronValidationFailureType::kUnsupportedInputType, \
               "Input type not in expected list " #__VA_ARGS__, &val_ctx)

void AddValidationFailure(NeuronValidationFailureType failure_type,
                          const char* message, OpValidationContext* val_ctx) {
  val_ctx->is_valid = false;

  if (val_ctx->validation_failures) {
    val_ctx->validation_failures->push_back({failure_type, message});
  }
}

template <typename... Args>
void AddValidationFailureFmt(OpValidationContext* val_ctx,
                             NeuronValidationFailureType failure_type,
                             const char* message_fmt, Args... args) {
  val_ctx->is_valid = false;
#ifdef NEURON_VERBOSE_VALIDATION
  if (val_ctx->validation_failures) {
    size_t req_buf_size = snprintf(nullptr, 0, message_fmt, args...) + 1;
    std::unique_ptr<char[]> tmp_buf(new char[req_buf_size]);
    snprintf(tmp_buf.get(), req_buf_size, message_fmt, args...);

    val_ctx->validation_failures->push_back({failure_type, tmp_buf.get()});
  }
#endif
}

bool Expect(bool condition, NeuronValidationFailureType failure_type,
            const char* message, OpValidationContext* val_ctx) {
  if (!condition) {
    AddValidationFailure(failure_type, message, val_ctx);
    return false;
  }
  return true;
}

template <typename... Args>
bool ExpectFmt(bool condition, OpValidationContext* val_ctx,
               NeuronValidationFailureType failure_type,
               const char* message_fmt, Args... args) {
  if (!condition) {
    AddValidationFailureFmt(val_ctx, failure_type, message_fmt, args...);
    return false;
  }
  return true;
}

bool ExpectTypeIn(TfLiteType actual_type,
                  std::initializer_list<TfLiteType> allowed_types,
                  NeuronValidationFailureType failure_type, const char* msg,
                  OpValidationContext* val_ctx) {
  return Expect(std::find(allowed_types.begin(), allowed_types.end(),
                          actual_type) != allowed_types.end(),
                failure_type, msg, val_ctx);
}

bool ExpectMinAndroidSdkVersion(int curr_version, int min_version,
                                OpValidationContext* val_ctx) {
  return ExpectFmt(curr_version >= min_version, val_ctx,
                   NeuronValidationFailureType::kUnsupportedAndroidVersion,
                   "Android sdk version less than %d", min_version);
}

bool ExpectMaxOpVersion(int curr_version, int max_version,
                        OpValidationContext* val_ctx) {
  return ExpectFmt(curr_version <= max_version, val_ctx,
                   NeuronValidationFailureType::kUnsupportedOperatorVersion,
                   "OP Version higher than %d", max_version);
}

bool ExpectOpVersion(int curr_version, int max_version,
                     OpValidationContext* val_ctx) {
  return ExpectFmt(curr_version <= max_version, val_ctx,
                   NeuronValidationFailureType::kUnsupportedOperatorVersion,
                   "OP Version different from %d", max_version);
}

bool ExpectIsFloatOperator(const TfLiteContext* context, const TfLiteNode* node,
                           OpValidationContext* val_ctx) {
  const auto input_type = context->tensors[node->inputs->data[0]].type;
  return Expect(IsFloat(input_type),
                NeuronValidationFailureType::kUnsupportedInputType,
                "Input should be Float", val_ctx);
}

bool ExpectIsFloatOrUint8Operator(const TfLiteContext* context,
                                  const TfLiteNode* node,
                                  OpValidationContext* val_ctx) {
  const auto input_type = context->tensors[node->inputs->data[0]].type;
  return Expect(IsFloatOrUInt8(input_type),
                NeuronValidationFailureType::kUnsupportedInputType,
                "Input should be Float or UINT8", val_ctx);
}

bool ExpectIsFloatOrQuant8Operator(const TfLiteContext* context,
                                   const TfLiteNode* node,
                                   OpValidationContext* val_ctx) {
  const auto input_type = context->tensors[node->inputs->data[0]].type;
  return Expect(IsFloat(input_type) || IsQuantized(input_type),
                NeuronValidationFailureType::kUnsupportedInputType,
                "Input should be Float or Quant8", val_ctx);
}

// When using Neuron, the condition below must be true
// for quantized versions of the following ops:
// * CONV_2D
// * DEPTHWISE_CONV_2D
// * FULLY_CONNECTED (where filter actually stands for weights)
bool ExpectIsRestrictedScalesCompliant(const TfLiteContext* context,
                                       const TfLiteNode* node,
                                       OpValidationContext* val_ctx) {
  const int input_id = node->inputs->data[0];
  const int filter_id = node->inputs->data[1];
  const int output_id = node->outputs->data[0];
  const float input_scale = context->tensors[input_id].params.scale;
  const float filter_scale = context->tensors[filter_id].params.scale;
  const float output_scale = context->tensors[output_id].params.scale;
  return Expect(
      input_scale * filter_scale < output_scale,
      NeuronValidationFailureType::kNotRestrictedScaleCompliant,
      "When using Neuron, input_scale * filter_scale < output_scale:", val_ctx);
}

}  // namespace

bool Validate(const TfLiteRegistration* registration, const TfLiteNode* node,
              TfLiteContext* context,
              std::vector<NeuronValidationFailure>* map_failures) {
  OpValidationContext val_ctx{true, map_failures};

  const auto builtin_code = registration->builtin_code;
  const auto version = registration->version;

  switch (builtin_code) {
    case kTfLiteBuiltinAdd: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinArgMax:
    case kTfLiteBuiltinArgMin: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const TfLiteType input_type =
          context->tensors[node->inputs->data[(0)]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat16, kTfLiteFloat32,
                           kTfLiteUInt8, kTfLiteInt8);
    } break;
    case kTfLiteBuiltinMul: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinAveragePool2d: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinMaxPool2d: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinL2Pool2d: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinConv2d: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      const TfLiteType input_type =
          context->tensors[node->inputs->data[(0)]].type;
      if (input_type == kTfLiteUInt8) {
        ExpectIsRestrictedScalesCompliant(context, node, &val_ctx);
      }
      // TODO(Code): Add support for Conv2D with omitted bias.
      Expect(node->inputs->size == 3,
             NeuronValidationFailureType::kMissingRequiredOperand,
             "Conv2D with omitted bias not supported", &val_ctx);
    } break;
    case kTfLiteBuiltinDepthwiseConv2d: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      const TfLiteType input_type =
          context->tensors[node->inputs->data[(0)]].type;
      if (input_type == kTfLiteUInt8) {
        ExpectIsRestrictedScalesCompliant(context, node, &val_ctx);
      }
    } break;
    case kTfLiteBuiltinFullyConnected: {
      ExpectMaxOpVersion(version, 5, &val_ctx);
      // TODO(Code): Add support for FullyConnected with no bias.
      Expect(node->inputs->size == 3 &&
                 node->inputs->data[2] != kTfLiteOptionalTensor,
             NeuronValidationFailureType::kMissingRequiredOperand,
             "FullyConnected with no bias not supported", &val_ctx);
      const TfLiteType input_type =
          context->tensors[node->inputs->data[(0)]].type;
      if (input_type == kTfLiteUInt8) {
        ExpectIsRestrictedScalesCompliant(context, node, &val_ctx);
      }
    } break;
    case kTfLiteBuiltinHardSwish: {
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinSoftmax: {
      ExpectOpVersion(version, 2, &val_ctx);
      const auto& input = context->tensors[node->outputs->data[0]];
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      const int input_rank = input.dims->size;
      Expect(input_rank <= 4,
             NeuronValidationFailureType::kUnsupportedOperandRank,
             "Input rank should be <= 4", &val_ctx);
    } break;
    case kTfLiteBuiltinReshape: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      Expect(node->inputs->size >= 2,
             NeuronValidationFailureType::kMissingRequiredOperand,
             "Expected at least 2 inputs", &val_ctx);
      if (node->inputs->size >= 2) {
        Expect(context->tensors[node->inputs->data[1]].allocation_type ==
                   kTfLiteMmapRo,
               NeuronValidationFailureType::kInputTensorShouldHaveConstantShape,
               "The shape input tensor must be constant.", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinResizeBilinear: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      const auto& input = context->tensors[node->inputs->data[0]];
      Expect(input.dims->size == 4,
             NeuronValidationFailureType::kUnsupportedOperandRank,
             "Input should have rank 4", &val_ctx);
      Expect(node->inputs->size >= 2,
             NeuronValidationFailureType::kUnsupportedOperatorVariant,
             "Expected at least 2 inputs", &val_ctx);
      if (node->inputs->size >= 2) {
        Expect(context->tensors[node->inputs->data[1]].allocation_type ==
                   kTfLiteMmapRo,
               NeuronValidationFailureType::kInputTensorShouldHaveConstantShape,
               "The size input tensor must be constant.", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinResizeNearestNeighbor: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
    } break;
    case kTfLiteBuiltinSqueeze: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinL2Normalization: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      auto builtin = reinterpret_cast<TfLiteL2NormParams*>(node->builtin_data);
      Expect(builtin->activation == kTfLiteActNone,
             NeuronValidationFailureType::kNoActivationExpected,
             "Expected no activation", &val_ctx);
    } break;
    case kTfLiteBuiltinConcatenation: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      Expect(reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data)
                     ->activation == kTfLiteActNone,
             NeuronValidationFailureType::kNoActivationExpected,
             "No activation function supported", &val_ctx);
      Expect(context->tensors[node->inputs->data[0]].dims->size <= 4,
             NeuronValidationFailureType::kUnsupportedOperandRank,
             "Input rank should be less than 4", &val_ctx);
    } break;
    case kTfLiteBuiltinDequantize: {
      ExpectOpVersion(version, 2, &val_ctx);
      const auto& input = context->tensors[node->inputs->data[0]];
      EXPECT_INPUT_TYPE_IN(input.type, kTfLiteUInt8, kTfLiteInt8);
    } break;
    case kTfLiteBuiltinFloor: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinReluN1To1:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinLogistic:
    case kTfLiteBuiltinTanh: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinSub: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      const int input0_rank =
          context->tensors[node->inputs->data[0]].dims->size;
      const int input1_rank =
          context->tensors[node->inputs->data[1]].dims->size;
      Expect(input0_rank <= 4 && input1_rank <= 4,
             NeuronValidationFailureType::kUnsupportedOperandRank,
             "Input rank must be <= 4", &val_ctx);
    } break;
    case kTfLiteBuiltinDiv: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      const TfLiteIntArrayView input_shape(
          context->tensors[node->inputs->data[0]].dims);
      Expect(!HasZeroes(input_shape),
             NeuronValidationFailureType::kUnsupportedOperandValue,
             "Neuron pad ops do not support input tensors with no elements",
             &val_ctx);
      Expect(node->inputs->size >= 2,
             NeuronValidationFailureType::kUnsupportedOperatorVariant,
             "Expecting at least 2 inputs", &val_ctx);
    } break;
    case kTfLiteBuiltinSpaceToBatchNd: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
    } break;
    case kTfLiteBuiltinBatchToSpaceNd: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      auto crops = context->tensors[node->inputs->data[2]];
      auto crops_data = crops.data.i32;
      Expect(crops_data && crops.bytes == 16 && crops_data[0] == 0 &&
                 crops_data[1] == 0 && crops_data[2] == 0 && crops_data[3] == 0,
             NeuronValidationFailureType::kUnsupportedOperandValue,
             "All crops should be 0.", &val_ctx);
    } break;
    case kTfLiteBuiltinStridedSlice: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
    } break;
    case kTfLiteBuiltinTranspose: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      // Note that the permutation input tensor value dictates the output
      // dimensions.
      // TODO(Code): Support dynamically-sized tensors in delegates.
      Expect((node->inputs->size > 1) &&
                 (context->tensors[node->inputs->data[1]].allocation_type ==
                  kTfLiteMmapRo),
             NeuronValidationFailureType::kInputTensorShouldHaveConstantShape,
             "Dynamically-sized tensors not supported.", &val_ctx);
    } break;
    case kTfLiteBuiltinAbs:
    case kTfLiteBuiltinExp:
    case kTfLiteBuiltinLog:
    case kTfLiteBuiltinRsqrt:
    case kTfLiteBuiltinPow: {
      ExpectOpVersion(version, 1, &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat16, kTfLiteFloat32);
    } break;
    case kTfLiteBuiltinSlice: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      const auto begin_type = context->tensors[node->inputs->data[1]].type;
      const auto size_type = context->tensors[node->inputs->data[2]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt32,
                           kTfLiteUInt8, kTfLiteInt8);
      Expect(begin_type == kTfLiteInt32,
             NeuronValidationFailureType::kUnsupportedInputType,
             "Begin type should be Int32", &val_ctx);
      Expect(size_type == kTfLiteInt32,
             NeuronValidationFailureType::kUnsupportedInputType,
             "Size type should be Int32", &val_ctx);
    } break;
    case kTfLiteBuiltinSin: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinTransposeConv: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      Expect((node->inputs->size > 1) &&
                 (context->tensors[node->inputs->data[0]].allocation_type ==
                  kTfLiteMmapRo) &&
                 (context->tensors[node->inputs->data[1]].allocation_type ==
                  kTfLiteMmapRo),
             NeuronValidationFailureType::kInputTensorShouldHaveConstantShape,
             "Dynamically-sized tensors not supported.", &val_ctx);
    } break;
    case kTfLiteBuiltinSqrt: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinSpaceToDepth: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinSvdf: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinLstm: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
    } break;
    case kTfLiteBuiltinMean: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      Expect(context->tensors[node->inputs->data[0]].type == kTfLiteFloat32 ||
                 IsQuantized(context->tensors[node->inputs->data[0]].type),
             NeuronValidationFailureType::kUnsupportedInputType,
             "Expected Float32 or Quantized input", &val_ctx);
    } break;
    case kTfLiteBuiltinEmbeddingLookup: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinHashtableLookup: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMinimum: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8, kTfLiteInt32);
      const TfLiteTensor& operand0 = context->tensors[node->inputs->data[0]];
      if (operand0.dims->size == 0) {
        Expect(operand0.allocation_type == kTfLiteMmapRo,
               NeuronValidationFailureType::kUnsupportedInputType,
               "Scalar operand should be constant", &val_ctx);
      }
      const TfLiteTensor& operand1 = context->tensors[node->inputs->data[1]];
      if (operand1.dims->size == 0) {
        Expect(operand1.allocation_type == kTfLiteMmapRo,
               NeuronValidationFailureType::kUnsupportedInputType,
               "Scalar operand should be constant", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinCast: {
      ExpectOpVersion(version, 1, &val_ctx);
      const TfLiteType input_type =
          context->tensors[node->inputs->data[0]].type;
      const TfLiteType output_type =
          context->tensors[node->outputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt32,
                           kTfLiteUInt8, kTfLiteInt8);
      ExpectTypeIn(output_type,
                   {kTfLiteFloat32, kTfLiteInt32, kTfLiteUInt8, kTfLiteInt8},
                   NeuronValidationFailureType::kUnsupportedOutputType,
                   "Output type should be one of kTfLiteFloat32, kTfLiteInt32, "
                   "kTfLiteUInt8, kTfLiteInt8.",
                   &val_ctx);
    } break;
    case kTfLiteBuiltinLeakyRelu:
    case kTfLiteBuiltinPrelu: {
      ExpectOpVersion(version, 1, &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8);
    } break;
    case kTfLiteBuiltinLogicalOr:
    case kTfLiteBuiltinLogicalAnd:
    case kTfLiteBuiltinLogicalNot: {
      ExpectOpVersion(version, 1, &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      Expect(input_type == kTfLiteBool,
             NeuronValidationFailureType::kUnsupportedInputType,
             "Input should be bool", &val_ctx);
    } break;
    case kTfLiteBuiltinLess:
    case kTfLiteBuiltinLessEqual:
    case kTfLiteBuiltinGreater:
    case kTfLiteBuiltinGreaterEqual:
    case kTfLiteBuiltinEqual:
    case kTfLiteBuiltinNotEqual: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8, kTfLiteBool, kTfLiteInt32);
    } break;
    case kTfLiteBuiltinNeg: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt32);
    } break;
    case kTfLiteBuiltinTopkV2: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const auto& input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt32,
                           kTfLiteUInt8, kTfLiteInt8);
      const auto& k_param = context->tensors[node->inputs->data[1]];
      Expect(k_param.type == kTfLiteInt32 &&
                 k_param.allocation_type == kTfLiteMmapRo,
             NeuronValidationFailureType::kUnsupportedInputType,
             "K param should be a constant of type Int32", &val_ctx);
    } break;
    case kTfLiteBuiltinSelect: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const auto value_type = context->tensors[node->inputs->data[1]].type;
      EXPECT_INPUT_TYPE_IN(value_type, kTfLiteFloat32, kTfLiteInt32,
                           kTfLiteUInt8, kTfLiteInt8);
      TfLiteIntArray* condition_shape =
          context->tensors[node->inputs->data[0]].dims;
      TfLiteIntArray* input_shape =
          context->tensors[node->inputs->data[1]].dims;
      Expect(TfLiteIntArrayEqual(condition_shape, input_shape),
             NeuronValidationFailureType::kUnsupportedOperandValue,
             "Condition and inputs tensors shuld have the same shape",
             &val_ctx);
    } break;
    case kTfLiteBuiltinGather: {
      ExpectOpVersion(version, 2, &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      const auto& positions = context->tensors[node->inputs->data[1]];

      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteFloat16,
                           kTfLiteInt32, kTfLiteUInt8, kTfLiteInt8);

      Expect(positions.type == kTfLiteInt32,
             NeuronValidationFailureType::kUnsupportedInputType,
             "Positions type should be one of kTfLiteInt32", &val_ctx);
      Expect(positions.dims->size != 0,
             NeuronValidationFailureType::kUnsupportedOperandRank,
             "0-dimension args are not supported by Neuron.", &val_ctx);
    } break;
    case kTfLiteBuiltinSplit: {
      ExpectOpVersion(version, 3, &val_ctx);
      const TfLiteTensor& input = context->tensors[node->inputs->data[1]];
      EXPECT_INPUT_TYPE_IN(input.type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8, kTfLiteInt32);
      const TfLiteTensor& axis = context->tensors[node->inputs->data[0]];
      Expect(axis.type == kTfLiteInt32 && axis.allocation_type == kTfLiteMmapRo,
             NeuronValidationFailureType::kUnsupportedInputType,
             "Neuron only supports constant int32 axis tensor.", &val_ctx);
    } break;
    case kTfLiteBuiltinQuantize: {
      ExpectOpVersion(version, 2, &val_ctx);
      const auto value_type = context->tensors[node->inputs->data[0]].type;
      Expect(value_type == kTfLiteFloat32 || value_type == kTfLiteInt16 ||
                 IsQuantized(value_type),
             NeuronValidationFailureType::kUnsupportedInputType,
             "Value should be quantized or Float32.", &val_ctx);
      if (value_type == kTfLiteInt16 || IsQuantized(value_type)) {
        const auto quantization_params =
            context->tensors[node->inputs->data[0]].params;
        Expect(quantization_params.scale > 0.f,
               NeuronValidationFailureType::kUnsupportedQuantizationParameters,
               "Input quantization scale should be > 0.", &val_ctx);
      }
      const auto output_type = context->tensors[node->outputs->data[0]].type;
      ExpectTypeIn(
          output_type, {kTfLiteInt16, kTfLiteUInt8, kTfLiteInt8},
          NeuronValidationFailureType::kUnsupportedOutputType,
          "Output should be kTfLiteInt16 or kTfLiteUInt8 or kTfLiteInt8.",
          &val_ctx);
      const auto quantization_params =
          context->tensors[node->outputs->data[0]].params;
      Expect(quantization_params.scale > 0.f,
             NeuronValidationFailureType::kUnsupportedQuantizationParameters,
             "Output quantization scale should be > 0.", &val_ctx);
    } break;
    case kTfLiteBuiltinReduceAny:
    case kTfLiteBuiltinReduceMin:
    case kTfLiteBuiltinReduceMax: {
      ExpectOpVersion(version, 2, &val_ctx);
    } break;
    case kTfLiteBuiltinDepthToSpace: {
      const TfLiteType input_type =
          context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8);
    } break;
    case kTfLiteBuiltinReduceProd:
    case kTfLiteBuiltinSum: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinElu: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinFill: {
      ExpectOpVersion(version, 1, &val_ctx);
      const auto& dims_tensor = context->tensors[node->inputs->data[0]];
      Expect(IsConstantTensor(&dims_tensor),
             NeuronValidationFailureType::kUnsupportedInputType,
             "Neuron doesn't support dynamic dimensions tensor.", &val_ctx);
      EXPECT_INPUT_TYPE_IN(dims_tensor.type, kTfLiteInt32, kTfLiteInt64);
      if (IsConstantTensor(&dims_tensor)) {
        Expect(dims_tensor.dims->data[0] != 0,
               NeuronValidationFailureType::kUnsupportedOperandValue,
               "Neuron doesn't support generating scalars from FILL", &val_ctx);
        if (dims_tensor.type == kTfLiteInt64) {
          bool fit_in_int32 =
              std::all_of(dims_tensor.data.i64,
                          dims_tensor.data.i64 + dims_tensor.dims->data[0],
                          [](int64_t dim) {
                            return std::numeric_limits<int32_t>::min() <= dim &&
                                   dim <= std::numeric_limits<int32_t>::max();
                          });
          Expect(fit_in_int32,
                 NeuronValidationFailureType::kUnsupportedOperandValue,
                 "Neuron only supports int32 dimensions tensor. If the "
                 "dimensions type is int64 and they are constant we can "
                 "convert them to int32 if the value isn't too large.",
                 &val_ctx);
        }
      }
      const auto& value_tensor = context->tensors[node->inputs->data[1]];
      EXPECT_INPUT_TYPE_IN(value_tensor.type, kTfLiteFloat32, kTfLiteInt32,
                           kTfLiteInt64);
      if (value_tensor.type == kTfLiteInt64) {
        Expect(
            IsConstantTensor(&value_tensor) &&
                *value_tensor.data.i64 <= std::numeric_limits<int32_t>::max() &&
                *value_tensor.data.i64 >= std::numeric_limits<int32_t>::min(),
            NeuronValidationFailureType::kUnsupportedInputType,
            "Neuron only supports int32 input. If the input type is int64 and "
            "constant we can convert it to int32 if the value isn't too "
            "large.",
            &val_ctx);
      }
    } break;
    default:
      // All other operators are not mapped.
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                      "Unsupported operation type %d", builtin_code);
      AddValidationFailure(NeuronValidationFailureType::kUnsupportedOperator,
                           "Unsupported operation type.", &val_ctx);
  }
  return val_ctx.is_valid;
}

}  // namespace neuron
}  // namespace tflite
