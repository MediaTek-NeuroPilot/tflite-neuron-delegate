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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_BUILDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_BUILDER_H_

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "neuron/neuron_delegate_kernel.h"
#include "neuron/neuron_delegate_utils.h"
#include "neuron/neuron_delegate_validation.h"
#include "neuron/neuron_implementation.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace neuron {

// Bit mask for tensor flags.
enum {
  NN_TENSOR_FLAG_SCALAR_AS_TENSOR = 1U << 0,
  NN_TENSOR_FLAG_INT8_CONVERSION = 1U << 1,
  NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED = 1U << 2,
  NN_TENSOR_FLAG_FORCE_PER_CHANNEL = 1U << 3,
};

class DequantizeMapping {
 public:
  int DequantizedAnnIndex(int neuron_index, TfLiteType type) const {
    for (const auto& element : mapping_) {
      if (neuron_index == std::get<0>(element) &&
          type == std::get<1>(element)) {
        return std::get<2>(element);
      }
    }
    return -1;
  }

  void Add(int neuron_index, TfLiteType type, int dequantized_ann_index) {
    // This assumes it is not already mapped.
    mapping_.emplace_back(neuron_index, type, dequantized_ann_index);
  }

 private:
  // Each tuple specifies the ANN (quantized) tensor index, the desired
  // floating-point type and the matching ANN (dequantized) tensor index. This
  // could use a map but instead std::vector is used to keep code size lower.
  std::vector<std::tuple<int, TfLiteType, int>> mapping_;
};

// Track tensor indices to Neuron tensor indices mapping.
class OperandMapping {
 public:
  // Given a TFLite index return the Neuron index. If it doesn't exist
  // return -1.
  int lite_index_to_neuron(int index) const {
    if (index >= 0 && index < lite_tensor_to_neuron_tensor_.size())
      return lite_tensor_to_neuron_tensor_[index];
    else
      return -1;
  }

  // Neuron uses non tensor operands instead of structs. This creates one
  // and returns the index. It uses a std::vector and resizes it as needed
  // keeping -1 to unmapped values. Intermediate tensors likely will not
  // be mapped.
  int add_new_non_tensor_operand() { return next_neuron_tensor_index_++; }

  // This call is necessary for input operands generated by the delegate
  // to map constant inputs not present in TFLite but required by Neuron,
  // for example when splitting one input in several ones.
  int add_delegate_generated_input_neuron_tensors_operand() {
    return next_neuron_tensor_index_++;
  }

  // Add a new mapping from `tflite_index` and return the Neuron tensor
  // index.
  int add_new_neuron_tensor_index(int tflite_index) {
    if (tflite_index >= lite_tensor_to_neuron_tensor_.size()) {
      lite_tensor_to_neuron_tensor_.resize(tflite_index + 1, -1);
    }
    const int new_tensor_index = next_neuron_tensor_index_++;
    lite_tensor_to_neuron_tensor_[tflite_index] = new_tensor_index;
    return new_tensor_index;
  }

  // Given a TFLite index returns a TFLite type to which a tensor must be
  // converted during copying the data to the memory allocated for Neuron.
  // kTfLiteNoType means no conversion is needed.
  TfLiteType lite_index_to_neuron_type_conversion(int index) const {
    if (index >= 0 && index < index_to_type_conversion_.size())
      return index_to_type_conversion_[index];
    else
      return kTfLiteNoType;
  }

  // Add a new mapping from TFLite index to a type conversion.
  void add_type_conversion(int tflite_index, TfLiteType tflite_type) {
    if (tflite_index >= index_to_type_conversion_.size()) {
      index_to_type_conversion_.resize(tflite_index + 1, kTfLiteNoType);
    }
    index_to_type_conversion_[tflite_index] = tflite_type;
  }

 private:
  // Next index of neuron tensor
  int next_neuron_tensor_index_ = 0;

  // Mapping from lite index. Use a std::vector for speed and code size
  // rather than a map.
  std::vector<int> lite_tensor_to_neuron_tensor_;
  // Mapping from lite index to a type which tensor must be converted to during
  // the copying of the data to the memory allocated for Neuron.
  // kTfLiteNoType means no conversion is needed. Use an std::vector for speed
  // and code size rather than a map.
  std::vector<TfLiteType> index_to_type_conversion_;
};

// Abstract builder for building an op in the Neuron graph. This handles
// the disparity between TFLite and Neuron operand types. Neuron has
// singular operands for both tensors and parameters, and TFLite separates the
// two.
class NeuronOpBuilder {
 public:
  NeuronOpBuilder(const NeuronApi* neuronapi, TfLiteContext* context,
                  OperandMapping* tensor_mapping,
                  DequantizeMapping* dequantize_mapping, NeuronModel* nn_model)
      : neuronapi_(neuronapi),
        context_(context),
        operand_mapping_(tensor_mapping),
        dequantize_mapping_(dequantize_mapping),
        nn_model_(nn_model) {}

  TfLiteStatus AddScalarBoolOperand(bool value) {
    return AddScalarOperand<bool>(value, NEURON_BOOL);
  }

  TfLiteStatus AddScalarInt32Operand(int32_t value) {
    return AddScalarOperand<int32_t>(value, NEURON_INT32);
  }

  TfLiteStatus AddScalarFloat32Operand(float value) {
    return AddScalarOperand<float>(value, NEURON_FLOAT32);
  }

  TfLiteStatus AddVectorInt32Operand(const int32_t* values,
                                     uint32_t num_values) {
    return AddVectorOperand<int32_t>(values, num_values, NEURON_TENSOR_INT32,
                                     /*scale=*/0.f, /*zero_point=*/0);
  }

  TfLiteStatus AddVectorInt32Operand(const int32_t* values, uint32_t num_values,
                                     float scale, int32_t zero_point) {
    return AddVectorOperand<int32_t>(values, num_values, NEURON_TENSOR_INT32,
                                     scale, zero_point);
  }

  TfLiteStatus AddVectorFloat32Operand(const float* values,
                                       uint32_t num_values) {
    return AddVectorOperand<float>(values, num_values, NEURON_TENSOR_FLOAT32);
  }

  TfLiteStatus AddPoolingParams(void* data) {
    auto builtin = reinterpret_cast<TfLitePoolParams*>(data);
    AddScalarInt32Operand(builtin->padding);
    AddScalarInt32Operand(builtin->stride_width);
    AddScalarInt32Operand(builtin->stride_height);
    AddScalarInt32Operand(builtin->filter_width);
    AddScalarInt32Operand(builtin->filter_height);
    AddScalarInt32Operand(builtin->activation);
    return kTfLiteOk;
  }

  TfLiteStatus AddTensorInput(int tensor_index, bool hybrid_op,
                              int tensor_flags = 0) {
    return AddTensor(tensor_index, hybrid_op, &augmented_inputs_, tensor_flags);
  }

  TfLiteStatus AddTensorOutput(int tensor_index, int tensor_flags = 0) {
    return AddTensor(tensor_index, /*hybrid_op=*/false, &augmented_outputs_,
                     tensor_flags);
  }

  TfLiteStatus AddAdditionalFloat32OutputTensor(uint32_t dimension_count) {
    std::vector<uint32_t> dims(dimension_count, 0);
    return AddFloat32OutputTensor(dimension_count, dims.data(), nullptr);
  }

  TfLiteStatus AddStateFloat32Tensor(int tensor_index,
                                     int* ann_tensor_index_out) {
    TfLiteTensor* tensor = &context_->tensors[tensor_index];
    return AddFloat32OutputTensor(
        tensor->dims->size, reinterpret_cast<uint32_t*>(tensor->dims->data),
        ann_tensor_index_out);
  }

  // Add a constant tensor with a single element, intended for broadcast capable
  // ops.
  TfLiteStatus AddSingleValueConstantTensor(float value, bool is_quantized) {
    if (!is_quantized) {
      return AddVectorFloat32Operand(&value, 1);
    } else {
      // in the case that we need to add a quantized tensor, set the value to
      // 64, zero_point to be 0 and adjust scale accordingly.
      const uint8_t quant8_value = 64;
      return AddVectorOperand<uint8_t>(&quant8_value, 1,
                                       NEURON_TENSOR_QUANT8_ASYMM,
                                       value / quant8_value, 0);
    }
  }

  // Calculate the scale and zero_point for 8-bit unsigned tensor, given float
  // min and max. zero_point is clamped to [0, 255].
  TfLiteStatus CalculateQuantizationParams(float min, float max, float* scale,
                                           int* zero_point) {
    if (max < min) return kTfLiteError;
    *scale = (max - min) / 255.f;
    if (min > 0.f) {
      *zero_point = 0;
    } else if (max < 0.f) {
      *zero_point = 255;
    } else {
      *zero_point = (0.f - min) / (*scale);
    }
    return kTfLiteOk;
  }

  // Lower hardswish according to the following equation:
  // hard_swish[x] = x (ReLU6(x + 3)) / 6 == x * (Relu_N1_to_1(x/3) * 3 + 3) / 6
  // = 0.5x * Relu_N1_to_1(x/3) + 0.5x
  TfLiteStatus AddHardSwish(int lite_input_index, int lite_output_index,
                            bool need_int8_conversion) {
    const TfLiteTensor& tensor = context_->tensors[lite_input_index];
    float input_scale = tensor.params.scale;
    int input_zero_point = tensor.params.zero_point;
    float input_min = 0.f;
    float input_max = 0.f;
    int tensor_flags = 0;
    if (need_int8_conversion) {
      tensor_flags = tensor_flags | NN_TENSOR_FLAG_INT8_CONVERSION;
      input_zero_point += 128;
    }
    bool is_quantized = false;
    int nn_type = NEURON_TENSOR_FLOAT32;
    if (tensor.type == kTfLiteInt8 || tensor.type == kTfLiteUInt8) {
      is_quantized = true;
      nn_type = NEURON_TENSOR_QUANT8_ASYMM;
      input_min = (0 - input_zero_point) * input_scale;
      input_max = (255 - input_zero_point) * input_scale;
    }

    // Stage1 : s1 = Relu1(x * 1/3)
    float s1_output_min = 0.f;
    float s1_output_max = 0.f;
    int s1_out_ann_index = 0;
    {
      float s1_output_scale = 0.f;
      int s1_output_zero_point = 0;
      if (is_quantized) {
        // clamp the output range to [-1, 1] if needed.
        s1_output_min = input_min / 3.f < -1.f ? -1.f : input_min / 3.f;
        s1_output_max = input_max / 3.f > 1.f ? 1.f : input_max / 3.f;
        CalculateQuantizationParams(s1_output_min, s1_output_max,
                                    &s1_output_scale, &s1_output_zero_point);
      }
      TF_LITE_ENSURE_OK(context_,
                        AddTensorInput(lite_input_index, false, tensor_flags));
      const float value3f = 1.f / 3.f;
      TF_LITE_ENSURE_OK(context_,
                        AddSingleValueConstantTensor(value3f, is_quantized));
      TF_LITE_ENSURE_OK(context_, AddScalarInt32Operand(NEURON_FUSED_RELU1));
      TF_LITE_ENSURE_OK(
          context_,
          AddAdditionalOutputTensor(
              tensor.dims->size, reinterpret_cast<uint32_t*>(tensor.dims->data),
              nn_type, s1_output_scale, s1_output_zero_point,
              &s1_out_ann_index));
      TF_LITE_ENSURE_OK(context_, FinalizeAddOperation(NEURON_MUL));
    }

    // Stage2 : s2 = x / 2
    float s2_output_min = input_min / 2.f;
    float s2_output_max = input_max / 2.f;
    int s2_out_ann_index = 0;
    {
      float s2_output_scale = input_scale / 2.0f;
      int s2_output_zero_point = input_zero_point;
      TF_LITE_ENSURE_OK(context_,
                        AddTensorInput(lite_input_index, false, tensor_flags));
      const float value2f = 0.5f;
      TF_LITE_ENSURE_OK(context_,
                        AddSingleValueConstantTensor(value2f, is_quantized));
      TF_LITE_ENSURE_OK(context_, AddScalarInt32Operand(NEURON_FUSED_NONE));
      TF_LITE_ENSURE_OK(
          context_,
          AddAdditionalOutputTensor(
              tensor.dims->size, reinterpret_cast<uint32_t*>(tensor.dims->data),
              nn_type, s2_output_scale, s2_output_zero_point,
              &s2_out_ann_index));
      TF_LITE_ENSURE_OK(context_, FinalizeAddOperation(NEURON_MUL));
    }

    // Stage 3 : s3 = s1 * s2
    int s3_out_ann_index = 0;
    {
      augmented_inputs_.push_back(s1_out_ann_index);
      augmented_inputs_.push_back(s2_out_ann_index);
      TF_LITE_ENSURE_OK(context_, AddScalarInt32Operand(NEURON_FUSED_NONE));
      float s3_output_scale = 0.f;
      int s3_output_zero_point = 0;
      if (is_quantized) {
        // the min for stage 3 is always 0.0f.
        float s3_output_min = 0.f;
        // the max for stage 3 is max(s1_min * s2_min, s1_max * s3_max).
        float s3_output_max =
            s1_output_max * s2_output_max > s1_output_min * s2_output_min
                ? s1_output_max * s2_output_max
                : s1_output_min * s2_output_min;
        CalculateQuantizationParams(s3_output_min, s3_output_max,
                                    &s3_output_scale, &s3_output_zero_point);
      }
      TF_LITE_ENSURE_OK(
          context_,
          AddAdditionalOutputTensor(
              tensor.dims->size, reinterpret_cast<uint32_t*>(tensor.dims->data),
              nn_type, s3_output_scale, s3_output_zero_point,
              &s3_out_ann_index));
      TF_LITE_ENSURE_OK(context_, FinalizeAddOperation(NEURON_MUL));
    }

    // Stage 4: y = s3 + s2
    {
      augmented_inputs_.push_back(s2_out_ann_index);
      augmented_inputs_.push_back(s3_out_ann_index);
      TF_LITE_ENSURE_OK(context_, AddScalarInt32Operand(NEURON_FUSED_NONE));
      TF_LITE_ENSURE_OK(context_,
                        AddTensorOutput(lite_output_index, tensor_flags));
      TF_LITE_ENSURE_OK(context_, FinalizeAddOperation(NEURON_ADD));
    }

    return kTfLiteOk;
  }

  // Adds a Dequantize operator and replaces the input tensor index with the
  // dequantized version. If the dequantized version of the operator already
  // exists then it is not added again.
  TfLiteStatus AddDequantize(int nn_input_index, int lite_index,
                             TfLiteType dequantized_type) {
    const int neuron_index = operand_mapping_->lite_index_to_neuron(lite_index);
    int dequantized_ann_index = dequantize_mapping_->DequantizedAnnIndex(
        neuron_index, dequantized_type);

    if (dequantized_ann_index == -1) {
      // The dequantized version does not exist yet, it has to be added: a new
      // Dequantize operation is added, yielding a new tensor.
      const TfLiteTensor& tensor = context_->tensors[lite_index];
      NeuronOperandType operand_type{
          NEURON_TENSOR_FLOAT32, static_cast<uint32_t>(tensor.dims->size),
          reinterpret_cast<uint32_t*>(tensor.dims->data), 0.f, 0};
      RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
          context_,
          neuronapi_->NeuronModel_addOperand(nn_model_, &operand_type),
          "adding operand");
      dequantized_ann_index = operand_mapping_->add_new_non_tensor_operand();

      // Add Dequantize operation.
      const uint32_t dequantize_input[1] = {
          static_cast<uint32_t>(neuron_index)};
      const uint32_t dequantize_output[1] = {
          static_cast<uint32_t>(dequantized_ann_index)};
      RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
          context_,
          neuronapi_->NeuronModel_addOperation(nn_model_, NEURON_DEQUANTIZE, 1,
                                               dequantize_input, 1,
                                               dequantize_output),
          "adding operation");
      dequantize_mapping_->Add(neuron_index, dequantized_type,
                               dequantized_ann_index);
    }

    // The input for the original operation is modified so that the operation
    // now uses the dequantized tensor as input.
    augmented_inputs_[nn_input_index] = dequantized_ann_index;

    return kTfLiteOk;
  }

  // Finish emitting the op (of type `type`) into the Neuron.
  TfLiteStatus FinalizeAddOperation(NeuronOperationType type) {
    // Actually add a Neuron operation
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context_,
        neuronapi_->NeuronModel_addOperation(
            nn_model_, type, static_cast<uint32_t>(augmented_inputs_.size()),
            augmented_inputs_.data(),
            static_cast<uint32_t>(augmented_outputs_.size()),
            augmented_outputs_.data()),
        "adding operation");
    augmented_inputs_.clear();
    augmented_outputs_.clear();
    return kTfLiteOk;
  }

  TfLiteStatus AddSingleValueTensorAsScalarOperand(int tensor_index,
                                                   int nn_type) {
    const TfLiteTensor* tensor = &context_->tensors[tensor_index];
    TF_LITE_ENSURE_EQ(context_, NumElements(tensor), 1);

    NeuronOperandType operand_type{.type = nn_type};
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context_, neuronapi_->NeuronModel_addOperand(nn_model_, &operand_type),
        "adding operand");
    int neuron_tensor_index =
        operand_mapping_->lite_index_to_neuron(tensor_index);
    if (neuron_tensor_index != -1) {
      augmented_inputs_.push_back(neuron_tensor_index);
      return kTfLiteOk;
    }
    // Allocate a new tensor index
    neuron_tensor_index =
        operand_mapping_->add_new_neuron_tensor_index(tensor_index);
    augmented_inputs_.push_back(neuron_tensor_index);

    const TfLiteType tensor_type = tensor->type;
    TfLiteType nn_type_equivalent;
    TF_LITE_ENSURE_OK(context_, GetEquivalentToNeuronType(context_, nn_type,
                                                          &nn_type_equivalent));
    if (tensor_type != nn_type_equivalent) {
      operand_mapping_->add_type_conversion(tensor_index, nn_type_equivalent);
    }
    return kTfLiteOk;
  }

  template <typename T>
  TfLiteStatus AddNewInputConstantTensor(
      int32_t nn_type, TfLiteType type, const TfLiteIntArray* dims,
      const std::vector<T>& tensor_value,
      const TfLiteQuantizationParams& quant_params, int* tensor_index) {
    TF_LITE_ENSURE_OK(context_,
                      context_->AddTensors(context_, 1, tensor_index));

    TfLiteTensor* new_tensor = &context_->tensors[*tensor_index];
    new_tensor->type = type;
    new_tensor->allocation_type = kTfLiteDynamic;
    new_tensor->params = quant_params;

    // Not removing the new tensor in case of resizing errors since it will
    // be cleared by the context
    TF_LITE_ENSURE_OK(
        context_,
        context_->ResizeTensor(
            context_, new_tensor,
            // Resize Tensor takes ownership of the dims array passed as param
            TfLiteIntArrayCopy(dims)));

    memcpy(new_tensor->data.raw,
           reinterpret_cast<const char*>(tensor_value.data()),
           tensor_value.size() * sizeof(T));

    const uint32_t tensor_rank = static_cast<uint32_t>(dims->size);
    const uint32_t* tensor_dims = reinterpret_cast<const uint32_t*>(dims->data);
    NeuronOperandType operand_type{nn_type, tensor_rank, tensor_dims,
                                   quant_params.scale, quant_params.zero_point};

    const int neuron_tensor_index =
        operand_mapping_->add_delegate_generated_input_neuron_tensors_operand();

    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context_, neuronapi_->NeuronModel_addOperand(nn_model_, &operand_type),
        "adding operand");

    augmented_inputs_.push_back(neuron_tensor_index);

    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context_,
        neuronapi_->NeuronModel_setOperandValue(nn_model_, neuron_tensor_index,
                                                new_tensor->data.raw,
                                                new_tensor->bytes),
        "setting new operand value");

    return kTfLiteOk;
  }

  template <typename T>
  TfLiteStatus AddNewInputConstantTensor(
      int32_t nn_type, TfLiteType type, std::initializer_list<int> dims,
      const std::vector<T>& tensor_value,
      const TfLiteQuantizationParams& quant_params, int* tensor_index) {
    TfLiteIntArray* dim_array = TfLiteIntArrayCreate(dims.size());
    dim_array->size = dims.size();
    std::copy(dims.begin(), dims.end(), dim_array->data);

    const auto result = AddNewInputConstantTensor(
        nn_type, type, dim_array, tensor_value, quant_params, tensor_index);
    TfLiteIntArrayFree(dim_array);
    return result;
  }

 private:
  // Returns a TF Lite type which has the same memory representation as a
  // provided Neuron type.
  TfLiteStatus GetEquivalentToNeuronType(TfLiteContext* context, int nn_type,
                                         TfLiteType* type) {
    switch (nn_type) {
      case NEURON_INT32:
        *type = kTfLiteInt32;
        return kTfLiteOk;
      case NEURON_FLOAT32:
        *type = kTfLiteFloat32;
        return kTfLiteOk;
      default:
        context->ReportError(context,
                             "Neuron Delegate: Can't get an equivalent TF Lite "
                             "type for provided Neuron type: %d.\n",
                             nn_type);
        return kTfLiteError;
    }
  }

  template <typename T>
  TfLiteStatus AddScalarOperand(T value, int32_t nn_type) {
    NeuronOperandType operand_type{.type = nn_type};
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context_, neuronapi_->NeuronModel_addOperand(nn_model_, &operand_type),
        "adding operand");
    const int neuron_index = operand_mapping_->add_new_non_tensor_operand();
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context_,
        neuronapi_->NeuronModel_setOperandValue(nn_model_, neuron_index, &value,
                                                sizeof(T)),
        "setting new operand value");
    augmented_inputs_.push_back(neuron_index);
    return kTfLiteOk;
  }

  template <typename T>
  TfLiteStatus AddVectorOperand(const T* values, uint32_t num_values,
                                int32_t nn_type, float scale,
                                int32_t zero_point) {
    NeuronOperandType operand_type{.type = nn_type,
                                   .dimensionCount = 1,
                                   .dimensions = &num_values,
                                   .scale = scale,
                                   .zeroPoint = zero_point};

    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context_, neuronapi_->NeuronModel_addOperand(nn_model_, &operand_type),
        "adding operand");

    const int neuron_index = operand_mapping_->add_new_non_tensor_operand();
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context_,
        neuronapi_->NeuronModel_setOperandValue(nn_model_, neuron_index, values,
                                                sizeof(T) * num_values),
        "settings new operand value");
    augmented_inputs_.push_back(neuron_index);
    return kTfLiteOk;
  }

  template <typename T>
  TfLiteStatus AddVectorOperand(const T* values, uint32_t num_values,
                                int32_t nn_type) {
    return AddVectorOperand(values, num_values, nn_type, /*scale=*/0.f,
                            /*zero_point=*/0);
  }

  TfLiteStatus AddFloat32OutputTensor(uint32_t dimension_count,
                                      const uint32_t* dimension_data,
                                      int* ann_index_out) {
    return AddAdditionalOutputTensor(
        dimension_count, dimension_data, NEURON_TENSOR_FLOAT32,
        /*scale=*/0.f, /*zero_point=*/0, ann_index_out);
  }

  TfLiteStatus AddAdditionalOutputTensor(uint32_t dimension_count,
                                         const uint32_t* dimension_data,
                                         int32_t nn_type, float scale,
                                         int32_t zero_point,
                                         int* ann_index_out) {
    NeuronOperandType operand_type{
        .type = nn_type,
        .dimensionCount = dimension_count,
        .dimensions = dimension_data,
        .scale = scale,
        .zeroPoint = zero_point,
    };
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context_, neuronapi_->NeuronModel_addOperand(nn_model_, &operand_type),
        "adding operand");
    const int neuron_index = operand_mapping_->add_new_non_tensor_operand();
    augmented_outputs_.push_back(neuron_index);
    if (ann_index_out) *ann_index_out = neuron_index;
    return kTfLiteOk;
  }

  // Adds a new Neuron tensor that shadows the TF Lite tensor
  // `tensor_index`. This returns the Neuron tensor index corresponding to
  // the created tensor. If another caller previously created a Neuron
  // tensor for `tensor_index` then the existing one is returned.
  TfLiteStatus AddTensor(int tensor_index, bool hybrid_op,
                         std::vector<uint32_t>* indices, int tensor_flags = 0) {
    const bool scalar_as_tensor =
        tensor_flags & NN_TENSOR_FLAG_SCALAR_AS_TENSOR;
    const bool need_int8_conversion =
        tensor_flags & NN_TENSOR_FLAG_INT8_CONVERSION;
    const bool use_int8_asymm_signed =
        tensor_flags & NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED;
    const bool force_per_channel =
        tensor_flags & NN_TENSOR_FLAG_FORCE_PER_CHANNEL;
    int neuron_tensor_index =
        operand_mapping_->lite_index_to_neuron(tensor_index);
    if (neuron_tensor_index != -1) {
      indices->push_back(neuron_tensor_index);
      return kTfLiteOk;
    }

    // Allocate a new tensor index
    neuron_tensor_index =
        operand_mapping_->add_new_neuron_tensor_index(tensor_index);

    // Parameters needed for new type.
    int32_t nn_type = 0;
    float scale = 0.0f;
    int32_t zeroPoint = 0;
    NeuronSymmPerChannelQuantParams ann_perchannel_params;
    TfLiteTensor* tensor = &context_->tensors[tensor_index];
    TfLiteType tensor_type = tensor->type;
    if (hybrid_op && (tensor_type == kTfLiteUInt8)) {
      // For legacy reason, UINT8 weights in hybrid operators are actually INT8
      // values and should be interpreted as such.
      tensor_type = kTfLiteInt8;
    }
    switch (tensor_type) {
      case kTfLiteNoType:
        // Tensors added during initialization of Ops don't have a type yet and
        // should not be registered with the Neuron.
        indices->push_back(-1);
        return kTfLiteOk;
      case kTfLiteFloat32:
        nn_type = NEURON_TENSOR_FLOAT32;
        break;
      case kTfLiteUInt8:
        nn_type = NEURON_TENSOR_QUANT8_ASYMM;
        scale = tensor->params.scale;
        zeroPoint = tensor->params.zero_point;
        if (scale == 0) {
          // TENSOR_QUANT8_ASYMM and NEURON_TENSOR_QUANT8_ASYMM
          // with zero scale are not valid in Neuron.
          scale = 1;
        }
        break;
      case kTfLiteInt8:
        // If explicit int8 conversion is needed, we still need
        // NEURON_TENSOR_QUANT8_ASYMM type.
        if (use_int8_asymm_signed) {
          nn_type = NEURON_TENSOR_QUANT8_ASYMM_SIGNED;
        } else if (need_int8_conversion) {
          nn_type = NEURON_TENSOR_QUANT8_ASYMM;
        } else {
          nn_type = NEURON_TENSOR_QUANT8_SYMM;
        }
        scale = tensor->params.scale;
        zeroPoint = tensor->params.zero_point;
        if (tensor->quantization.type == kTfLiteAffineQuantization) {
          TfLiteAffineQuantization* quantization_params =
              static_cast<TfLiteAffineQuantization*>(
                  tensor->quantization.params);
          if (quantization_params->scale->size > 1 || force_per_channel) {
            // Set up per-channel quantization.
            ann_perchannel_params = {
                .channelDim = static_cast<uint32_t>(
                    quantization_params->quantized_dimension),
                .scaleCount =
                    static_cast<uint32_t>(quantization_params->scale->size),
                .scales = quantization_params->scale->data,
            };
            nn_type = NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL;
            scale = 0.0f;
            zeroPoint = 0;
          } else if (quantization_params->scale->size == 1) {
            scale = quantization_params->scale->data[0];
            zeroPoint = quantization_params->zero_point->data[0];
          }
        }
        if (nn_type != NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL) {
          if (need_int8_conversion) {
            zeroPoint += 128;
            operand_mapping_->add_type_conversion(tensor_index, kTfLiteUInt8);
          }
          if (scale == 0) {
            // QUANT8 tensors with zero scale are not valid in Neuron.
            scale = 1;
          }
        }
        break;
      case kTfLiteInt32:
        nn_type = NEURON_TENSOR_INT32;
        scale = tensor->params.scale;
        zeroPoint = tensor->params.zero_point;
        break;
      case kTfLiteBool:
        nn_type = NEURON_TENSOR_BOOL8;
        break;
      case kTfLiteInt16:
        nn_type = NEURON_TENSOR_QUANT16_SYMM;
        scale = tensor->params.scale;
        zeroPoint = tensor->params.zero_point;
        break;
      default:
        context_->ReportError(
            context_, "Failed to add Neuron tensor: type %s is not supported.",
            TfLiteTypeGetName(tensor_type));
        return kTfLiteError;
    }
    uint32_t tensor_rank = static_cast<uint32_t>(tensor->dims->size);
    uint32_t* tensor_dims = reinterpret_cast<uint32_t*>(tensor->dims->data);
    if (scalar_as_tensor && tensor_rank == 0) {
      // Use rank 1, shape {1} operand for TFLite scalar tensors.
      tensor_rank = 1;
      tensor_dims = &tensor_rank;
    }
    if (tensor_rank == 0) {
      // if the tensor_rank is 0, the dimension ptr must be nullptr.
      tensor_dims = nullptr;
    }

    NeuronOperandType operand_type{nn_type, tensor_rank, tensor_dims, scale,
                                   zeroPoint};
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context_, neuronapi_->NeuronModel_addOperand(nn_model_, &operand_type),
        "adding operand");
    if (nn_type == NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL) {
      RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
          context_,
          neuronapi_->NeuronModel_setOperandSymmPerChannelQuantParams(
              nn_model_, neuron_tensor_index, &ann_perchannel_params),
          "setting new operand per channel quantization params");
    }
    if (tensor->allocation_type == kTfLiteMmapRo) {
      if (IsQuantized(tensor_type) && need_int8_conversion) {
        // We need to to add a tensor and convert the weights into uint8.
        // Currently this is only needed for fully_connected. The new_tensor is
        // needed for lifetime management for the converted weights.
        int new_tensor_index = -1;
        TF_LITE_ENSURE_OK(context_,
                          context_->AddTensors(context_, 1, &new_tensor_index));
        TfLiteTensor* new_tensor = &context_->tensors[new_tensor_index];
        new_tensor->type = kTfLiteUInt8;
        new_tensor->allocation_type = kTfLiteDynamic;
        new_tensor->params.scale = scale;
        new_tensor->params.zero_point = zeroPoint;
        // Not removing the new tensor in case of resizing errors since it will
        // be cleared by the context
        TF_LITE_ENSURE_OK(
            context_, context_->ResizeTensor(context_, new_tensor,
                                             // Resize Tensor takes ownership of
                                             // the dims array passed as param
                                             TfLiteIntArrayCopy(tensor->dims)));
        // Convert the int8 value into corresponding uint8 value;
        const auto num_elements = NumElements(tensor);
        for (int i = 0; i < num_elements; ++i) {
          new_tensor->data.uint8[i] = static_cast<const uint8_t>(
              static_cast<int32_t>(tensor->data.int8[i]) + 128);
        }
        RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
            context_,
            neuronapi_->NeuronModel_setOperandValue(
                nn_model_, neuron_tensor_index, new_tensor->data.raw,
                new_tensor->bytes),
            "setting new operand value");
      } else {
        RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
            context_,
            neuronapi_->NeuronModel_setOperandValue(
                nn_model_, neuron_tensor_index, tensor->data.raw,
                tensor->bytes),
            "setting new operand value");
      }
    }

    indices->push_back(neuron_tensor_index);
    return kTfLiteOk;
  }

  // Access to Neuron.
  const NeuronApi* const neuronapi_;

  // TfLiteContext for error handling.
  TfLiteContext* const context_;

  // Tracks relationship between indices.
  OperandMapping* const operand_mapping_;

  // Keeps mapping of ANN quantized tensor and float data type to equivalent
  // dequantized ANN tensor. For example, tensor #4 (UINT8) + FLOAT32 could map
  // to tensor #10 (FLOAT32) because a DEQUANTIZE operator was added to convert
  // tensor #4 to a FLOAT32 tensor.
  DequantizeMapping* const dequantize_mapping_;

  // The Neuron model.
  NeuronModel* const nn_model_;

  // Inputs and outputs for the current op. These are augmented in the sense
  // that Neuron uses operands for all arguments, not just tensors, unlike
  // TensorFlow Lite.
  std::vector<uint32_t> augmented_inputs_;
  std::vector<uint32_t> augmented_outputs_;
};

}  // namespace neuron
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_BUILDER_H_
