/* Copyright Statement:
 *
 * This software/firmware and related documentation ("MediaTek Software") are
 * protected under relevant copyright laws. The information contained herein
 * is confidential and proprietary to MediaTek Inc. and/or its licensors.
 * Without the prior written permission of MediaTek inc. and/or its licensors,
 * any reproduction, modification, use or disclosure of MediaTek Software,
 * and information contained herein, in whole or in part, shall be strictly
 * prohibited.
 */
/* MediaTek Inc. (C) 2019. All rights reserved.
 *
 * BY OPENING THIS FILE, RECEIVER HEREBY UNEQUIVOCALLY ACKNOWLEDGES AND AGREES
 * THAT THE SOFTWARE/FIRMWARE AND ITS DOCUMENTATIONS ("MEDIATEK SOFTWARE")
 * RECEIVED FROM MEDIATEK AND/OR ITS REPRESENTATIVES ARE PROVIDED TO RECEIVER ON
 * AN "AS-IS" BASIS ONLY. MEDIATEK EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE OR NONINFRINGEMENT.
 * NEITHER DOES MEDIATEK PROVIDE ANY WARRANTY WHATSOEVER WITH RESPECT TO THE
 * SOFTWARE OF ANY THIRD PARTY WHICH MAY BE USED BY, INCORPORATED IN, OR
 * SUPPLIED WITH THE MEDIATEK SOFTWARE, AND RECEIVER AGREES TO LOOK ONLY TO SUCH
 * THIRD PARTY FOR ANY WARRANTY CLAIM RELATING THERETO. RECEIVER EXPRESSLY
 * ACKNOWLEDGES THAT IT IS RECEIVER'S SOLE RESPONSIBILITY TO OBTAIN FROM ANY
 * THIRD PARTY ALL PROPER LICENSES CONTAINED IN MEDIATEK SOFTWARE. MEDIATEK
 * SHALL ALSO NOT BE RESPONSIBLE FOR ANY MEDIATEK SOFTWARE RELEASES MADE TO
 * RECEIVER'S SPECIFICATION OR TO CONFORM TO A PARTICULAR STANDARD OR OPEN
 * FORUM. RECEIVER'S SOLE AND EXCLUSIVE REMEDY AND MEDIATEK'S ENTIRE AND
 * CUMULATIVE LIABILITY WITH RESPECT TO THE MEDIATEK SOFTWARE RELEASED HEREUNDER
 * WILL BE, AT MEDIATEK'S OPTION, TO REVISE OR REPLACE THE MEDIATEK SOFTWARE AT
 * ISSUE, OR REFUND ANY SOFTWARE LICENSE FEES OR SERVICE CHARGE PAID BY RECEIVER
 * TO MEDIATEK FOR SUCH MEDIATEK SOFTWARE AT ISSUE.
 *
 * The following software/firmware and/or related documentation ("MediaTek
 * Software") have been modified by MediaTek Inc. All revisions are subject to
 * any receiver's applicable license agreements with MediaTek Inc.
 */

#define LOG_TAG "MtkUtils"
#include "neuron/mtk/mtk_utils.h"

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/minimal_logging.h"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif  // __ANDROID__

#define MAX_OEM_OP_STRING_LEN 100

namespace tflite {
namespace mtk {

/*
 * The format of data will be:
 *  -------------------------------------------------------------------------------
 *  | 1 byte typeLen  | N bytes type     | 4 bytes dataLen  | N bytes data |
 *  -------------------------------------------------------------------------------
 */
static int EncodeOperandValue(OemOperandValue* operand, uint8_t* output) {
  size_t currPos = 0;

  // 1 byte for typeLen, 4 bytes for bufferLen
  if (output == nullptr) {
    return -1;
  }

  // Set length of type
  *output = operand->typeLen;
  currPos += sizeof(uint8_t);

  // Copy type to output
  memcpy(output + currPos, operand->type, operand->typeLen);
  currPos += operand->typeLen;

  // Set the length of buffer
  uint32_t* dataLen = reinterpret_cast<uint32_t*>(&output[currPos]);
  *dataLen = operand->dataLen;
  currPos += sizeof(uint32_t);

  // Copy  operand value to output
  memcpy(&output[currPos], operand->data, operand->dataLen);

  return 0;
}

size_t PackOemScalarString(const char* str, uint8_t** out_buffer) {
  if (str == nullptr) {
    return 0;
  }
  size_t out_len = 0;
  uint8_t type[] = {'s', 't', 'r', 'i', 'n', 'g'};
  OemOperandValue operand_value;

  operand_value.typeLen = sizeof(type);
  operand_value.type = type;
  operand_value.dataLen = strlen(str);
  if (operand_value.dataLen > MAX_OEM_OP_STRING_LEN) {
    return 0;
  }
  operand_value.data =
      reinterpret_cast<uint8_t*>(malloc(operand_value.dataLen));
  if (operand_value.data == nullptr) {
    return 0;
  }
  memcpy(operand_value.data, str, operand_value.dataLen);

  out_len =
      operand_value.typeLen + operand_value.dataLen + (sizeof(size_t) * 2);
  *out_buffer = reinterpret_cast<uint8_t*>(calloc(out_len, sizeof(uint8_t)));
  if (*out_buffer == nullptr) {
    free(operand_value.data);
    return 0;
  }
  EncodeOperandValue(&operand_value, *out_buffer);
  free(operand_value.data);

  // TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
  //    "PackOemScalarString: %s, buffer size:%zu", str, out_len);
  return out_len;
}

}  // namespace mtk
}  // namespace tflite
