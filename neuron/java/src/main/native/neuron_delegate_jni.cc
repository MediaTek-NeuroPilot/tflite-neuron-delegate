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

#include <jni.h>
#include <sstream>

#include "neuron/neuron_delegate.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL Java_org_tensorflow_lite_NeuronDelegate_createDelegate(
    JNIEnv* env, jclass clazz, jint preference, jboolean allow_fp16) {
  auto options = TfLiteNeuronDelegateOptionsDefault();
  options.execution_preference = (ExecutionPreference)preference;
  options.allow_fp16 = allow_fp16;
  return reinterpret_cast<jlong>(TfLiteNeuronDelegateCreate(&options));
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_NeuronDelegate_deleteDelegate(
    JNIEnv* env, jclass clazz, jlong delegate) {
  TfLiteNeuronDelegateDelete(reinterpret_cast<TfLiteDelegate*>(delegate));
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
