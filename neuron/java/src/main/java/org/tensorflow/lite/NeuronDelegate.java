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

package org.tensorflow.lite;

import android.content.Context;
import java.io.Closeable;

/** {@link Delegate} for Neuron inference. */
public class NeuronDelegate implements Delegate, Closeable {

  private static final long INVALID_DELEGATE_HANDLE = 0;

  private static final String TFLITE_NEURON_LIB = "tensorflowlite_neuron_jni";

  private long delegateHandle;

  public static final class Options {
    public Options() {}

    /**
     * undefined, specifies default behavior. so far, the default setting of NEURON is
     * EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER
     */
    public static final int EXECUTION_PREFERENCE_UNDEFINED = -1;

    /**
     * Prefer executing in a way that minimizes battery drain. This is desirable for compilations
     * that will be executed often.
     */
    public static final int EXECUTION_PREFERENCE_LOW_POWER = 0;

    /**
     * Prefer returning a single answer as fast as possible, even if this causes more power
     * consumption.
     */
    public static final int EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER = 1;

    /**
     * Prefer maximizing the throughput of successive frames, for example when processing successive
     * frames coming from the camera.
     */
    public static final int EXECUTION_PREFERENCE_SUSTAINED_SPEED = 2;

    /**
     * Sets the inference preference for precision/compilation/runtime tradeoffs.
     *
     * @param preference One of EXECUTION_PREFERENCE_LOW_POWER,
     *     EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER, and EXECUTION_PREFERENCE_SUSTAINED_SPEED.
     */
    public Options setExecutionPreference(int preference) {
      this.executionPreference = preference;
      return this;
    }

    public Options setAllowFp16(boolean enable) {
      this.allowFp16 = enable;
      return this;
    }

    private int executionPreference = EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER;
    private Boolean allowFp16 = null;
  }

  public NeuronDelegate(Options options) {
    TensorFlowLite.init();
    delegateHandle =
        createDelegate(
            options.executionPreference,
            options.allowFp16 != null ? options.allowFp16 : false);
  }

  public NeuronDelegate() {
    this(new Options());
  }

  @Override
  public long getNativeHandle() {
    return delegateHandle;
  }

  /**
   * Frees TFLite resources in C runtime.
   *
   * <p>User is expected to call this method explicitly.
   */
  @Override
  public void close() {
    if (delegateHandle != INVALID_DELEGATE_HANDLE) {
      deleteDelegate(delegateHandle);
      delegateHandle = INVALID_DELEGATE_HANDLE;
    }
  }

  static {
    System.loadLibrary(TFLITE_NEURON_LIB);
  }

  private static native long createDelegate(
    int preference,
    boolean allowFp16);

  private static native void deleteDelegate(long delegateHandle);

}
