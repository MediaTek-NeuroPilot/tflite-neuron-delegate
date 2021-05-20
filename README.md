# tflite-neuron-delegate
MediaTek's TFLite delegate

0. Install the Android SDK and NDK. Set environmental variables `ANDROID_HOME` and `ANDROID_NDK_HOME` to SDK and NDK roots respectively.
1. `bazel build -c opt  //neuron/java:tensorflow-lite-neuron` then you can get `bazel-bin/neuron/java/tensorflow-lite-neuron.aar` which can be used with Android app
2. (optional) You can build command line tools ([`benchmark_model`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark), [`imagenet_classification_eval`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification), [`coco_object_detection_eval`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/coco_object_detection), and [`inference_diff`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/inference_diff) ) with neuron_delegate enabled.
```
bazel build --config android_arm64 -c opt //neuron:benchmark_model_plus_neuron_delegate
bazel build --config android_arm64 -c opt //neuron:coco_object_detection_eval_plus_neuron_delegate
bazel build --config android_arm64 -c opt //neuron:imagenet_classification_eval_plus_neuron_delegate
bazel build --config android_arm64 -c opt //neuron:inference_diff_plus_neuron_delegate
bazel build --config android_arm64 -c opt //neuron:label_image_plus_neuron_delegate
```
or
```
bazel build --config android_arm64 -c opt //neuron:all
```
