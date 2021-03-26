# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Neuron delegate Java API using the TensorFlow Lite CC library.

package(default_visibility = ["//visibility:public"])

load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts")

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "native",
    srcs = ["neuron_delegate_jni.cc"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//neuron:neuron_delegate",
        "@org_tensorflow//tensorflow/lite/java/jni",
    ],
    alwayslink = 1,
)
