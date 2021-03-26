load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_jni_binary")
load("@org_tensorflow//tensorflow/lite/java:aar_with_jni.bzl", "aar_with_jni")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# EXPERIMENTAL: Native target that runs inference on the Neuron backend.
# The Neuron backend-related targets are intentionally not included in BUILD.bazel.
tflite_jni_binary(
    name = "libtensorflowlite_neuron_jni.so",
    linkscript = "//neuron:version_script.lds",
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//neuron/java/src/main/native",
    ],
)

cc_library(
    name = "tensorflowlite_neuron",
    srcs = [
        "libtensorflowlite_neuron_jni.so",
    ] + select({
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
)

android_library(
    name = "tensorflowlite_java_neuron",
    srcs = ["//neuron/java/src/main/java/org/tensorflow/lite:neuron_delegate"],
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":tensorflowlite_neuron",
        "@org_tensorflow//tensorflow/lite/java:tensorflowlite_java",
        "@org_checkerframework_qual",
    ],
)

aar_with_jni(
    name = "tensorflow-lite-neuron",
    android_library = ":tensorflowlite_java_neuron",
    headers = ["//neuron:neuron_delegate.h"],
)
