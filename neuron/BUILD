load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts")
load("@org_tensorflow//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//neuron:build_def.bzl", "android_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "neuronapi_lib",
    hdrs = [
        "neuron_types.h",
    ],
)

cc_library(
    name = "neuron_delegate_utils",
    hdrs = [
        "neuron_delegate_utils.h",
    ],
)

cc_library(
    name = "neuron_implementation",
    srcs = ["neuron_implementation.cc"],
    hdrs = [
        "neuron_implementation.h",
    ],
    linkopts = android_linkopts(),
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        ":neuronapi_lib",
    ],
)

cc_library(
    name = "neuron_performance",
    srcs = [
        "lib/libneuron_performance.a",
    ],
    hdrs = ["neuron_performance.h"],
)

cc_library(
    name = "neuron_delegate_kernel",
    srcs = [
        "mtk/mtk_utils.cc",
        "mtk/mtk_utils.h",
        "neuron_delegate.h",
        "neuron_delegate_builder.h",
        "neuron_delegate_kernel.cc",
        "neuron_nnapi_delegate_kernel.h",
        "neuron_performance.h",
        "neuropilot_trace.h",
    ],
    hdrs = [
        "APUWareUtilsApi.h",
        "neuron_delegate_kernel.h",
    ],
    linkopts = android_linkopts(),
    deps = [
        "neuron_delegate_utils",
        "neuron_delegate_validation",
        "neuron_implementation",
        ":neuron_performance",
        "@farmhash_archive//:farmhash",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/core/api",
        "@org_tensorflow//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "@org_tensorflow//tensorflow/lite/delegates/utils:simple_delegate",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
    ],
)

cc_library(
    name = "neuron_delegate_validation",
    srcs = [
        "neuron_delegate_validation.cc",
    ],
    hdrs = ["neuron_delegate_validation.h"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "neuron_delegate_utils",
        "neuronapi_lib",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/core/api",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "neuron_delegate",
    srcs = [
        "neuron_delegate.cc",
    ],
    hdrs = [
        "APUWareUtilsApi.h",
        "neuron_delegate.h",
    ],
    deps = [
        "neuron_delegate_kernel",
        "neuron_implementation",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/delegates/utils:simple_delegate",
    ],
)

#### The following are for using the dummy test delegate in TFLite tooling ####
cc_library(
    name = "neuron_delegate_provider",
    srcs = ["neuron_delegate_provider.cc"],
    copts = tflite_copts(),
    deps = [
        ":neuron_delegate",
        "@org_tensorflow//tensorflow/lite/tools/delegates:delegate_provider_hdr",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "benchmark_model_plus_neuron_delegate",
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":neuron_delegate_provider",
        "@org_tensorflow//tensorflow/lite/tools/benchmark:benchmark_model_main",
    ],
)

cc_binary(
    name = "inference_diff_plus_neuron_delegate",
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":neuron_delegate_provider",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval_lib",
    ],
)

cc_binary(
    name = "imagenet_classification_eval_plus_neuron_delegate",
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":neuron_delegate_provider",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval_lib",
    ],
)

cc_binary(
    name = "coco_object_detection_eval_plus_neuron_delegate",
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":neuron_delegate_provider",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval_lib",
    ],
)

cc_binary(
    name = "label_image_plus_neuron_delegate",
    srcs = ["@org_tensorflow//tensorflow/lite/examples/label_image:label_image.cc"],
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":neuron_delegate_provider",
        "@org_tensorflow//tensorflow/lite/examples/label_image",
        "@org_tensorflow//tensorflow/lite/examples/label_image:bitmap_helpers",
        "@org_tensorflow//tensorflow/lite/tools/delegates:tflite_execution_providers",
    ],
)

exports_files([
    "neuron_delegate.h",
    "version_script.lds",
])
