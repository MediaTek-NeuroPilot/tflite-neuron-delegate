workspace(name = "neuron_delegate")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "org_tensorflow",
    patch_args = ["-p1"],
    patches = ["//third_party:tf_hacks_for_gpu_delegates.diff"],
    sha256 = "e3d0ee227cc19bd0fa34a4539c8a540b40f937e561b4580d4bbb7f0e31c6a713",
    strip_prefix = "tensorflow-2.5.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/v2.5.0.zip",
    ],
)

# Initialize tensorflow workspace.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

# Android.
android_sdk_repository(
    name = "androidsdk",
    api_level = 28,
)

android_ndk_repository(
    name = "androidndk",
)

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("3.1.0")
