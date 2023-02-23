workspace(name = "neuron_delegate")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "org_tensorflow",
    sha256 = "d2948c066a0bc3f45cb8072def03c85f50af8a75606bbdff91715ef8c5f2a28c",
    strip_prefix = "tensorflow-2.8.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/v2.8.0.zip",
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
    api_level = 31,
)

android_ndk_repository(
    name = "androidndk",
)

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("3.1.0")
