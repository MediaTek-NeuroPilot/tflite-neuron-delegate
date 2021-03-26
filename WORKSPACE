workspace(name = "neuron_delegate")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "org_tensorflow",
    sha256 = "d22d64a6de3f2970ca3daba241912721c94c7c7aa0e6d2026cd43eb6b1ce93d2",
    strip_prefix = "tensorflow-9f069e02551098bb05e7cf49ddc950411bebd4e8",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/9f069e02551098bb05e7cf49ddc950411bebd4e8.zip",
    ],
)

# TensorFlow build depends on following dependencies.
# Needs to be in-sync with TensorFlow sources.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "7d206c2383811f378a5ef03f4aacbcf5f47fd8650f6abbc3fa89f3a27dd8b176",
    strip_prefix = "rules_closure-0.10.0",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/0.10.0.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/0.10.0.tar.gz",
    ],
)

load("@io_bazel_rules_closure//closure:repositories.bzl", "rules_closure_dependencies", "rules_closure_toolchains")

rules_closure_dependencies()

rules_closure_toolchains()

http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

# Initialize tensorflow workspace.
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace(
    path_prefix = "",
    tf_repo_name = "org_tensorflow",
)

# Android.
android_sdk_repository(
    name = "androidsdk",
    api_level = 28,
)

android_ndk_repository(
    name = "androidndk",
)

RULES_JVM_EXTERNAL_TAG = "2.10"

RULES_JVM_EXTERNAL_SHA = "1bbf2e48d07686707dd85357e9a94da775e1dbd7c464272b3664283c9c716d26"

http_archive(
    name = "rules_jvm_external",
    sha256 = RULES_JVM_EXTERNAL_SHA,
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
    url = "https://github.com/bazelbuild/rules_jvm_external/archive/%s.zip" % RULES_JVM_EXTERNAL_TAG,
)

load("@rules_jvm_external//:defs.bzl", "maven_install")

http_archive(
    name = "com_google_protobuf_javalite",
    sha256 = "757038e6363ec3ad9df4f9548105289767e03f8c1efb000181cafa16ccdf2e69",
    strip_prefix = "protobuf-javalite",
    urls = [
        "https://mirror.bazel.build/github.com/google/protobuf/archive/javalite.zip",
        "https://github.com/google/protobuf/archive/javalite.zip",
    ],
)

http_archive(
    name = "build_bazel_rules_android",
    sha256 = "cd06d15dd8bb59926e4d65f9003bfc20f9da4b2519985c27e190cddc8b7a7806",
    strip_prefix = "rules_android-0.1.1",
    urls = ["https://github.com/bazelbuild/rules_android/archive/v0.1.1.zip"],
)

http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "7a7afdd4869bb201c9352eed2daf37294d42b093579b70423490c1b4d4f6ce42",
    urls = ["https://github.com/bazelbuild/rules_apple/releases/download/0.19.0/rules_apple.0.19.0.tar.gz"],
)

load("@build_bazel_rules_apple//apple:repositories.bzl", "apple_rules_dependencies")

apple_rules_dependencies()

http_archive(
    name = "build_bazel_rules_swift",
    sha256 = "18cd4df4e410b0439a4935f9ca035bd979993d42372ba79e7f2d4fafe9596ef0",
    urls = ["https://github.com/bazelbuild/rules_swift/releases/download/0.12.1/rules_swift.0.12.1.tar.gz"],
)

load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")

swift_rules_dependencies()

http_archive(
    name = "rules_python",
    sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("0.24.1")
