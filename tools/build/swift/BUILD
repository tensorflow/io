load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

package(default_visibility = ["//visibility:public"])

swift_library(
    name = "audio_swift",
    srcs = [
        "//tensorflow_io/core:swift/audio.swift",
    ],
    linkopts = [
        "-L/usr/lib/swift",
    ],
    module_name = "audio",
    alwayslink = True,
)

swift_library(
    name = "video_swift",
    srcs = [
        "//tensorflow_io/core:swift/video.swift",
    ],
    linkopts = [
        "-L/usr/lib/swift",
    ],
    module_name = "video",
    alwayslink = True,
)
