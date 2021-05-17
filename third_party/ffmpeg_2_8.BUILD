# Description:
#   FFmpeg

licenses(["notice"])  # LGPL v2.1+ license

exports_files(["LICENSE.md"])

cc_library(
    name = "ffmpeg",
    srcs = [
        ":stub/libavcodec-ffmpeg.so",
        ":stub/libavformat-ffmpeg.so",
        ":stub/libavutil-ffmpeg.so",
        ":stub/libswscale-ffmpeg.so",
    ],
    hdrs = [
        "libavcodec/avcodec.h",
        "libavcodec/old_codec_ids.h",
        "libavcodec/version.h",
        "libavformat/avformat.h",
        "libavformat/avio.h",
        "libavformat/version.h",
        "libavutil/attributes.h",
        "libavutil/avconfig.h",
        "libavutil/avutil.h",
        "libavutil/buffer.h",
        "libavutil/channel_layout.h",
        "libavutil/common.h",
        "libavutil/cpu.h",
        "libavutil/dict.h",
        "libavutil/error.h",
        "libavutil/frame.h",
        "libavutil/imgutils.h",
        "libavutil/intfloat.h",
        "libavutil/log.h",
        "libavutil/macros.h",
        "libavutil/mathematics.h",
        "libavutil/mem.h",
        "libavutil/old_pix_fmts.h",
        "libavutil/pixdesc.h",
        "libavutil/pixfmt.h",
        "libavutil/rational.h",
        "libavutil/samplefmt.h",
        "libavutil/version.h",
        "libswscale/swscale.h",
        "libswscale/version.h",
    ],
    visibility = ["//visibility:public"],
)

genrule(
    name = "libavutil_avconfig_h",
    outs = ["libavutil/avconfig.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#ifndef AVUTIL_AVCONFIG_H",
        "#define AVUTIL_AVCONFIG_H",
        "#define AV_HAVE_BIGENDIAN 0",
        "#define AV_HAVE_FAST_UNALIGNED 1",
        "#endif /* AVUTIL_AVCONFIG_H */",
        "EOF",
    ]),
)

cc_binary(
    name = "stub/libavformat-ffmpeg.so",
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin": [
            "-install_name",
            "libavformat-ffmpeg.56.dylib",
        ],
        "//conditions:default": [
            "-Wl,--disable-new-dtags",
            "-Wl,-soname,libavformat-ffmpeg.so.56",
        ],
    }),
    linkshared = 1,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "stub/libavcodec-ffmpeg.so",
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin": [
            "-install_name",
            "libavcodec-ffmpeg.56.dylib",
        ],
        "//conditions:default": [
            "-Wl,--disable-new-dtags",
            "-Wl,-soname,libavcodec-ffmpeg.so.56",
        ],
    }),
    linkshared = 1,
)

cc_binary(
    name = "stub/libavutil-ffmpeg.so",
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin": [
            "-install_name",
            "libavutil-ffmpeg.54.dylib",
        ],
        "//conditions:default": [
            "-Wl,--disable-new-dtags",
            "-Wl,-soname,libavutil-ffmpeg.so.54",
        ],
    }),
    linkshared = 1,
)

cc_binary(
    name = "stub/libswscale-ffmpeg.so",
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin": [
            "-install_name",
            "libswscale-ffmpeg.3.dylib",
        ],
        "//conditions:default": [
            "-Wl,--disable-new-dtags",
            "-Wl,-soname,libswscale-ffmpeg.so.3",
        ],
    }),
    linkshared = 1,
)
