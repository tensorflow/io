# Description:
#   FFmpeg

licenses(["notice"])  # LGPL v2.1+ license

exports_files(["LICENSE.md"])

cc_library(
    name = "ffmpeg",
    srcs = [
    ],
    hdrs = [
        "libavformat/avformat.h",
        "libavformat/avio.h",
        "libavformat/version.h",
        "libavcodec/version.h",
        "libavcodec/avcodec.h",
        "libavutil/avconfig.h",
        "libavutil/samplefmt.h",
        "libavutil/avutil.h",
        "libavutil/common.h",
        "libavutil/attributes.h",
        "libavutil/macros.h",
        "libavutil/version.h",
        "libavutil/mem.h",
        "libavutil/error.h",
        "libavutil/rational.h",
        "libavutil/mathematics.h",
        "libavutil/intfloat.h",
        "libavutil/log.h",
        "libavutil/pixfmt.h",
        "libavutil/buffer.h",
        "libavutil/cpu.h",
        "libavutil/channel_layout.h",
        "libavutil/dict.h",
        "libavutil/frame.h",
        "libavutil/imgutils.h",
        "libavutil/pixdesc.h",
        "libswscale/swscale.h",
        "libswscale/version.h",
    ],
    copts = [],
    defines = [],
    includes = [],
    linkopts = [
        "-l:libavformat.so.57",
        "-l:libavcodec.so.57",
        "-l:libavutil.so.55",
        "-l:libswscale.so.4",
    ],
    visibility = ["//visibility:public"],
    deps = [],
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
