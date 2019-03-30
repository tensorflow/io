# Description:
#   FFmpeg

licenses(["notice"])  # LGPL v2.1+ license

exports_files(["LICENSE.md"])

cc_import(
    name = "ffmpeg",
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
