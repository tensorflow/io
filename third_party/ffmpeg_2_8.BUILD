# Description:
#   FFmpeg

licenses(["notice"])  # LGPL v2.1+ license

exports_files(["LICENSE.md"])

load("@org_tensorflow//third_party:repo.bzl", "cc_import_library")

cc_import_library(
    name = "ffmpeg",
    hdrs = [
        "libavformat/avformat.h",
        "libavformat/avio.h",
        "libavformat/version.h",
        "libavcodec/version.h",
        "libavcodec/avcodec.h",
        "libavcodec/old_codec_ids.h",
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
        "libavutil/old_pix_fmts.h",
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
    libraries = [
        "libavformat-ffmpeg.so.56",
        "libavcodec-ffmpeg.so.56",
        "libavutil-ffmpeg.so.54",
        "libswscale-ffmpeg.so.3",
    ],
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
