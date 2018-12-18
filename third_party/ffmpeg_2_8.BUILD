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
    copts = [],
    defines = [],
    includes = [],
    linkopts = [
        "-L$(GENDIR)/external/ffmpeg_2_8",
        "-l:libavformat-ffmpeg.so.56",
        "-l:libavcodec-ffmpeg.so.56",
        "-l:libavutil-ffmpeg.so.54",
        "-l:libswscale-ffmpeg.so.3",
    ],
    visibility = ["//visibility:public"],
    deps = [],
    data = [
        "libavformat-ffmpeg.so.56",
        "libavcodec-ffmpeg.so.56",
        "libavutil-ffmpeg.so.54",
        "libswscale-ffmpeg.so.3",
    ],
)

# Stab library files for build to be successful
# even when those files are not installed (e.g., Ubuntu 14.04)
# In runtime (e.g., Ubuntu 18.04) system files will be used.
genrule(
    name = "libavformat-ffmpeg_so_56",
    outs = ["libavformat-ffmpeg.so.56"],
    cmd = "echo '' | g++ -shared -fPIC -x c++ - -o $@",
)

genrule(
    name = "libavcodec-ffmpeg_so_56",
    outs = ["libavcodec-ffmpeg.so.56"],
    cmd = "echo '' | g++ -shared -fPIC -x c++ - -o $@",
)

genrule(
    name = "libavutil-ffmpeg_so_54",
    outs = ["libavutil-ffmpeg.so.54"],
    cmd = "echo '' | g++ -shared -fPIC -x c++ - -o $@",
)

genrule(
    name = "libswscale-ffmpeg_so_3",
    outs = ["libswscale-ffmpeg.so.3"],
    cmd = "echo '' | g++ -shared -fPIC -x c++ - -o $@",
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
