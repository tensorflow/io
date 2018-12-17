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
        "-L$(GENDIR)/external/ffmpeg",
        "-l:libavformat.so.57",
        "-l:libavcodec.so.57",
        "-l:libavutil.so.55",
        "-l:libswscale.so.4",
    ],
    visibility = ["//visibility:public"],
    deps = [],
    data = [
        "libavformat.so.57",
        "libavcodec.so.57",
        "libavutil.so.55",
        "libswscale.so.4",
    ],
)

# Stab library files for build to be successful
# even when those files are not installed (e.g., Ubuntu 14.04)
# In runtime (e.g., Ubuntu 18.04) system files will be used.
genrule(
    name = "libavformat_so_57",
    outs = ["libavformat.so.57"],
    cmd = "echo '' | g++ -shared -fPIC -x c++ - -o $@",
)

genrule(
    name = "libavcodec_so_57",
    outs = ["libavcodec.so.57"],
    cmd = "echo '' | g++ -shared -fPIC -x c++ - -o $@",
)

genrule(
    name = "libavutil_so_55",
    outs = ["libavutil.so.55"],
    cmd = "echo '' | g++ -shared -fPIC -x c++ - -o $@",
)

genrule(
    name = "libswscale_so_4",
    outs = ["libswscale.so.4"],
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
