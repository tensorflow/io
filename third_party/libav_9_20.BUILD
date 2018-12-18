# Description:
#   FFmpeg

licenses(["notice"])  # LGPL v2.1+ license

exports_files(["LICENSE.md"])

cc_library(
    name = "libav",
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
        "libavutil/version.h",
        "libavutil/mem.h",
        "libavutil/error.h",
        "libavutil/rational.h",
        "libavutil/mathematics.h",
        "libavutil/intfloat.h",
        "libavutil/log.h",
        "libavutil/pixfmt.h",
        "libavutil/old_pix_fmts.h",
        "libavutil/cpu.h",
        "libavutil/channel_layout.h",
        "libavutil/dict.h",
        "libavutil/imgutils.h",
        "libavutil/pixdesc.h",
        "libavutil/time.h",
        "libswscale/swscale.h",
        "libswscale/version.h",
    ],
    copts = [],
    defines = [],
    includes = [],
    linkopts = [
        "-L$(GENDIR)/external/libav_9_20",
        "-l:libavformat.so.54",
        "-l:libavcodec.so.54",
        "-l:libavutil.so.52",
        "-l:libswscale.so.2",
    ],
    visibility = ["//visibility:public"],
    deps = [],
    data = [
        "libavformat.so.54",
        "libavcodec.so.54",
        "libavutil.so.52",
        "libswscale.so.2",
    ],
)

# Stab library files for build to be successful
# even when those files are not installed (e.g., Ubuntu 14.04)
# In runtime (e.g., Ubuntu 18.04) system files will be used.
genrule(
    name = "libavformat_so_54",
    outs = ["libavformat.so.54"],
    cmd = "echo '' | g++ -shared -fPIC -x c++ - -o $@",
)

genrule(
    name = "libavcodec_so_54",
    outs = ["libavcodec.so.54"],
    cmd = "echo '' | g++ -shared -fPIC -x c++ - -o $@",
)

genrule(
    name = "libavutil_so_52",
    outs = ["libavutil.so.52"],
    cmd = "echo '' | g++ -shared -fPIC -x c++ - -o $@",
)

genrule(
    name = "libswscale_so_2",
    outs = ["libswscale.so.2"],
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
