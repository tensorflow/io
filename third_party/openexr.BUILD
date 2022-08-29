package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD 3-Clause license

cc_library(
    name = "openexr",
    srcs = glob(
        [
            "OpenEXR/IlmImf/*.h",
            "OpenEXR/IlmImf/*.cpp",
            "IlmBase/Iex/*.h",
            "IlmBase/Iex/*.cpp",
            "IlmBase/Imath/*.h",
            "IlmBase/Imath/*.cpp",
            "IlmBase/IlmThread/*.h",
            "IlmBase/IlmThread/*.cpp",
            "IlmBase/Half/*.h",
            "IlmBase/Half/*.cpp",
        ],
        exclude = [
            "OpenEXR/IlmImf/b44ExpLogTable.cpp",
            "IlmBase/Half/toFloat.cpp",
            "OpenEXR/IlmImf/dwaLookups.cpp",
            "IlmBase/Half/eLut.cpp",
        ],
    ) + [
        "config/IlmBaseConfig.h",
        "config/OpenEXRConfig.h",
        "config/OpenEXRConfigInternal.h",
    ],
    hdrs = [],
    copts = select({
        "@bazel_tools//src/conditions:windows": [
            "/std:c++14",
        ],
        "//conditions:default": [],
    }),
    defines = select({
        "@bazel_tools//src/conditions:darwin": [],
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "HAVE_POSIX_SEMAPHORES=1",
        ],
    }),
    includes = [
        ".",
        "IlmBase/Half",
        "IlmBase/Iex",
        "IlmBase/IlmThread",
        "IlmBase/Imath",
        "OpenEXR/IlmImf",
        "config",
    ],
    deps = [
        "@org_tensorflow_io//third_party:openexr",
        "@zlib",
    ],
)

genrule(
    name = "IlmBaseConfig_h",
    outs = ["config/IlmBaseConfig.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "",
        "#ifndef INCLUDED_ILMBASE_CONFIG_H",
        "#define INCLUDED_ILMBASE_CONFIG_H 1",
        "#pragma once",
        "#define HAVE_PTHREAD 1",
        "#define ILMBASE_INTERNAL_NAMESPACE_CUSTOM 0",
        "#define IMATH_INTERNAL_NAMESPACE Imath_2_4",
        "#define IEX_INTERNAL_NAMESPACE Iex_2_4",
        "#define ILMTHREAD_INTERNAL_NAMESPACE IlmThread_2_4",
        "#define ILMBASE_NAMESPACE_CUSTOM 0",
        "#define IMATH_NAMESPACE Imath",
        "#define IEX_NAMESPACE Iex",
        "#define ILMTHREAD_NAMESPACE IlmThread",
        "#define ILMBASE_VERSION_STRING \"2.4.0\"",
        "#define ILMBASE_PACKAGE_STRING \"IlmBase 2.4.0\"",
        "#define ILMBASE_VERSION_MAJOR 2",
        "#define ILMBASE_VERSION_MINOR 4",
        "#define ILMBASE_VERSION_PATCH 0",
        "#define ILMBASE_VERSION_HEX ((uint32_t(ILMBASE_VERSION_MAJOR) << 24) | \\",
        "                             (uint32_t(ILMBASE_VERSION_MINOR) << 16) | \\",
        "                             (uint32_t(ILMBASE_VERSION_PATCH) <<  8))",
        "#endif // INCLUDED_ILMBASE_CONFIG_H",
        "",
        "EOF",
    ]),
)

genrule(
    name = "OpenEXRConfig_h",
    outs = ["config/OpenEXRConfig.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "",
        "#ifndef INCLUDED_OPENEXR_CONFIG_H",
        "#define INCLUDED_OPENEXR_CONFIG_H 1",
        "#pragma once",
        "#define OPENEXR_IMF_INTERNAL_NAMESPACE_CUSTOM 0",
        "#define OPENEXR_IMF_INTERNAL_NAMESPACE Imf_2_4",
        "#define OPENEXR_IMF_NAMESPACE_CUSTOM 0",
        "#define OPENEXR_IMF_NAMESPACE Imf",
        "#define OPENEXR_VERSION_STRING \"2.4.0\"",
        "#define OPENEXR_PACKAGE_STRING \"IlmBase \"",
        "#define OPENEXR_VERSION_MAJOR 2",
        "#define OPENEXR_VERSION_MINOR 4",
        "#define OPENEXR_VERSION_PATCH 0",
        "#define OPENEXR_VERSION_HEX ((uint32_t(OPENEXR_VERSION_MAJOR) << 24) | \\",
        "                             (uint32_t(OPENEXR_VERSION_MINOR) << 16) | \\",
        "                             (uint32_t(OPENEXR_VERSION_PATCH) <<  8))",
        "#endif // INCLUDED_OPENEXR_CONFIG_H",
        "",
        "EOF",
    ]),
)

genrule(
    name = "OpenEXRConfigInternal_h",
    outs = ["config/OpenEXRConfigInternal.h"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "",
        "#ifndef INCLUDED_OPENEXR_INTERNAL_CONFIG_H",
        "#define INCLUDED_OPENEXR_INTERNAL_CONFIG_H 1",
        "#pragma once",
        "#define OPENEXR_IMF_HAVE_LINUX_PROCFS 1",
        "#define OPENEXR_IMF_HAVE_COMPLETE_IOMANIP 1",
        "#define OPENEXR_IMF_HAVE_SYSCONF_NPROCESSORS_ONLN 1",
        "#endif // INCLUDED_OPENEXR_INTERNAL_CONFIG_H",
        "",
        "EOF",
    ]),
)
