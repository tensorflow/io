# Description:
#   Apache Thrift library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "thrift",
    srcs = [
        "lib/cpp/src/thrift/Thrift.h",
        "lib/cpp/src/thrift/TLogging.h",
        "lib/cpp/src/thrift/TOutput.h",
        "lib/cpp/src/thrift/TBase.h",
        "lib/cpp/src/thrift/TToString.h",
        "lib/cpp/src/thrift/TApplicationException.h",
        "lib/cpp/src/thrift/transport/PlatformSocket.h",
        "lib/cpp/src/thrift/transport/TTransport.h",
        "lib/cpp/src/thrift/transport/TBufferTransports.h",
        "lib/cpp/src/thrift/transport/TBufferTransports.cpp",
        "lib/cpp/src/thrift/transport/TTransportException.h",
        "lib/cpp/src/thrift/transport/TTransportException.cpp",
        "lib/cpp/src/thrift/transport/TVirtualTransport.h",
        "lib/cpp/src/thrift/protocol/TProtocol.h",
        "lib/cpp/src/thrift/protocol/TProtocol.cpp",
        "lib/cpp/src/thrift/protocol/TProtocolException.h",
        "lib/cpp/src/thrift/protocol/TVirtualProtocol.h",
        "lib/cpp/src/thrift/thrift-config.h",
        "lib/cpp/src/thrift/config.h",
        "lib/cpp/src/thrift/stdcxx.h",
        "compiler/cpp/src/thrift/version.h",
    ],
    hdrs = [
        "lib/cpp/src/thrift/protocol/TCompactProtocol.h",
        "lib/cpp/src/thrift/protocol/TCompactProtocol.tcc",
        "lib/cpp/src/thrift/protocol/TDebugProtocol.h",
        "lib/cpp/src/thrift/protocol/TBinaryProtocol.h",
        "lib/cpp/src/thrift/protocol/TBinaryProtocol.tcc",
    ],
    includes = [
        "lib/cpp/src",
    ],
    copts = [
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    deps = [
        "@boost",
    ],
)

genrule(
    name = "version_h",
    srcs = [
        "compiler/cpp/src/thrift/version.h.in",
    ],
    cmd = "sed 's/@PACKAGE_VERSION@/0.11.0/g' $< > $@",
    outs = [
        "compiler/cpp/src/thrift/version.h",
    ],
)

genrule(
    name = "config_h",
    srcs = ["build/cmake/config.h.in"],
    outs = ["lib/cpp/src/thrift/config.h"],
    cmd = ("sed " +
           "-e 's/cmakedefine/define/g' " +
           "-e 's/$${PACKAGE}/thrift/g' " +
           "-e 's/$${PACKAGE_BUGREPORT}//g' " +
           "-e 's/$${PACKAGE_NAME}/thrift/g' " +
           "-e 's/$${PACKAGE_TARNAME}/thrift/g' " +
           "-e 's/$${PACKAGE_URL}//g' " +
           "-e 's/$${PACKAGE_VERSION}/0.11.0/g' " +
           "-e 's/$${PACKAGE_STRING}/thrift 0.11.0/g' " +
           "$< >$@"),
)
