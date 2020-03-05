load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def tf_io_copts():
    return (
        select({
            "@bazel_tools//src/conditions:windows": [
                "/DEIGEN_STRONG_INLINE=inline",
                "-DTENSORFLOW_MONOLITHIC_BUILD",
                "/DPLATFORM_WINDOWS",
                "/DEIGEN_HAS_C99_MATH",
                "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
                "/DEIGEN_AVOID_STL_ARRAY",
                "/Iexternal/gemmlowp",
                "/wd4018",
                "/wd4577",
                "/DNOGDI",
                "/UTF_COMPILE_LIBRARY",
                "/DNDEBUG",
            ],
            "@bazel_tools//src/conditions:darwin": [
                "-std=c++11",
                "-DNDEBUG",
            ],
            "//conditions:default": [
                "-std=c++11",
                "-DNDEBUG",
                "-pthread",
            ],
        })
    )

def tf_io_features():
    return (
        select({
            "@bazel_tools//src/conditions:windows": [
                "windows_export_all_symbols",
            ],
            "//conditions:default": [],
        })
    )

# The following is a patch to bazel's rules_swift to allow windows/linux skip
def _create_xcode_toolchain(repository_ctx):
    """Creates BUILD targets for the Swift toolchain on macOS using Xcode.

    Args:
      repository_ctx: The repository rule context.
    """
    path_to_swiftc = repository_ctx.which("swiftc")

    repository_ctx.file(
        "BUILD",
        """
load(
    "@build_bazel_rules_swift//swift/internal:xcode_swift_toolchain.bzl",
    "xcode_swift_toolchain",
)

package(default_visibility = ["//visibility:public"])

xcode_swift_toolchain(
    name = "toolchain",
)
""",
    )

def _swift_autoconfiguration_impl(repository_ctx):
    os_name = repository_ctx.os.name.lower()
    if os_name.startswith("mac os"):
        _create_xcode_toolchain(repository_ctx)

swift_autoconfiguration = repository_rule(
    environ = ["CC", "PATH"],
    implementation = _swift_autoconfiguration_impl,
)

def tf_io_swift():
    http_archive(
        name = "build_bazel_rules_swift",
        sha256 = "da799f591aed933f63575ef0fbf7b7a20a84363633f031fcd48c936cee771502",
        strip_prefix = "rules_swift-1b0fd91696928ce940bcc220f36c898694f10115",
        urls = [
            "https://github.com/bazelbuild/rules_swift/archive/1b0fd91696928ce940bcc220f36c898694f10115.tar.gz",
        ],
    )
    http_archive(
        name = "com_github_nlohmann_json",
        urls = [
            "https://github.com/nlohmann/json/releases/download/v3.6.1/include.zip",
        ],
        sha256 = "69cc88207ce91347ea530b227ff0776db82dcb8de6704e1a3d74f4841bc651cf",
        type = "zip",
        build_file = "@build_bazel_rules_swift//third_party:com_github_nlohmann_json/BUILD.overlay",
    )
    swift_autoconfiguration = repository_rule(
        environ = ["CC", "PATH"],
        implementation = _swift_autoconfiguration_impl,
    )
    swift_autoconfiguration(
        name = "build_bazel_rules_swift_local_config",
    )
