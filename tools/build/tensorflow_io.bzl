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
