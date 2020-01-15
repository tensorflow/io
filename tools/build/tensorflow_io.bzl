def tf_io_copts():
    return (
        [
            "-std=c++11",
            "-DNDEBUG",
        ] +
        select({
            "@bazel_tools//src/conditions:darwin": [],
            "//conditions:default": ["-pthread"],
        })
    )
