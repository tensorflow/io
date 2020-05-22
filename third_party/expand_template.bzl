"""
Helper rule used by libmemcached.BUILD to build libmemcached's
depdendency on headers/libmemcached-1.0/configure.h
"""

def _expand_template_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.out,
        substitutions = ctx.attr.substitutions,
    )

expand_template = rule(
    attrs = {
        "out": attr.output(mandatory = True),
        "substitutions": attr.string_dict(mandatory = True),
        "template": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
    },
    output_to_genfiles = True,
    implementation = _expand_template_impl,
)
