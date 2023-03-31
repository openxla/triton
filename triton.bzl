"""Bazel macros used by the triton build."""

def if_msvc(if_true, if_false = []):
    return select({
        ":compiler_is_msvc": if_true,
        "//conditions:default": if_false,
    })

def if_not_msvc(a):
    return if_msvc([], a)
