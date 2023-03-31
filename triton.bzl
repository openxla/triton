def if_not_msvc(a):
  return select({
    ":compiler_is_msvc": [],
    "//conditions:default": a,
  })
