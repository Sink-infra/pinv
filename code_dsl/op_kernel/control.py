"""DSL admission output: either an UB slot or alloc_var's local scalar buffer."""

def admission_writer(scalar=False):
    import tilelang.language as T
    @T.macro
    def store(destination,value):
        destination[0]=value
    return store
