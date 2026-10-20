"""Runtime memory and work balance conditions for packet preparation."""
def use_prepared(n,stride,batch,blocks,window):
    import tvm
    import tilelang.language as T
    group=blocks*2//batch
    workers=T.max(group-2,1)
    owned=T.ceildiv(n,workers)
    factor_rows=T.ceildiv(owned,8)*8
    k=T.ceildiv(n,8)*8
    wide=T.ceildiv(k,64)*64
    output=T.max(T.max(owned*k,512),16*factor_rows*8+owned*16+256)
    worker=owned*stride+16*(k+16)+output+8*T.max(factor_rows,16)+64+5*wide+32+factor_rows+16+k
    producer=n*window+4096+10*k+3*wide+64+T.ceildiv(n,16)*64+512
    # Extra packet preparation serves groups whose row updates still dominate
    # the producer. Larger groups retain the original window path.
    legacy_owned=T.ceildiv(n,T.max(group-1,1))
    return tvm.tir.all(group>2,n%16==0,window>0,window%16==0,worker<=49088,producer<=49088,legacy_owned*4>window)
