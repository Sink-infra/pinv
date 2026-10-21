"""Runtime geometry and scratch limits for the window recovery strategy."""


def supported(plan):
    g=plan.geometry
    f=plan.fields
    n=g.n
    cores=2*plan.blocks
    if not (g.m==n and 0<n<=256 and n%128==0 and cores>0 and
            cores%g.batch==0 and cores*n<=16384 and cores*16<=1024 and cores*8+32<=512):
        return False
    group=cores//g.batch
    window=f['local']
    if not (2<=group<=48 and 0<window<n and window%16==0):
        return False
    k=(n+7)//8*8
    wide=(k+63)//64*64
    producer=n*window
    producer+=4096+10*k+2*wide+64+(n+15)//16*64
    owned=(n+group-2)//(group-1)
    factor_rows=(owned+7)//8*8
    worker=(owned*f['stride']+16*(k+16)+max(owned*k,512)+8*max(factor_rows,16)
            +64+5*wide+32+factor_rows+16+k)
    return max(producer,worker)<=49088 and (n+cores-1)//cores*n<=2048


def scratch(n,batch,stride,blocks):
    extent=n*n
    offset=(2*batch*n*stride+batch*8192+127)//128*128
    # Five digit planes per side, 25 INT32 products, residual, two
    # FP16 operands, correction, distributed norms and private status.
    qa=0
    qy=5*extent//4
    product=5*extent//2
    residual=product+25*extent
    half_y=residual+extent
    half_r=half_y+extent//2
    correction=half_r+extent//2
    partials=correction+extent
    quant_norms=partials+2*blocks*n
    raw_status=quant_norms+2*blocks*16
    fallback_status=raw_status+batch*32
    words=(offset+fallback_status+batch*32+127)//128*128
    block=13*batch*extent
    if isinstance(words,int) and isinstance(block,int):
        words=max(words,block)
    else:
        import tvm
        words=tvm.tir.max(words,block)
    return {'words':words,'status_words':batch*512+128,'offset':offset,
            'qa':qa,'qy':qy,'product':product,'residual':residual,'half_y':half_y,
            'half_r':half_r,'correction':correction,'partials':partials,
            'quant_norms':quant_norms,'raw_status':raw_status,'fallback_status':fallback_status}
