"""Resource bounds and row-work crossover for runtime block inversion."""
def use_block(n,stride,batch,blocks):
 import tvm
 import tilelang.language as T
 from code_dsl.op_host.recovery_layout import scratch
 cores=blocks*2;group=cores//batch
 owned=T.ceildiv(n,T.max(group-1,1))
 words=scratch(n,batch,stride,blocks)['words']
 return tvm.tir.all(n//64>=4,n<=256,n%128==0,batch<=16,cores%batch==0,
                    batch*n//64<=cores,batch*n//2<=1280,
                    11*batch*n*n<=words,owned>=16)
