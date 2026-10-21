"""Compact equal-round Cube grids without exceeding recovery tile capacity."""
from types import SimpleNamespace
from code_dsl.op_host.recovery_layout import scratch, supported

def compatible(n,stride,batch,blocks):
    if not (blocks>0 and batch>0 and n//64>=4 and n<=256 and n%128==0 and batch<=16):
        return False
    cores=2*blocks
    group=cores//batch
    return (cores%batch==0 and group>1 and batch*n//64<=cores and batch*n//2<=1280
            and 13*batch*n*n<=scratch(n,batch,stride,blocks)['words']
            and (n+group-2)//(group-1)>=16)

def select_blocks(n,stride,batch,window,blocks):
    if not compatible(n,stride,batch,blocks):return blocks
    tiles=batch*(n//64)*(n//128)
    rounds=(tiles+blocks-1)//blocks
    balance=batch if batch%2 else batch//2
    target=(tiles+rounds-1)//rounds
    target=(target+balance-1)//balance*balance
    if not (0<target<blocks and compatible(n,stride,batch,target)):return blocks
    plan=SimpleNamespace(geometry=SimpleNamespace(m=n,n=n,batch=batch),
                         fields={'local':window,'stride':stride},blocks=target)
    return target if supported(plan) else blocks

CPP=r"""
#if !defined(__NPU_ARCH__)
namespace pinv_dsl {
inline bool BlockGridCompatible(uint32_t n,uint32_t stride,uint32_t batch,uint32_t blocks) {
    if(!blocks || !batch || n/64<4 || n>256 || n%128 || batch>16)return false;
    uint64_t cores=2*uint64_t(blocks),group=cores/batch;
    return !(cores%batch) && group>1 && uint64_t(batch)*n/64<=cores &&
           uint64_t(batch)*n/2<=1280 &&
           13*uint64_t(batch)*n*n<=WindowWorkFloats(batch,n,stride,blocks) &&
           (n+group-2)/(group-1)>=16;
}
inline uint32_t BlockLaunch(uint32_t n,uint32_t stride,uint32_t batch,
                           uint32_t window,uint32_t blocks) {
    if(!BlockGridCompatible(n,stride,batch,blocks))return blocks;
    uint64_t tiles=uint64_t(batch)*(n/64)*(n/128);
    uint64_t rounds=(tiles+blocks-1)/blocks,balance=batch%2?batch:batch/2;
    uint64_t target=(tiles+rounds-1)/rounds;
    target=(target+balance-1)/balance*balance;
    if(!target || target>=blocks || !BlockGridCompatible(n,stride,batch,target) ||
       !WindowSupported(n,n,batch,stride,window,target))return blocks;
    return uint32_t(target);
}
}
#endif
"""
