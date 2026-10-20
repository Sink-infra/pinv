"""CANN ABI planning for the runtime window recovery strategy."""

CPP = r"""
#if defined(__NPU_ARCH__)
#define PV_WINDOW_LAYOUT_FN __aicore__ inline
#else
#define PV_WINDOW_LAYOUT_FN inline
#endif
namespace pinv_dsl {
PV_WINDOW_LAYOUT_FN bool WindowSupported(uint32_t m,uint32_t n,uint32_t batch,
    uint32_t stride,uint32_t window,uint32_t blocks) {
    uint64_t cores=uint64_t(blocks)*2;
    if(m!=n || !n || n>256 || n%128 || !batch || !cores || cores%batch ||
       cores*n>16384 || cores*16>1024 || cores*8+32>512)return false;
    uint64_t group=cores/batch;
    if(group<2 || group>48 || !window || window>=n || window%16)return false;
    uint64_t k=(n+7)/8*8,wide=(k+63)/64*64;
    uint64_t trailing=(n-window)*uint64_t((n-window+7)/8*8);
    uint64_t capacity=uint64_t(n)*window;
    if(trailing>capacity)capacity=trailing;
    uint64_t producer=capacity+4096+10*k+2*wide+64+(n+15)/16*64;
    uint64_t owned=(n+group-2)/(group-1),factors=(owned+7)/8*8;
    uint64_t output=owned*k>512?owned*k:512;
    uint64_t worker=owned*stride+16*(k+16)+output+8*(factors>16?factors:16)
                    +64+5*wide+32+factors+16+k;
    return producer<=49088 && worker<=49088 && (n+cores-1)/cores*n<=2048;
}
PV_WINDOW_LAYOUT_FN uint64_t WindowWorkFloats(uint32_t batch,uint32_t n,
    uint32_t stride,uint32_t blocks) {
    uint64_t extent=uint64_t(n)*n,cores=uint64_t(blocks)*2;
    uint64_t offset=(2*uint64_t(batch)*n*stride+uint64_t(batch)*8192+127)/128*128;
    uint64_t words=offset+61*extent/2+cores*n+cores*16+uint64_t(batch)*64;
    return (words+127)/128*128;
}
PV_WINDOW_LAYOUT_FN uint64_t WindowStatusFloats(uint32_t batch) {
    return uint64_t(batch)*512+128;
}
}
#undef PV_WINDOW_LAYOUT_FN
"""
