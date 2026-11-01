#ifndef PINV_BLOCK_LOOKAHEAD_H
#define PINV_BLOCK_LOOKAHEAD_H
#include "kernel_operator.h"
namespace tl::ascend {
template<class D,class A,class B> __aicore__ inline void br_late(D,A,B){AscendC::CrossCoreSetFlag<2,PIPE_FIX>(12);}
}
#endif
