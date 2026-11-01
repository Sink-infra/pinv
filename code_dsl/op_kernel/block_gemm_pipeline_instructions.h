namespace tl::ascend {
__aicore__ inline void br_load_b_batch(AscendC::LocalTensor<float>d,AscendC::GlobalTensor<float>s,AscendC::LocalTensor<float>,int rows,int cols,int stride){
 AscendC::Nd2NzParams p;p.ndNum=rows/16;p.nValue=16;p.dValue=cols;p.srcNdMatrixStride=16*stride;p.srcDValue=stride;p.dstNzC0Stride=16;p.dstNzNStride=1;p.dstNzMatrixStride=16*cols;AscendC::DataCopy(d,s,p);
}
}

namespace tl::ascend {
__aicore__ inline void bg_copy_a(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>){copy_l1_to_l0a<float,128,64>(d,a,128,64);}
__aicore__ inline void bg_copy_b(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>){copy_l1_to_l0b<float,64,128>(d,a,64,128);}
__aicore__ inline void bg_mma(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>b,int clear){mma<float,float,128,128>(a,b,d,clear!=0,64);}
}
