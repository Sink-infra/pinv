namespace tl::ascend {
__aicore__ inline void pf_shift(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,int bias,int n){AscendC::Adds(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),int32_t(bias),n);}
__aicore__ inline void pf_indices(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>b,int width,int rows,int pitch){AscendC::Add(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),b.ReinterpretCast<int32_t>(),uint64_t(width),uint8_t(rows),AscendC::BinaryRepeatParams{1,0,1,uint8_t(pitch/8),1,0});}
}
