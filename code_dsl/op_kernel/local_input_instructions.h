namespace tl::ascend {
__aicore__ inline void li_abs_packed(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,int width,int rows,int pitch){AscendC::Abs(d,a,uint64_t(width),uint8_t(rows),AscendC::UnaryRepeatParams{1,1,uint8_t(width/8),uint8_t(pitch/8)});}
__aicore__ inline void li_scale_rows(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,float scale,int width,int rows,int pitch){AscendC::Muls(d,a,scale,uint64_t(width),uint8_t(rows),AscendC::UnaryRepeatParams{1,1,uint8_t(pitch/8),uint8_t(pitch/8)});}
}
