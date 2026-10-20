#pragma once
__aicore__ inline float PinvCubePeakScale(__gm__ float *base,LocalTensor<float> cache,
    LocalTensor<float> scratch[],uint32_t count,uint32_t core,uint32_t cores,float fallback){
    Abs(scratch[0],cache,count);PipeBarrier<PIPE_V>();ReduceMax(scratch[2],scratch[0],scratch[1],count,false);
    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
    auto statistic=scratch[7];statistic.SetValue(0,scratch[2].GetValue(0));
    GlobalTensor<float> statistics;statistics.SetGlobalBuffer(base);
    SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyPad(statistics[36+core*8],statistic,DataCopyExtParams{1,4,0,0,0});SyncAll();
    uint32_t chunks=(cores+7)/8;
    Duplicate(statistic,0.0f,chunks*64);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);DataCopy(statistic,statistics[32],cores*8);
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    uint64_t selected[2]={uint64_t(0x1010101010101010),0};
    WholeReduceMax(scratch[0],statistic,selected,static_cast<uint8_t>(chunks),1,1,8);PipeBarrier<PIPE_V>();
    uint64_t summary[2]={uint64_t(0x5555)&((uint64_t(1)<<(chunks*2))-1),0};
    WholeReduceMax(scratch[2],scratch[0],summary,1,1,1,8);
    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
    float peak,bits;GetReduceRepeatMaxMinSpr<float>(peak,bits);
    if(!(peak>0 && peak<1.0e20f))return fallback;
    float scale=1.0f,target=4.0f*peak;
    while(scale<target)scale*=2;while(scale*.5f>=target)scale*=.5f;
    return scale;
}
template<bool RIGHT>
__aicore__ inline float PinvCubeQuantize(LocalTensor<float> values,LocalTensor<float> scratch[],
    GlobalTensor<int8_t> digits,uint32_t n,uint32_t owned,uint32_t core,uint32_t cores,float scale){
    uint32_t count=owned*n;
    auto integers=scratch[0].ReinterpretCast<int32_t>();auto temp=scratch[0],rounded=scratch[1],carry=scratch[2];
    LocalTensor<half> planes[pinv::kCubePlanes];
    for(uint32_t limb=0;limb<pinv::kCubePlanes;++limb)planes[limb]=scratch[4+limb/2].ReinterpretCast<half>()[(limb%2)*count];
    Muls(values,values,pinv::kCubeRadix/scale,count);PipeBarrier<PIPE_V>();
    for(uint32_t limb=0;limb<pinv::kCubePlanes;++limb){
        Cast(integers,values,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
        Cast(rounded,integers,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
        Sub(values,values,rounded,count);PipeBarrier<PIPE_V>();Muls(values,values,pinv::kCubeRadix,count);PipeBarrier<PIPE_V>();
        Cast(planes[limb],rounded,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
    }
    for(uint32_t limb=pinv::kCubePlanes-1;limb>0;--limb){
        Cast(rounded,planes[limb],RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        Adds(carry,rounded,-127.0f,count);PipeBarrier<PIPE_V>();Maxs(carry,carry,0.0f,count);PipeBarrier<PIPE_V>();
        Mins(carry,carry,1.0f,count);PipeBarrier<PIPE_V>();
        Muls(temp,carry,pinv::kCubeRadix,count);PipeBarrier<PIPE_V>();Sub(rounded,rounded,temp,count);PipeBarrier<PIPE_V>();
        Cast(planes[limb],rounded,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        Cast(rounded,planes[limb-1],RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        Add(rounded,rounded,carry,count);PipeBarrier<PIPE_V>();
        Cast(planes[limb-1],rounded,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
    }
    auto small=scratch[3].ReinterpretCast<int8_t>();
    for(uint32_t limb=0;limb<pinv::kCubePlanes;++limb){
        Cast(small,planes[limb],RoundMode::CAST_RINT,count);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        uint32_t destination=RIGHT?core*pinv::kCubePlanes*n+limb*n:limb*n*n+core*n;
        uint32_t step=RIGHT?(cores*pinv::kCubePlanes-1)*n:(cores-1)*n;
        DataCopyPad(digits[destination],small,DataCopyExtParams{static_cast<uint16_t>(owned),n,0,step,0});
        SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
    }
    Mul(scratch[0],values,values,count);PipeBarrier<PIPE_V>();
    ReduceSum(scratch[2],scratch[0],scratch[1],count);
    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
    return scratch[2].GetValue(0);
}
