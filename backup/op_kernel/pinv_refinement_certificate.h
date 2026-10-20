#pragma once
    __aicore__ inline void CubeRefineRowNorms(LocalTensor<float> source,LocalTensor<float> dest,uint32_t owned){
        auto square=batchCalc[0],parts=batchCalc[1];
        Mul(square,source,source,owned*r8);Duplicate(parts,0.0f,8*(r8/64));PipeBarrier<PIPE_V>();
        for(uint32_t chunk=0;chunk<r8/64;++chunk)WholeReduceSum(parts[chunk*8],square[chunk*64],64,owned,1,1,r8/8);
        PipeBarrier<PIPE_V>();DataCopy(dest,parts,8);PipeBarrier<PIPE_V>();
        for(uint32_t chunk=1;chunk<r8/64;++chunk){Add(dest,dest,parts[chunk*8],8);PipeBarrier<PIPE_V>();}
        Sqrt(dest,dest,8);PipeBarrier<PIPE_V>();Muls(dest,dest,pinv::kNormSlack,8);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool CertifyCubeRefinement(LocalTensor<float> cache,uint32_t owned,float gamma,float rf,float er,float denominator,float power,float &correctedTotal){
        uint32_t count=owned*r8;
        Add(tileHigh,cache,tileLow,count);PipeBarrier<PIPE_V>();
        auto cn=batchCalc[2],yn=batchCalc[3];CubeRefineRowNorms(tileLow,cn,owned);CubeRefineRowNorms(cache,yn,owned);
        auto ct=cn[8],part=cn[16],scaleA=cn[24],scaleB=cn[32];
        Muls(scaleA,yn,gamma,8);PipeBarrier<PIPE_V>();Muls(part,scaleA,rf,8);PipeBarrier<PIPE_V>();
        Add(part,cn,part,8);Muls(ct,yn,er,8);PipeBarrier<PIPE_V>();Add(ct,part,ct,8);PipeBarrier<PIPE_V>();
        Muls(ct,ct,1.0f/denominator,8);PipeBarrier<PIPE_V>();Add(scaleA,scaleA,ct,8);Add(scaleB,yn,ct,8);PipeBarrier<PIPE_V>();
        auto fa=batchCalc[4],fb=batchCalc[5];Brcb(fa,scaleA,1,{1,8});Brcb(fb,scaleB,1,{1,8});PipeBarrier<PIPE_V>();
        auto bound=batchCalc[0],threshold=batchCalc[1],roundoff=batchCalc[6];
        for(uint32_t column=0;column<r8;column+=64){
            Mul(bound[column],gather[column],fa,uint64_t(64),static_cast<uint8_t>(owned),{1,1,0,static_cast<uint8_t>(r8/8),0,1});
            Mul(threshold[column],output[column],fb,uint64_t(64),static_cast<uint8_t>(owned),{1,1,0,static_cast<uint8_t>(r8/8),0,1});
        }
        PipeBarrier<PIPE_V>();Add(bound,bound,threshold,count);Abs(threshold,cache,count);Abs(roundoff,tileLow,count);PipeBarrier<PIPE_V>();
        Add(threshold,threshold,roundoff,count);PipeBarrier<PIPE_V>();Muls(threshold,threshold,pinv::kUnitRoundoff,count);PipeBarrier<PIPE_V>();
        Add(bound,bound,threshold,count);PipeBarrier<PIPE_V>();Muls(bound,bound,pinv::kResidualSlack,count);Abs(threshold,tileHigh,count);PipeBarrier<PIPE_V>();
        Muls(threshold,threshold,pinv::kOutputRtol,count);PipeBarrier<PIPE_V>();Adds(threshold,threshold,pinv::kOutputAtol*power,count);PipeBarrier<PIPE_V>();
        Sub(bound,bound,threshold,count);PipeBarrier<PIPE_V>();ReduceMax(red,bound,tmp,count,false);VS();bool valid=red.GetValue(0)<=0.0f;
        Mul(bound,tileHigh,tileHigh,count);PipeBarrier<PIPE_V>();ReduceSum(red,bound,tmp,count);VS();correctedTotal=red.GetValue(0);
        return valid;
    }
