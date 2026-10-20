// Three-component arithmetic is confined to posterior residual measurements.
// Error-free FP32 products and TwoSum leave a third-order accumulation remainder.
    __aicore__ inline void CertTwoSum(LocalTensor<float> sum,LocalTensor<float> error,
        LocalTensor<float> a,LocalTensor<float> b,uint32_t count){
        auto v=calc[6],w=calc[7];
        Add(sum,a,b,count);PipeBarrier<PIPE_V>();Sub(v,sum,a,count);PipeBarrier<PIPE_V>();
        Sub(w,sum,v,count);PipeBarrier<PIPE_V>();Sub(w,a,w,count);Sub(v,b,v,count);PipeBarrier<PIPE_V>();
        Add(error,w,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CertAdd(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,
        LocalTensor<float> value,uint32_t count){
        CertTwoSum(calc[0],calc[1],a,value,count);
        CertTwoSum(calc[2],calc[3],b,calc[1],count);
        CertTwoSum(calc[4],calc[5],c,calc[3],count);
        CertTwoSum(a,calc[2],calc[0],calc[2],count);
        CertTwoSum(b,c,calc[2],calc[4],count);
    }
    __aicore__ inline void CertProduct(LocalTensor<float> p,LocalTensor<float> error,
        LocalTensor<float> a,LocalTensor<float> b,uint32_t count){
        auto hi=calc[0],lo=calc[1],ch=calc[2],cl=calc[3],v=calc[4];
        Muls(hi,a,pinv::kSplitFactor,count);Muls(ch,b,pinv::kSplitFactor,count);PipeBarrier<PIPE_V>();
        Sub(lo,hi,a,count);Sub(cl,ch,b,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);Sub(ch,ch,cl,count);PipeBarrier<PIPE_V>();
        Sub(lo,a,hi,count);Sub(cl,b,ch,count);Mul(p,a,b,count);PipeBarrier<PIPE_V>();
        Mul(error,hi,ch,count);PipeBarrier<PIPE_V>();Sub(error,error,p,count);PipeBarrier<PIPE_V>();
        Mul(v,hi,cl,count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,ch,count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,cl,count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CertProductScalar(LocalTensor<float> p,LocalTensor<float> error,
        LocalTensor<float> a,float b,uint32_t count){
        auto hi=calc[0],lo=calc[1],v=calc[2];
        volatile float split=b*pinv::kSplitFactor;
        volatile float temp=split-b;
        volatile float ch=split-temp;
        volatile float cl=b-ch;
        Muls(hi,a,pinv::kSplitFactor,count);PipeBarrier<PIPE_V>();Sub(lo,hi,a,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);PipeBarrier<PIPE_V>();Sub(lo,a,hi,count);Muls(p,a,b,count);PipeBarrier<PIPE_V>();
        Muls(error,hi,static_cast<float>(ch),count);PipeBarrier<PIPE_V>();Sub(error,error,p,count);PipeBarrier<PIPE_V>();
        Muls(v,hi,static_cast<float>(cl),count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,static_cast<float>(ch),count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,static_cast<float>(cl),count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline Wide CertScalarSum(float a,float b){
        volatile float sum=a+b;
        volatile float v=sum-a;
        volatile float w=sum-v;
        volatile float ar=a-w;
        volatile float br=b-v;
        return {sum,ar+br};
    }
    __aicore__ inline void CertScalarAdd(float &a,float &b,float &c,float value){
        Wide first=CertScalarSum(a,value),second=CertScalarSum(b,first.l),third=CertScalarSum(c,second.l);
        first=CertScalarSum(first.h,second.h);
        second=CertScalarSum(first.l,third.h);
        a=first.h;b=second.h;c=second.l;
    }
    __aicore__ inline void CertScale(LocalTensor<float> a,Wide factor,uint32_t count){
        Duplicate(certH,0.0f,count);Duplicate(certL,0.0f,count);Duplicate(certT,0.0f,count);PipeBarrier<PIPE_V>();
        CertProductScalar(bh,bl,a,factor.h,count);
        CertAdd(certH,certL,certT,bh,count);CertAdd(certH,certL,certT,bl,count);
        CertProductScalar(bh,bl,a,factor.l,count);
        CertAdd(certH,certL,certT,bh,count);CertAdd(certH,certL,certT,bl,count);
    }
    __aicore__ inline Wide CertDot(LocalTensor<float> a,LocalTensor<float> b,
        LocalTensor<float> c,LocalTensor<float> d,uint32_t count){
        Duplicate(certH,0.0f,count);Duplicate(certL,0.0f,count);Duplicate(certT,0.0f,count);PipeBarrier<PIPE_V>();
        LocalTensor<float> left[2]={a,b},right[2]={c,d};
        for(uint32_t i=0;i<2;++i)for(uint32_t j=0;j<2;++j){
            CertProduct(bh,bl,left[i],right[j],count);
            CertAdd(certH,certL,certT,bh,count);CertAdd(certH,certL,certT,bl,count);
        }
        for(uint32_t size=count;size>8;size/=2){
            uint32_t half=size/2;
            CertAdd(certH,certL,certT,certH[half],half);
            CertAdd(certH,certL,certT,certL[half],half);
            CertAdd(certH,certL,certT,certT[half],half);
        }
        VS();float sh=0,sl=0,st=0;
        for(uint32_t i=0;i<8;++i){
            CertScalarAdd(sh,sl,st,certH.GetValue(i));
            CertScalarAdd(sh,sl,st,certL.GetValue(i));
            CertScalarAdd(sh,sl,st,certT.GetValue(i));
        }
        return Plus({sh,sl},{st,0});
    }
