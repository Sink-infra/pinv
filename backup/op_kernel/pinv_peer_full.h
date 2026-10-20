#pragma once
#include "pinv_execution.h"
#if defined(TILING_KEY_VAR) && TILING_KEY_VAR==(PINV_KEY_GROUP | PINV_KEY_PIPELINE | PINV_KEY_CUBE)
class PeerWindowProducer {
    TBuf<TPosition::VECCALC> mem;
    LocalTensor<float> metaData,a,scratch,factors,v,prod,tmp,red;
    LocalTensor<uint32_t> indices,perm;
    GlobalTensor<float> input,stage,header;
    uint32_t n,k,wide,window,role,pitch,ownWidth,frontHalf,trailHalf;
    bool headerIssued=false,pendingReady=false;
    __aicore__ inline void VS(){SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);}
    __aicore__ inline void SV(){SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);}
    __aicore__ inline void Drain(){
        if(headerIssued){if(pendingReady)CrossCoreWaitFlag(7);CrossCoreWaitFlag(9);headerIssued=false;}
    }
    __aicore__ inline void Exchange(uint32_t first,uint32_t size,uint32_t half){
        Drain();uint32_t owner=first<half?0:1;
        if(role==owner){
            SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyPad(header,a,DataCopyExtParams{static_cast<uint16_t>(size),64,(pitch-16)/8,0,0});
            CrossCoreSetFlag<1,PIPE_MTE3>(7);
            SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
            pendingReady=true;
        }else{
            CrossCoreSetFlag<1,PIPE_MTE3>(7);CrossCoreWaitFlag(7);
            SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
            DataCopyPad(a,header,DataCopyExtParams{static_cast<uint16_t>(size),64,0,(pitch-16)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            pendingReady=false;
        }
        CrossCoreSetFlag<1,PIPE_MTE3>(9);headerIssued=true;
    }
public:
    __aicore__ inline LocalTensor<float> Flags(){return scratch;}
    __aicore__ inline void Init(GM_ADDR in,GM_ADDR workspace,const LinalgPinvTilingData &data,TPipe &pipe,uint32_t batch,uint32_t c){
        n=data.cols;k=(n+7)/8*8;wide=(k+63)/64*64;window=data.local;role=c;
        uint32_t trailing=n-window;frontHalf=(window+31)/32*16;trailHalf=(trailing+31)/32*16;
        ownWidth=role?(window-frontHalf+7)/8*8:frontHalf;pitch=ownWidth+(role?16:0);if(pitch<16)pitch=16;
        uint32_t laterWidth=role?trailing-trailHalf:trailHalf;
        uint32_t capacity=n*pitch>trailing*(16+laterWidth)?n*pitch:trailing*(16+laterWidth);
        uint32_t count=capacity+4096+10*k+2*wide+64+(n+15)/16*64;
        pipe.InitBuffer(mem,count*4);auto all=mem.Get<float>();uint32_t off=0;
        a=all;off+=capacity;scratch=all[off];off+=4096;factors=all[off];off+=8*k;
        v=all[off];off+=wide;prod=all[off];off+=wide;tmp=all[off];off+=32;red=all[off];off+=32;
        indices=all[off].ReinterpretCast<uint32_t>();off+=k;perm=all[off].ReinterpretCast<uint32_t>();off+=k;metaData=all[off];
        input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(in));
        stage.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace)+uint64_t(batch)*n*data.stride);
        header.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace)+pinv::MetadataOffset(data.batch,n,data.stride,batch)+3072);
    }
    __aicore__ inline bool Run(uint32_t batch,GlobalTensor<float> metadata){

        uint32_t offset=batch*n*n,actualWidth=role?window-frontHalf:frontHalf;
        Duplicate(a,0.0f,n*pitch);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(a[role?16:0],input[offset+(role?frontHalf:0)],DataCopyExtParams{static_cast<uint16_t>(n),actualWidth*4,(n-actualWidth)*4,role?2U:0U,0},DataCopyPadExtParams<float>{true,0,static_cast<uint8_t>(ownWidth-actualWidth),0});
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        float localScale=0,localTotal=0,localPower=1;constexpr uint32_t chunkSize=4096;
        for(uint32_t begin=0;begin<n*pitch;begin+=chunkSize){
            uint32_t count=n*pitch-begin<chunkSize?n*pitch-begin:chunkSize;
            Abs(scratch,a[begin],count);PipeBarrier<PIPE_V>();ReduceMax(red,scratch,prod,count,false);VS();
            float value=red.GetValue(0);localScale=value>localScale?value:localScale;
        }
        if(localScale>0 && localScale<pinv::kScaleLimit){while(localPower<localScale)localPower*=2;while(localPower*.5f>=localScale)localPower*=.5f;}
        for(uint32_t begin=0;begin<n*pitch;begin+=16320){uint32_t count=n*pitch-begin<16320?n*pitch-begin:16320;Muls(a[begin],a[begin],1.0f/localPower,count);}
        PipeBarrier<PIPE_V>();
        for(uint32_t begin=0;begin<n*pitch;begin+=chunkSize){
            uint32_t count=n*pitch-begin<chunkSize?n*pitch-begin:chunkSize;
            Mul(scratch,a[begin],a[begin],count);PipeBarrier<PIPE_V>();ReduceSum(red,scratch,prod,count);VS();localTotal+=red.GetValue(0);
        }
        scratch.SetValue(0,localScale);scratch.SetValue(1,localTotal);scratch.SetValue(2,localPower);
        SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);DataCopy(header[role*8],scratch,8);
        CrossCoreSetFlag<1,PIPE_MTE3>(4);CrossCoreWaitFlag(4);
        SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
        DataCopy(scratch,header,16);SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);
        float scale0=scratch.GetValue(0),scale1=scratch.GetValue(8),scale=scale0>scale1?scale0:scale1;
        float sum0=scratch.GetValue(1),sum1=scratch.GetValue(9),power0=scratch.GetValue(2),power1=scratch.GetValue(10);
        CrossCoreSetFlag<1,PIPE_MTE3>(5);CrossCoreWaitFlag(5);
        bool valid=scale>=.1f && scale<pinv::kScaleLimit;float power=1;
        if(valid){while(power<scale)power*=2;while(power*.5f>=scale)power*=.5f;}
        float ratio0=power0/power,ratio1=power1/power,total=(sum0*ratio0*ratio0+sum1*ratio1*ratio1)*1.00001f;
        for(uint32_t begin=0;begin<n*pitch;begin+=16320){uint32_t count=n*pitch-begin<16320?n*pitch-begin:16320;Muls(a[begin],a[begin],localPower/power,count);}
        PipeBarrier<PIPE_V>();for(uint32_t row=0;row<k;++row)perm.SetValue(row,row);SV();
        uint32_t size=n;
        for(uint32_t first=0;first<n;first+=16){
            if(first==window){
                PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<0,PIPE_MTE3>(3);CrossCoreWaitFlag(3);
                CrossCoreSetFlag<0,PIPE_MTE3>(2);CrossCoreWaitFlag(2);
                ownWidth=role?n-window-trailHalf:trailHalf;pitch=16+ownWidth;
                if(valid){
                    for(uint32_t row=0;row<size;++row){uint32_t original=perm.GetValue(row);
                        DataCopyPad(a[row*pitch+16],stage[original*k+window+(role?trailHalf:0)],DataCopyExtParams{1,ownWidth*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                    }
                    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
                }
                PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<0,PIPE_MTE3>(4);CrossCoreWaitFlag(4);
            }
            if(valid){
                uint32_t localFirst=first<window?first:first-window,half=first<window?frontHalf:trailHalf;
                Exchange(localFirst,size,half);
                auto offsets=indices.ReinterpretCast<int32_t>();CreateVecIndex(offsets,int32_t(0),size);PipeBarrier<PIPE_V>();Muls(offsets,offsets,static_cast<int32_t>(pitch*4),size);PipeBarrier<PIPE_V>();
                uint32_t block=size<16?size:16;
                for(uint32_t p=0;p<block;++p){
                    uint32_t remaining=size-p;Gather(v,a,indices,(p*pitch+p)*4,remaining);PipeBarrier<PIPE_V>();Abs(prod,v,remaining);PipeBarrier<PIPE_V>();
                    uint32_t full=remaining/64,tail=remaining%64,chunks=full+(tail!=0);
                    if(full)WholeReduceMax(red,prod,uint64_t(64),static_cast<uint8_t>(full),1,1,8);
                    if(tail)WholeReduceMax(red[full*2],prod[full*64],static_cast<uint64_t>(tail),1,1,1,8);
                    uint64_t summary[2]={uint64_t(0x5555)&((uint64_t(1)<<(chunks*2))-1),0};PipeBarrier<PIPE_V>();WholeReduceMax(tmp,red,summary,1,1,1,8);VS();
                    float best,bits;GetReduceRepeatMaxMinSpr<float>(best,bits);uint32_t chunk=(*reinterpret_cast<uint32_t *>(&bits))/2;
                    uint32_t selected=p+chunk*64+red.ReinterpretCast<uint32_t>().GetValue(chunk*2+1);
                    if(!(best>1.0e-6f)){valid=false;break;}
                    float divisor=v.GetValue(selected-p);
                    if(selected!=p){uint32_t old=perm.GetValue(p);perm.SetValue(p,perm.GetValue(selected));perm.SetValue(selected,old);}
                    metaData.SetValue(first/16*64+p,static_cast<float>(static_cast<int32_t>(perm.GetValue(p))));metaData.SetValue(first/16*64+16+p,1.0f/divisor);
                    if(remaining<=1)continue;
                    v.SetValue(selected-p,0.0f);DataCopy(scratch,a[selected*pitch],pitch);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-(1.0f/divisor),pitch);PipeBarrier<PIPE_V>();
                    SV();Brcb(factors,v,(remaining+7)/8,{1,8});PipeBarrier<PIPE_V>();
                    for(uint32_t col=(p/8)*8;col<pitch;col+=64){uint32_t count=pitch-col<64?pitch-col:64;
                        for(uint32_t row=0;row<remaining;row+=255){uint32_t repeats=remaining-row<255?remaining-row:255;
                            MulAddDst(a[(p+row)*pitch+col],scratch[col],factors[row*8],static_cast<uint64_t>(count),static_cast<uint8_t>(repeats),{1,1,0,static_cast<uint8_t>(pitch/8),0,1});
                        }
                    }
                    PipeBarrier<PIPE_V>();if(selected!=p){DataCopy(a[selected*pitch],a[p*pitch],pitch);PipeBarrier<PIPE_V>();}
                }
                uint32_t remaining=size-block,owner=localFirst<half?0:1,drop=role==owner?block:0,newWidth=ownWidth-drop;
                uint32_t oldPrefix=role==owner?0:16,nextOwner=localFirst+block<half?0:1,newPrefix=role==nextOwner?0:16,newPitch=newPrefix+newWidth;if(newPitch<16)newPitch=16;
                VS();
                if(remaining && newWidth && first+block!=window){
                    for(uint32_t row=0;row<remaining;row+=16){uint32_t count=remaining-row<16?remaining-row:16;
                        DataCopy(a[row*newPitch+newPrefix],a[(row+block)*pitch+oldPrefix+drop],DataCopyParams{static_cast<uint16_t>(count),static_cast<uint16_t>(newWidth/8),static_cast<uint16_t>((pitch-newWidth)/8),static_cast<uint16_t>((newPitch-newWidth)/8)});PipeBarrier<PIPE_V>();
                    }
                }
                if(remaining){auto order=perm.ReinterpretCast<int32_t>();auto temporary=scratch.ReinterpretCast<int32_t>();uint32_t count=(remaining+7)/8*8;SV();DataCopy(temporary,order[block],count);PipeBarrier<PIPE_V>();DataCopy(order,temporary,count);VS();}
                SV();size=remaining;ownWidth=newWidth;pitch=newPitch;
            }
            if(first){CrossCoreWaitFlag(0);CrossCoreWaitFlag(1);}
            auto record=metaData[first/16*64];record.SetValue(32,power);record.SetValue(33,total);record.SetValue(34,valid?1.0f:0.0f);
            if(role==1){SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);DataCopy(metadata[first/16*64],record,64);}
            CrossCoreSetFlag<0,PIPE_MTE3>(0);CrossCoreSetFlag<0,PIPE_MTE3>(1);
        }
        Drain();CrossCoreWaitFlag(0);CrossCoreWaitFlag(1);return valid;
    }
};
__aicore__ inline bool TryPeerFullPinv(GM_ADDR in,GM_ADDR dst,GM_ADDR workspace,const LinalgPinvTilingData &data){
    uint32_t all=PinvVectorBlockCount(),physical=GetBlockIdx(),groupSize=all/data.batch,workers=groupSize-2,batch=physical/groupSize,core=physical%groupSize;
    TPipe pipe;LocalTensor<float> local;bool valid;float inverse=0;
    GlobalTensor<float> metadata;metadata.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace)+pinv::MetadataOffset(data.batch,data.cols,data.stride,batch));
    if(core>=workers){auto producerData=data;producerData.local=data.cols;PeerWindowProducer op;op.Init(in,workspace,producerData,pipe,batch,core-workers);valid=op.Run(batch,metadata);local=op.Flags();}
    else{PipelineRows op;op.Init(in,dst,workspace,data,pipe,batch,core,workers);valid=op.Run(batch,metadata);inverse=op.inverseTotal;local=op.Flags();}
    PipeBarrier<PIPE_ALL>();local.SetValue(0,valid?1.0f:0.0f);local.SetValue(1,inverse);
    SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);DataCopyPad(metadata[pinv::kGroupStatus+core*8],local,DataCopyExtParams{1,8,0,0,0});SyncAll();
    valid=true;
    for(uint32_t b=0;b<data.batch;++b){
        metadata.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace)+pinv::MetadataOffset(data.batch,data.cols,data.stride,b));
        DataCopy(local,metadata[pinv::kGroupStatus],groupSize*8);SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);
        float inverseTotal=0;
        for(uint32_t c=0;c<groupSize;++c){if(local.GetValue(c*8)<0.5f)valid=false;inverseTotal+=local.GetValue(c*8+1);}
        DataCopy(local,metadata,64);SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);
        float power=local.GetValue(32),total=local.GetValue(33);
        local.SetValue(256,total);SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);Sqrt(local[256],local[256],1);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        float cutoff=local.GetValue(256)*data.rtol,absolute=data.atol/power;cutoff=cutoff>absolute?cutoff:absolute;
        if(!(inverseTotal>0 && inverseTotal*cutoff*cutoff<pinv::kGapGateSquared && total*inverseTotal<pinv::kDirectConditionLimit))valid=false;
    }
    PipeBarrier<PIPE_ALL>();pipe.Destroy();return valid;
}


#endif
