// Host tiling only. The planning body below is mechanically extracted from
// baseline ec5ec7a op_host/linalg_pinv.cpp; no matrix data enters this bridge.
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../code/op_kernel/linalg_pinv_tiling.h"
#include <algorithm>
#include <cstring>
#include <limits>
struct PvHostPlan {
    alignas(64) unsigned char tiling[4096];
    uint64_t workspace,lib_workspace,ub_size;
    uint32_t tiling_bytes,blocks,schedule,flags[9],aic,aiv;
};
struct PvHostContext {
    PvHostPlan *out;
    template<class T>T* GetTilingData(){out->tiling_bytes=std::max<uint32_t>(out->tiling_bytes,sizeof(T));return reinterpret_cast<T*>(out->tiling);}
    void SetBlockDim(uint32_t n){out->blocks=n;}
    uint32_t GetBlockDim(){return out->blocks;}
    void SetScheduleMode(uint32_t n){out->schedule=n;}
    uint64_t *GetWorkspaceSizes(uint32_t){return &out->workspace;}
    void SetFlags(uint32_t c,uint32_t t,uint32_t w,uint32_t l,uint32_t g,uint32_t p,uint32_t v,uint32_t b,uint32_t r){uint32_t flags[]={c,t,w,l,g,p,v,b,r};std::memcpy(out->flags,flags,sizeof(flags));}
};
extern "C" uint64_t pv_host_plan_size(){return sizeof(PvHostPlan);}
extern "C" uint64_t pv_host_plan_alignment(){return alignof(PvHostPlan);}
extern "C" int pv_host_plan(int64_t m,int64_t n,uint64_t batch,float atol,float rtol,const char *soc,PvHostPlan *out){
    if(!out || m<1 || m>512 || n<1 || n>512 || batch<1 || batch>std::numeric_limits<uint32_t>::max())return -1;
    auto *sdk=platform_ascendc::PlatformAscendCManager::GetInstance(soc);if(!sdk)return -2;
    std::memset(out,0,sizeof(*out));out->aic=sdk->GetCoreNumAic();out->aiv=sdk->GetCoreNumAiv();
    sdk->GetCoreMemSize(platform_ascendc::CoreMemType::UB,out->ub_size);out->lib_workspace=sdk->GetLibApiWorkSpaceSize();
    PvHostContext state{out};auto *context=&state;
    const float *at=&atol,*rt=&rtol;
    auto t = context->GetTilingData<LinalgPinvTilingData>();
    t->m = m; t->n = n; t->batch = batch;
    t->rows = std::max(m, n); t->cols = std::min(m, n);
    t->parallel = batch <= 4 && t->cols >= 16;
    uint32_t r8 = (t->rows + 7) / 8 * 8, k8 = (t->cols + 7) / 8 * 8;
    t->stride = r8 + k8 + 8;
    auto platform = *platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (m != n) {
        if(uint64_t(pinv::RectLocalFloats(t->rows,t->cols))*sizeof(float)>ubSize)return -1;
        t->parallel=0;t->local=0;t->localInverse=0;t->inverseTile=0;
        t->resident=0;t->groupCount=0;t->singleResident=0;
        t->atol=at?*at:0.0f;
        t->rtol=rt && *rt>=0?*rt:(t->atol>0?0.0f:std::max(m,n)*1.1920928955078125e-7f);
        context->SetBlockDim(std::min<uint64_t>(batch,platform.GetCoreNumAiv()));
        auto ws=context->GetWorkspaceSizes(1);
        ws[0]=platform.GetLibApiWorkSpaceSize()+pinv::RectWorkspaceFloats(batch,t->rows,t->cols)*sizeof(float);
        uint32_t DT_A=0U,COMPACT=0,TAIL=0,WAVE=0,LOCAL=0,GROUP=0,PIPELINE=0,WINDOW=0,CUBE=0,RECT=1;
        context->SetFlags(COMPACT,TAIL,WAVE,LOCAL,GROUP,PIPELINE,WINDOW,CUBE,RECT);
        return 0;
    }
    uint64_t inverseBytes = (2 * uint64_t(t->cols) * t->stride + 4 * t->stride + 4 * r8 + 8 * t->rows + k8 + 32 + 8 * (r8 + k8)) * 4;
    if (t->rows <= 32) inverseBytes += 4 * k8 * k8 * sizeof(float);
    if (m == n && (t->stride <= 64 || (t->cols > 32 && t->cols <= 96))) inverseBytes += (k8 - t->cols) * t->stride * sizeof(float);
    t->inverseTile = 8;
    while (t->inverseTile > 1 && inverseBytes + 12 * t->inverseTile * t->stride * sizeof(float) > ubSize) t->inverseTile /= 2;
    inverseBytes += 12 * t->inverseTile * t->stride * sizeof(float);
    t->localInverse = m == n && inverseBytes <= ubSize;
    if (t->localInverse) t->parallel = 0;
    uint32_t parallelPadding = r8 % 64 ? 8 : 0;
    if (t->parallel) {
        uint32_t narrowStride = (t->stride + 63) / 64 * 64 + parallelPadding;
        t->stride = (t->stride + 127) / 128 * 128 + parallelPadding;
        if (m == n && narrowStride != t->stride) {
            uint32_t cores = std::min<uint32_t>(platform.GetCoreNumAic(), (t->cols + 1) / 2);
            uint32_t owned = (t->cols + cores - 1) / cores;
            auto capacity = [&](uint32_t candidate) {
                uint64_t base = (4 * uint64_t(candidate) + 4 * r8 + 8 * t->rows + k8 + 32 + 8 * (r8 + k8)) * sizeof(float);
                uint64_t resident = 2 * uint64_t(owned) * candidate * sizeof(float);
                return ubSize > base + resident ? std::min<uint64_t>(owned, (ubSize - base - resident) / (12 * candidate * sizeof(float))) : uint64_t(0);
            };
            uint32_t oldTile = capacity(t->stride), newTile = capacity(narrowStride);
            if (newTile && (!oldTile || (owned + newTile - 1) / newTile < (owned + oldTile - 1) / oldTile)) t->stride = narrowStride;
        }
    }
    t->resident = 0; t->groupCount = 0; t->singleResident = 0;
    if (t->parallel && m == n) {
        uint64_t baseBytes = (4 * uint64_t(t->stride) + 4 * r8 + 8 * t->rows + k8 + 32 + 8 * (r8 + k8)) * sizeof(float);
        uint32_t cores = std::min<uint32_t>(platform.GetCoreNumAic(), (t->cols + 1) / 2);
        uint32_t owned = (t->cols + cores - 1) / cores;
        uint64_t capacity = ubSize > baseBytes ? (ubSize - baseBytes) / (14 * t->stride * sizeof(float)) : 1;
        t->inverseTile = std::max<uint32_t>(1, std::min<uint64_t>(owned, capacity));
        uint64_t residentBytes = 2 * uint64_t(owned) * t->stride * sizeof(float);
        uint64_t residentCapacity = ubSize > baseBytes + residentBytes ? (ubSize - baseBytes - residentBytes) / (12 * t->stride * sizeof(float)) : 0;
        if (residentCapacity > 0) {
            t->resident = owned;
            t->inverseTile = std::min<uint64_t>(owned, residentCapacity);
        }
    }
    if (t->parallel && t->resident && batch > 1) {
        uint32_t cores = std::min<uint32_t>(platform.GetCoreNumAic(), (t->cols + 1) / 2);
        for (uint32_t groups = batch; groups > 1; --groups) {
            if (batch % groups != 0) continue;
            uint32_t groupCores = cores / groups;
            if (groupCores == 0) continue;
            uint32_t owned = (t->cols + groupCores - 1) / groupCores;
            uint64_t baseBytes = (4 * uint64_t(t->stride) + 4 * r8 + 8 * t->rows + k8 + 32 + 8 * (r8 + k8)) * sizeof(float);
            uint64_t residentBytes = 2 * uint64_t(owned) * t->stride * sizeof(float);
            uint64_t capacity = ubSize > baseBytes + residentBytes ? (ubSize - baseBytes - residentBytes) / (12 * t->stride * sizeof(float)) : 0;
            if (owned <= 64 && capacity > 0) {
                t->groupCount = groups; t->resident = owned;
                t->inverseTile = std::min<uint64_t>(t->inverseTile, capacity);
                break;
            }
        }
    }
    if (t->parallel && t->resident && batch > 1 && t->groupCount < batch) {
        uint32_t cores = std::min<uint32_t>(platform.GetCoreNumAic(), (t->cols + 1) / 2);
        uint32_t active = (t->cols + cores - 1) / cores;
        uint64_t baseBytes = (4 * uint64_t(t->stride) + 4 * r8 + 8 * t->rows + k8 + 32 + 8 * (r8 + k8)) * sizeof(float);
        for (uint32_t groups = batch; groups > std::max<uint32_t>(t->groupCount, 1); --groups) {
            if (batch % groups != 0) continue;
            uint32_t groupCores = cores / groups;
            if (groupCores == 0) continue;
            uint32_t owned = (t->cols + groupCores - 1) / groupCores;
            uint64_t residentBytes = uint64_t(owned) * t->stride * sizeof(float);
            uint64_t capacity = ubSize > baseBytes + residentBytes ? (ubSize - baseBytes - residentBytes) / (12 * t->stride * sizeof(float)) : 0;
            uint32_t tile = std::min<uint64_t>(t->inverseTile, capacity);
            uint32_t factorRows = (owned + 7) / 8 * 8;
            if (owned <= 64 && owned >= 2 * active && tile > 0 && tile * t->stride >= factorRows * 8) {
                t->groupCount = groups; t->resident = owned; t->inverseTile = tile; t->singleResident = 1;
                break;
            }
        }
    }
    uint32_t panelLaunch = 0;
    if (t->parallel && m == n && batch > 1 && t->groupCount > 0 && (t->groupCount < t->batch || t->singleResident)) {
        uint32_t vectorCores = std::min<uint32_t>(platform.GetCoreNumAiv(), (t->cols + 1) / 2);
        uint32_t groupCores = vectorCores / batch;
        uint32_t narrow = (r8 + k8 + 8 + 63) / 64 * 64 + parallelPadding;
        uint32_t candidates[2] = {t->stride, narrow};
        for (uint32_t c = 0; c < 2 && groupCores > 0; ++c) {
            uint64_t pitch = candidates[c];
            if (pitch < r8 + k8 + 8) continue;
            uint64_t owned = (t->cols + groupCores - 1) / groupCores;
            uint64_t baseBytes = (4 * pitch + 4 * r8 + 8 * t->rows + k8 + 32 + 8 * (r8 + k8)) * sizeof(float);
            uint64_t residentBytes = 2 * owned * pitch * sizeof(float);
            uint64_t capacity = ubSize > baseBytes + residentBytes ? (ubSize - baseBytes - residentBytes) / (12 * pitch * sizeof(float)) : 0;
            uint64_t tile = std::min<uint64_t>(owned, capacity);
            uint64_t factorRows = (owned + 7) / 8 * 8;
            if (owned > 64 || capacity == 0) continue;
            if (owned * pitch >= 8 * pitch && owned * pitch >= uint64_t(t->cols) * 8 + uint64_t(k8) * 8
                && uint64_t(t->cols) * 8 + 8 * pitch <= uint64_t(t->cols) * pitch && tile * pitch >= factorRows * 8) {
                t->stride = static_cast<uint32_t>(pitch);
                t->groupCount = batch;
                t->resident = static_cast<uint32_t>(owned);
                t->singleResident = 0;
                t->inverseTile = static_cast<uint32_t>(tile);
                panelLaunch = vectorCores;
                break;
            }
        }
    }
    uint32_t launchCores = std::min<uint32_t>(platform.GetCoreNumAic(), (t->cols + 1) / 2);
    if (panelLaunch) launchCores = panelLaunch;
    if (!panelLaunch && t->parallel && t->groupCount > 0 && (t->groupCount < t->batch || t->singleResident)) {
        uint32_t vectorCores = std::min<uint32_t>(platform.GetCoreNumAiv(), (t->cols + 1) / 2);
        uint32_t groupCores = vectorCores / t->groupCount;
        uint32_t owned = (t->cols + groupCores - 1) / groupCores;
        uint64_t baseBytes = (4 * uint64_t(t->stride) + 4 * r8 + 8 * t->rows + k8 + 32 + 8 * (r8 + k8)) * sizeof(float);
        uint64_t residentBytes = uint64_t(t->singleResident ? 1 : 2) * owned * t->stride * sizeof(float);
        uint64_t capacity = ubSize > baseBytes + residentBytes ? (ubSize - baseBytes - residentBytes) / (12 * t->stride * sizeof(float)) : 0;
        if (vectorCores > launchCores && owned <= 64 && capacity > 0) {
            launchCores = vectorCores;
            t->resident = owned;
            t->inverseTile = std::min<uint64_t>((t->cols + launchCores - 1) / launchCores, capacity);
        }
    }
    uint64_t localBytes = (uint64_t(t->cols) * t->stride + 4 * t->stride + 4 * r8 + 8 * t->rows + k8 + 32) * 4;
    if (t->rows <= 32) localBytes += 4 * k8 * k8 * sizeof(float);
    t->local = uint64_t(t->cols) * t->stride * 4 <= 160 * 1024 && localBytes <= ubSize;
    t->atol = at ? *at : 0.0f;
    t->rtol = rt && *rt >= 0 ? *rt : (t->atol > 0 ? 0.0f : std::max(m, n) * 1.1920928955078125e-7f);
    if (t->parallel) {
        t->local = 0;
        context->SetBlockDim(launchCores);
        context->SetScheduleMode(1);
    } else context->SetBlockDim(std::min<uint64_t>(batch, platform.GetCoreNumAiv()));
    auto ws = context->GetWorkspaceSizes(1);
    ws[0] = t->local ? 0 : platform.GetLibApiWorkSpaceSize() + batch * (t->cols * t->stride * (t->parallel ? 2 : 1) + (t->parallel ? pinv::kMetadataFloats : 0)) * sizeof(float);
    uint32_t DT_A = 0U;
    uint32_t COMPACT = t->localInverse && t->stride <= 64;
    if(COMPACT && batch>platform.GetCoreNumAiv())COMPACT=0;
    if (COMPACT) context->SetBlockDim(platform.GetCoreNumAiv());
    if(COMPACT && t->cols%8!=0)context->SetBlockDim(std::min<uint64_t>(platform.GetCoreNumAiv(),
        std::max<uint64_t>(batch,std::max<uint32_t>(1,platform.GetCoreNumAiv()/4))));

    if (!COMPACT && ws[0] == 0) ws[0] = 32;
    uint32_t TAIL = COMPACT && t->cols % 8 != 0;
    uint32_t WAVE = t->groupCount > 0 && (t->groupCount < t->batch || t->singleResident);
    uint32_t LOCAL = !COMPACT && t->localInverse && !t->parallel;
    if(LOCAL && t->cols%8!=0)TAIL=1;
    if (LOCAL) context->SetBlockDim(std::min<uint64_t>(platform.GetCoreNumAiv(),
        std::max<uint64_t>(batch, platform.GetCoreNumAic())));
    uint32_t factorRows = (t->resident + 7) / 8 * 8;
    uint32_t GROUP = !COMPACT && !LOCAL && !WAVE && t->groupCount
        && t->resident * t->stride >= 16 * t->stride
        && t->resident * t->stride >= (t->cols + k8) * 16
        && (t->cols + t->stride) * 16 <= t->cols * t->stride
        && t->inverseTile * t->stride >= factorRows * 8;
    if (GROUP && r8 % 64 != 0) {
        uint32_t target = (3 * platform.GetCoreNumAic() + platform.GetCoreNumAiv()) / 4;
        uint32_t candidate = target / t->groupCount * t->groupCount;
        if (candidate > launchCores && candidate <= platform.GetCoreNumAiv()) {
            uint32_t groupCores = candidate / t->groupCount;
            uint32_t owned = (t->cols + groupCores - 1) / groupCores;
            uint32_t factors = (owned + 7) / 8 * 8;
            uint64_t baseBytes = (4 * uint64_t(t->stride) + 4 * r8 + 8 * t->rows + k8 + 32 + 8 * (r8 + k8)) * sizeof(float);
            uint64_t totalBytes = baseBytes + 2 * uint64_t(owned) * t->stride * sizeof(float) + 12 * uint64_t(t->inverseTile) * t->stride * sizeof(float);
            if (owned <= 64 && owned >= 16 && uint64_t(owned) * t->stride >= uint64_t(t->cols + k8) * 16
                && totalBytes <= ubSize && t->inverseTile * t->stride >= factors * 8) {
                t->resident = owned;
                context->SetBlockDim(candidate);
                launchCores = candidate;
            }
        }
    }
    uint32_t PIPELINE = 0, WINDOW = 0;
    if (GROUP) {
        uint64_t pipelineK=(t->cols+7)/8*8,pipelineWide=(pipelineK+63)/64*64;
        uint64_t producerBytes=(uint64_t(t->cols)*pipelineK+4096+10*pipelineK+2*pipelineWide+64+(t->cols+15)/16*64)*sizeof(float);
        uint32_t workers=launchCores/batch;
        uint64_t owned=workers?(t->cols+workers-1)/workers:t->cols,factorRowsPipeline=(owned+7)/8*8;
        uint64_t outputSize=std::max<uint64_t>(owned*pipelineK,512);
        uint64_t workerBytes=(owned*t->stride+16*(pipelineK+16)+outputSize+8*std::max<uint64_t>(factorRowsPipeline,16)+64+5*pipelineWide+32+factorRowsPipeline+16+pipelineK)*sizeof(float);
        PIPELINE = t->m==t->n && t->cols<=255 && t->groupCount==batch && launchCores%batch==0 && workers>=2 && launchCores+batch<=platform.GetCoreNumAiv() && producerBytes<=ubSize && workerBytes<=ubSize;
        if (PIPELINE) context->SetBlockDim(launchCores + batch);
        if(!PIPELINE && t->m==t->n && t->groupCount==batch && launchCores%batch==0){
            uint32_t available=platform.GetCoreNumAiv()/batch;
            uint32_t windowWorkers=available>1?std::min<uint32_t>(workers,available-1):0;
            uint32_t window=((t->cols*3/8+15)/16)*16;
            uint32_t trailing=t->cols-window,trailingPitch=(trailing+7)/8*8;
            uint64_t capacity=std::max<uint64_t>(uint64_t(t->cols)*window,uint64_t(trailing)*trailingPitch);
            uint64_t windowProducer=(capacity+4096+10*pipelineK+2*pipelineWide+64+(t->cols+15)/16*64)*sizeof(float);
            uint64_t windowOwned=windowWorkers?(t->cols+windowWorkers-1)/windowWorkers:t->cols;
            uint64_t windowFactors=(windowOwned+7)/8*8;
            uint64_t windowOutput=std::max<uint64_t>(windowOwned*pipelineK,512);
            uint64_t windowWorker=(windowOwned*t->stride+16*(pipelineK+16)+windowOutput+8*std::max<uint64_t>(windowFactors,16)+64+5*pipelineWide+32+windowFactors+16+pipelineK)*sizeof(float);
            if(window>0 && window<t->cols && windowWorkers>=2 && windowProducer<=ubSize && windowWorker<=ubSize){
                WINDOW=1;PIPELINE=1;t->local=window;
                context->SetBlockDim((windowWorkers+1)*batch);
            }
        }
    }
    uint32_t CUBE=WINDOW && t->cols%32==0 && (t->cols+context->GetBlockDim()-1)/context->GetBlockDim()<=8 && context->GetBlockDim()%2==0;
    if(CUBE){
        context->SetBlockDim(context->GetBlockDim()/2);
        matmul_tiling::MatmulApiTiling mt(platform);
        using P=matmul_tiling::TPosition;using F=matmul_tiling::CubeFormat;using D=matmul_tiling::DataType;
        mt.SetAType(P::GM,F::ND,D::DT_INT8,false);mt.SetBType(P::GM,F::ND,D::DT_INT8,false);
        mt.SetCType(P::GM,F::ND,D::DT_INT32);
        uint32_t extent=pinv::kCubePlanes*t->cols,tile=((extent+context->GetBlockDim()-1)/context->GetBlockDim()+15)/16*16;
        mt.SetShape(tile,extent,t->cols);mt.SetOrgShape(extent,extent,t->cols);mt.SetBias(false);mt.SetFixSplit(32,128,-1);
        mt.SetMatmulConfigParams(1,false);mt.SetBufferSpace(-1,-1,-1);
        AscendC::tiling::TCubeTiling ct;
        if(mt.GetTiling(ct)!=0)return -1;
        LinalgPinvTilingData saved=*t;
        auto ext=context->GetTilingData<LinalgPinvCubeTilingData>();ext->base=saved;ext->cube=ct;
        matmul_tiling::MatmulApiTiling ft(platform);
        ft.SetAType(P::GM,F::ND,D::DT_FLOAT,false);ft.SetBType(P::GM,F::ND,D::DT_FLOAT,false);
        ft.SetCType(P::GM,F::ND,D::DT_FLOAT);
        uint32_t rowTile=((saved.cols+context->GetBlockDim()-1)/context->GetBlockDim()+15)/16*16;
        ft.SetShape(rowTile,saved.cols,saved.cols);ft.SetOrgShape(saved.cols,saved.cols,saved.cols);
        ft.SetBias(false);ft.SetFixSplit(16,64,-1);ft.SetMatmulConfigParams(1,false);ft.SetBufferSpace(-1,-1,-1);
        if(ft.GetTiling(ext->correction)!=0)return -1;
        uint64_t cubeOffset=pinv::CubeOffsetFloats(t->batch,t->cols,t->stride);
        ws[0]=platform.GetLibApiWorkSpaceSize()+cubeOffset*sizeof(float)+pinv::CubeStorageBytes(t->cols);
    }
    if(PIPELINE && !WINDOW && GROUP && !WAVE){
        uint32_t pairedVectors=std::min<uint32_t>(platform.GetCoreNumAiv(),2*platform.GetCoreNumAic());
        uint32_t pairedGroup=pairedVectors/t->batch/2*2,pairedTotal=pairedGroup*t->batch;
        if(pairedGroup>=4 && pairedTotal>=launchCores){
            uint64_t pairedWorkers=pairedGroup-2,pairedK=(t->cols+7)/8*8,pairedWide=(pairedK+63)/64*64;
            uint64_t pairedOwned=(t->cols+pairedWorkers-1)/pairedWorkers,pairedFactors=(pairedOwned+7)/8*8;
            uint64_t pairedRows=(pairedOwned*t->stride+16*(pairedK+16)+std::max<uint64_t>(pairedOwned*pairedK,512)+8*std::max<uint64_t>(pairedFactors,16)+64+5*pairedWide+32+pairedFactors+16+pairedK)*4;
            uint64_t left=(t->cols+31)/32*16,right=(t->cols-left+7)/8*8;
            uint64_t pairedCapacity=t->cols*(16+std::max<uint64_t>(left,right));
            uint64_t pairedProducer=(pairedCapacity+4096+10*pairedK+2*pairedWide+64+(t->cols+15)/16*64)*4;
            if(left<t->cols && pairedRows<=ubSize && pairedProducer<=ubSize && t->cols*16<=pinv::kMetadataFloats-pinv::kPeerExchange){
                LinalgPinvTilingData saved=*t;auto ext=context->GetTilingData<LinalgPinvCubeTilingData>();
                *ext=LinalgPinvCubeTilingData{};ext->base=saved;
                CUBE=1;context->SetBlockDim(pairedTotal/2);
            }
        }
    }
    if(GROUP && PIPELINE && CUBE && !WINDOW && !WAVE &&
       t->m==t->n && t->cols==200 &&
       uint64_t(t->batch)*13<=platform.GetCoreNumAiv() &&
       (uint64_t(t->batch)*13+1)/2<=platform.GetCoreNumAic()) {
        t->stride=456;
        context->SetBlockDim((t->batch*13+1)/2);
    }
    uint32_t RECT=0;
    context->SetFlags(COMPACT,TAIL,WAVE,LOCAL,GROUP,PIPELINE,WINDOW,CUBE,RECT);
    return 0;
}
