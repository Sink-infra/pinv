#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_pinv_tiling.h"
#include "../op_kernel/tiling_key_linalg_pinv.h"
#include <algorithm>
#include <limits>
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto shape = context->GetInputShape(0);
    auto desc = context->GetInputDesc(0);
    auto attrs = context->GetAttrs();
    if (!shape || !desc || !attrs || desc->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    const auto &s = shape->GetStorageShape();
    size_t rank = s.GetDimNum();
    if (rank < 2) return ge::GRAPH_FAILED;
    int64_t m = s.GetDim(rank - 2), n = s.GetDim(rank - 1);
    if (m < 1 || m > 512 || n < 1 || n > 512) return ge::GRAPH_FAILED;
    uint64_t batch = 1;
    for (size_t i = 0; i + 2 < rank; ++i) {
        int64_t d = s.GetDim(i);
        if (d < 1 || d > 64 || batch > std::numeric_limits<uint32_t>::max() / uint64_t(d)) return ge::GRAPH_FAILED;
        batch *= d;
    }
    auto at = attrs->GetFloat(0); auto rt = attrs->GetFloat(1); auto herm = attrs->GetBool(2);
    if (herm && *herm) return ge::GRAPH_FAILED;
    auto t = context->GetTilingData<LinalgPinvTilingData>();
    t->m = m; t->n = n; t->batch = batch;
    t->rows = std::max(m, n); t->cols = std::min(m, n);
    t->parallel = batch <= 4 && t->cols >= 16;
    uint32_t r8 = (t->rows + 7) / 8 * 8, k8 = (t->cols + 7) / 8 * 8;
    t->stride = r8 + k8 + 8;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t inverseBytes = (2 * uint64_t(t->cols) * t->stride + 4 * t->stride + 4 * r8 + 8 * t->rows + k8 + 32 + 8 * (r8 + k8)) * 4;
    if (t->rows <= 32) inverseBytes += 4 * k8 * k8 * sizeof(float);
    if (m == n && (t->stride <= 64 || (t->cols > 32 && t->cols <= 96))) inverseBytes += (k8 - t->cols) * t->stride * sizeof(float);
    t->inverseTile = 8;
    while (t->inverseTile > 1 && inverseBytes + 12 * t->inverseTile * t->stride * sizeof(float) > ubSize) t->inverseTile /= 2;
    inverseBytes += 12 * t->inverseTile * t->stride * sizeof(float);
    t->localInverse = m == n && inverseBytes <= ubSize;
    if (t->localInverse) t->parallel = 0;
    if (t->parallel) {
        uint32_t narrowStride = (t->stride + 63) / 64 * 64;
        t->stride = (t->stride + 127) / 128 * 128;
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
    t->resident = 0; t->groupCount = 0;
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
    uint64_t localBytes = (uint64_t(t->cols) * t->stride + 4 * t->stride + 4 * r8 + 8 * t->rows + k8 + 32) * 4;
    if (t->rows <= 32) localBytes += 4 * k8 * k8 * sizeof(float);
    t->local = uint64_t(t->cols) * t->stride * 4 <= 160 * 1024 && localBytes <= ubSize;
    t->atol = at ? *at : 0.0f;
    t->rtol = rt && *rt >= 0 ? *rt : (t->atol > 0 ? 0.0f : std::max(m, n) * 1.1920928955078125e-7f);
    if (t->parallel) {
        t->local = 0;
        context->SetBlockDim(std::min<uint32_t>(platform.GetCoreNumAic(), (t->cols + 1) / 2));
        context->SetScheduleMode(1);
    } else context->SetBlockDim(std::min<uint64_t>(batch, platform.GetCoreNumAiv()));
    auto ws = context->GetWorkspaceSizes(1);
    ws[0] = t->local ? 0 : platform.GetLibApiWorkSpaceSize() + batch * (t->cols * t->stride * (t->parallel ? 2 : 1) + (t->parallel ? 8192 : 0)) * sizeof(float);
    uint32_t DT_A = static_cast<uint32_t>(ge::DT_FLOAT);
    uint32_t COMPACT = t->localInverse && t->stride <= 64;
    if (!COMPACT && ws[0] == 0) ws[0] = 32;
    uint32_t TAIL = COMPACT && t->cols % 8 != 0;
    uint32_t WAVE = t->groupCount > 0 && t->groupCount < t->batch;
    ASCENDC_TPL_SEL_PARAM(context, DT_A, COMPACT, TAIL, WAVE);
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    auto in = context->GetInputShape(0); auto out = context->GetOutputShape(0);
    if (!in || !out || in->GetDimNum() < 2) return GRAPH_FAILED;
    *out = *in;
    size_t r = in->GetDimNum();
    out->SetDim(r - 2, in->GetDim(r - 1)); out->SetDim(r - 1, in->GetDim(r - 2));
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, ge::DT_FLOAT); return GRAPH_SUCCESS;
}
}
namespace ops {
class LinalgPinv : public OpDef {
public:
    explicit LinalgPinv(const char *name) : OpDef(name) {
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("A_pinv").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("atol").AttrType(OPTIONAL).Float(0.0);
        this->Attr("rtol").AttrType(OPTIONAL).Float(-1.0);
        this->Attr("hermitian").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).LaunchWithZeroEleOutputTensors(true).AddConfig("ascend910b");
    }
};
OP_ADD(LinalgPinv);
}
