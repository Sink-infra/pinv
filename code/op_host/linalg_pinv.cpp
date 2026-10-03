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
    ASCENDC_TPL_SEL_PARAM(context, DT_A);
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
