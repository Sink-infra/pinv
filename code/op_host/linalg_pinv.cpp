// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/linalg_pinv_tiling.h"
#include "../op_kernel/tiling_key_linalg_pinv.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_A = context->GetRequiredInputTensor(0);
        ge::DataType dtype_A = tensor_A->GetDataType(); // 获取数据类型
        int dtype_size_A = ge::GetSizeByDataType(dtype_A); // 获取数据类型的字长
        uint32_t length_A = tensor_A->GetShapeSize(); // 获取元素个数
        uint32_t size_A = tensor_A->GetSize(); // 获取内存大小
        // 示例: 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_atol = attrs->GetFloat(0);
        const float *attr_rtol = attrs->GetFloat(1);
        const bool *attr_hermitian = attrs->GetBool(2);
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_A = static_cast<uint32_t>(dtype_A);
        ASCENDC_TPL_SEL_PARAM(context, DT_A);
        // 示例: 计算tiling方案并填充tiling结构体
        LinalgPinvTilingData *tiling = context->GetTilingData<LinalgPinvTilingData>();
        tiling->length = length_A;
        // 配置启动核数
        context->SetBlockDim(num_cores_aiv);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class LinalgPinv : public OpDef {
    public:
        explicit LinalgPinv(const char *name) : OpDef(name) {
            this->Input("A")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("A_pinv")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("atol").AttrType(OPTIONAL).Float();
            this->Attr("rtol").AttrType(OPTIONAL).Float(-1.0);
            this->Attr("hermitian").AttrType(OPTIONAL).Bool();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(LinalgPinv);
}  // namespace ops
