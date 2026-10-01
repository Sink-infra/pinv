// LinalgPinv device execution, output capture, and event timing.
#include <acl/acl.h>
#include <acl/acl_prof.h>
#include "aclnn_linalg_pinv.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct ApiError : std::runtime_error {
    std::string api;
    int code;
    ApiError(std::string name, int value) : std::runtime_error(name), api(name), code(value) {}
};
void Check(const char* api, int code) { if (code) throw ApiError(api, code); }

std::string Quote(const std::string& value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') result += '\\';
        if (c >= 32) result += char(c);
    }
    return result + "\"";
}
void StoreArray(const std::string& path, const std::vector<float>& values) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open output array");
    const char* data = reinterpret_cast<const char*>(values.data());
    for (size_t i = 0; i < values.size() * sizeof(float); ++i) stream.put(data[i]);
    if (!stream) throw std::runtime_error("cannot save output array");
}

int main(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i + 1 < argc; i += 2) args[argv[i]] = argv[i + 1];
    if ((argc - 1) % 2 || !args.count("--result")) return 2;
    std::vector<float> samples, first, final;
    std::vector<std::pair<std::string, int>> teardown;
    std::string status = "Runtime Error", failedApi, detail, outputInit = "runtime";
    int failedCode = 0;
    bool initialized = false, selected = false, profiling = false, profileInitialized = false;
    int device = 0;
    void *input = nullptr, *output = nullptr, *workspace = nullptr;
    aclTensor *tin = nullptr, *tout = nullptr;
    aclrtStream stream = nullptr;
    aclrtEvent start = nullptr, end = nullptr;
    aclprofConfig* config = nullptr;
    try {
        for (auto name : {"--input", "--output", "--shape", "--atol", "--rtol", "--warmup", "--repeats"})
            if (!args.count(name)) throw std::runtime_error("missing runner argument");
        outputInit = args.count("--output-init") ? args["--output-init"] : "runtime";
        if (outputInit != "runtime" && outputInit != "nan")
            throw std::runtime_error("invalid output initialization mode");
        int warmup = std::stoi(args["--warmup"]), repeats = std::stoi(args["--repeats"]);
        device = args.count("--device") ? std::stoi(args["--device"]) : 0;
        if (warmup < 0 || repeats < 1 || warmup > 10000 || repeats > 10000 || device < 0)
            throw std::runtime_error("invalid launch counts or device");
        double atol = std::stod(args["--atol"]), rtol = std::stod(args["--rtol"]);
        if (!std::isfinite(atol) || !std::isfinite(rtol)) throw std::runtime_error("nonfinite attribute");
        std::vector<int64_t> shape;
        std::string text = args["--shape"];
        size_t position = 0;
        while (position < text.size()) {
            auto comma = text.find(',', position);
            shape.push_back(std::stoll(text.substr(position, comma - position)));
            if (comma == std::string::npos) break;
            position = comma + 1;
        }
        if (shape.size() < 2) throw std::runtime_error("matrix shape required");
        uint64_t count = 1;
        for (int64_t dim : shape) {
            if (dim <= 0 || count > (uint64_t(1) << 28) / uint64_t(dim))
                throw std::runtime_error("invalid tensor dimensions");
            count *= uint64_t(dim);
        }
        size_t bytes = count * sizeof(float);
        std::vector<float> a(count);
        std::ifstream source(args["--input"], std::ios::binary);
        source.read(reinterpret_cast<char*>(a.data()), bytes);
        if (size_t(source.gcount()) != bytes || source.peek() != std::ifstream::traits_type::eof())
            throw std::runtime_error("input byte count differs from shape");
        first.resize(count); final.resize(count);
        auto outShape = shape;
        std::swap(outShape[outShape.size()-2], outShape[outShape.size()-1]);
        auto strides = [](const std::vector<int64_t>& dims) {
            std::vector<int64_t> result(dims.size(), 1);
            for (size_t i = dims.size()-1; i > 0; --i) result[i-1] = result[i] * dims[i];
            return result;
        };
        auto inStrides = strides(shape), outStrides = strides(outShape);
        Check("aclInit", aclInit(nullptr)); initialized = true;
        Check("aclrtSetDevice", aclrtSetDevice(device)); selected = true;
        Check("aclrtSetOpExecuteTimeOut", aclrtSetOpExecuteTimeOut(30));
        Check("aclrtCreateStream", aclrtCreateStream(&stream));
        Check("aclrtMalloc", aclrtMalloc(&input, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        Check("aclrtMalloc", aclrtMalloc(&output, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        Check("aclrtMemcpy", aclrtMemcpy(input, bytes, a.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        tin = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, inStrides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), input);
        tout = aclCreateTensor(outShape.data(), outShape.size(), ACL_FLOAT, outStrides.data(), 0, ACL_FORMAT_ND,
                               outShape.data(), outShape.size(), output);
        if (!tin || !tout) throw ApiError("aclCreateTensor", -1);
        Check("aclrtCreateEvent", aclrtCreateEvent(&start));
        Check("aclrtCreateEvent", aclrtCreateEvent(&end));
        if (args.count("--profile")) {
            const std::string& folder = args["--profile"];
            Check("aclprofInit", aclprofInit(folder.data(), folder.size())); profileInitialized = true;
            uint32_t devices[] = {uint32_t(device)};
            config = aclprofCreateConfig(devices, 1, ACL_AICORE_NONE, nullptr, ACL_PROF_TASK_TIME);
            if (!config) throw ApiError("aclprofCreateConfig", -1);
            Check("aclprofStart", aclprofStart(config)); profiling = true;
        }
        for (int iteration = 0; iteration < warmup + repeats; ++iteration) {
            uint64_t workspaceBytes = 0;
            aclOpExecutor* executor = nullptr;
            bool hermitian = args.count("--hermitian") && args["--hermitian"] == "1";
            Check("aclnnLinalgPinvGetWorkspaceSize", aclnnLinalgPinvGetWorkspaceSize(
                  tin, atol, rtol, hermitian, tout, &workspaceBytes, &executor));
            if (workspaceBytes) Check("aclrtMalloc", aclrtMalloc(&workspace, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST));
            if (outputInit == "nan")
                Check("aclrtMemset", aclrtMemset(output, bytes, 255, bytes));
            Check("aclrtRecordEvent", aclrtRecordEvent(start, stream));
            Check("aclnnLinalgPinv", aclnnLinalgPinv(workspace, workspaceBytes, executor, stream));
            Check("aclrtRecordEvent", aclrtRecordEvent(end, stream));
            Check("aclrtSynchronizeStream", aclrtSynchronizeStream(stream));
            float ms = 0;
            Check("aclrtEventElapsedTime", aclrtEventElapsedTime(&ms, start, end));
            if (iteration >= warmup) {
                if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("invalid device event duration");
                samples.push_back(ms * 1000);
            }
            if (iteration == warmup) Check("aclrtMemcpy", aclrtMemcpy(
                first.data(), bytes, output, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
            if (workspace) { Check("aclrtFree", aclrtFree(workspace)); workspace = nullptr; }
        }
        Check("aclrtMemcpy", aclrtMemcpy(final.data(), bytes, output, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        StoreArray(args["--output"], first);
        StoreArray(args["--output"] + ".final.bin", final);
        status = "executed";
    } catch (const ApiError& error) {
        failedApi = error.api; failedCode = error.code;
    } catch (const std::exception& error) {
        detail = error.what(); status = "Harness Error";
    }
    auto record = [&](const char* api, int code) { if (code) teardown.emplace_back(api, code); };
    if (profiling) record("aclprofStop", aclprofStop(config));
    if (config) record("aclprofDestroyConfig", aclprofDestroyConfig(config));
    if (profileInitialized) record("aclprofFinalize", aclprofFinalize());
    if (workspace) record("aclrtFree", aclrtFree(workspace));
    if (tin) record("aclDestroyTensor", aclDestroyTensor(tin));
    if (tout) record("aclDestroyTensor", aclDestroyTensor(tout));
    if (start) record("aclrtDestroyEvent", aclrtDestroyEvent(start));
    if (end) record("aclrtDestroyEvent", aclrtDestroyEvent(end));
    if (input) record("aclrtFree", aclrtFree(input));
    if (output) record("aclrtFree", aclrtFree(output));
    if (stream) record("aclrtDestroyStream", aclrtDestroyStream(stream));
    if (selected) record("aclrtResetDevice", aclrtResetDevice(device));
    if (initialized) record("aclFinalize", aclFinalize());
    std::ofstream report(args["--result"]);
    report << "{\"status\":" << Quote(status) << ",\"api\":" << Quote(failedApi)
           << ",\"code\":" << failedCode << ",\"detail\":" << Quote(detail)
           << ",\"output_init\":" << Quote(outputInit) << ",\"event_micros\":[";
    for (size_t i = 0; i < samples.size(); ++i) { if (i) report << ','; report << samples[i]; }
    report << "],\"teardown\":[";
    for (size_t i = 0; i < teardown.size(); ++i) {
        if (i) report << ',';
        report << "{\"api\":" << Quote(teardown[i].first) << ",\"code\":" << teardown[i].second << '}';
    }
    report << "]}\n";
    return report && status == "executed" && teardown.empty() ? 0 : 1;
}
