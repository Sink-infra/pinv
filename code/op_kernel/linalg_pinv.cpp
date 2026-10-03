#include "kernel_operator.h"
#include "linalg_pinv_tiling.h"
#include "tiling_key_linalg_pinv.h"
using namespace AscendC;
class Pinv {
    TBuf<TPosition::VECCALC> mem, precision, vectorPrecision;
    LocalTensor<float> calc[8];
    LocalTensor<float> gh, gl, ih, il;
    LocalTensor<float> matrix, x, y, u, v, prod, tmp, red, gather, norms, output;
    GlobalTensor<float> input, result, work;
    __gm__ float *workBase;
    LinalgPinvTilingData t;
    uint32_t r8, k8, s;
    __aicore__ inline void VS() { SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0); }
    __aicore__ inline void SV() { SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0); }
    __aicore__ inline void Load(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t count) {
        DataCopyPad(dst, src, DataCopyExtParams{1, count * 4, 0, 0, 0}, DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - count % 8) % 8), 0});
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
    }
    __aicore__ inline void Store(GlobalTensor<float> dst, LocalTensor<float> src, uint32_t count) {
        SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        DataCopyPad(dst, src, DataCopyExtParams{1, count * 4, 0, 0, 0});
        SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0); WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
    __aicore__ inline LocalTensor<float> Row(uint32_t i) { return matrix[i * s]; }
    __aicore__ inline void ReadRow(LocalTensor<float> dst, uint32_t i) {
        if (t.local) { DataCopy(dst, Row(i), s); PipeBarrier<PIPE_V>(); }
        else Load(dst, work[i * s], s);
    }
    __aicore__ inline void SaveRow(LocalTensor<float> src, uint32_t i) {
        if (t.local) { DataCopy(Row(i), src, s); PipeBarrier<PIPE_V>(); }
        else Store(work[i * s], src, s);
    }
    __aicore__ inline float Dot(LocalTensor<float> a, LocalTensor<float> b) {
        Mul(prod, a, b, r8); PipeBarrier<PIPE_V>();
        ReduceSum(red, prod, tmp, r8); VS(); return red.GetValue(0);
    }
    __aicore__ inline float Root(float value) {
        red.SetValue(0, value); SV(); Sqrt(red, red, 1); VS(); return red.GetValue(0);
    }
    __aicore__ inline float AbsScalar(float value) { return value < 0 ? -value : value; }
    struct Wide { float h, l; };
    __aicore__ inline Wide Plus(Wide a, Wide b) {
        volatile float sum = a.h + b.h;
        volatile float bv = sum - a.h;
        volatile float av = sum - bv;
        volatile float br = b.h - bv;
        volatile float ar = a.h - av;
        volatile float e = ar + br;
        e = e + a.l; e = e + b.l;
        volatile float h = sum + e;
        volatile float z = h - sum;
        return {h, e - z};
    }
    __aicore__ inline Wide Minus(Wide a, Wide b) { return Plus(a, {-b.h, -b.l}); }
    __aicore__ inline Wide Times(Wide a, Wide b) {
        volatile float ahc = 4097.0f * a.h, bhc = 4097.0f * b.h;
        volatile float ahv = ahc - a.h, bhv = bhc - b.h;
        volatile float ah = ahc - ahv, bh = bhc - bhv;
        volatile float al = a.h - ah, bl = b.h - bh;
        volatile float p = a.h * b.h;
        volatile float e = ah * bh; e = e - p;
        volatile float x1 = ah * bl; e = e + x1;
        volatile float x2 = al * bh; e = e + x2;
        volatile float x3 = al * bl; e = e + x3;
        volatile float x4 = a.h * b.l; e = e + x4;
        volatile float x5 = a.l * b.h; e = e + x5;
        return Plus({p, 0}, {e, 0});
    }
    __aicore__ inline Wide Divide(Wide a, Wide b) {
        float q = a.h / b.h;
        Wide residual = Minus(a, Times(b, {q, 0}));
        return Plus({q, 0}, {residual.h / b.h, 0});
    }
    __aicore__ inline Wide Entry(LocalTensor<float> h, LocalTensor<float> l, uint32_t pos) {
        return {h.GetValue(pos), l.GetValue(pos)};
    }
    __aicore__ inline void SaveEntry(LocalTensor<float> h, LocalTensor<float> l, uint32_t pos, Wide val) {
        h.SetValue(pos, val.h); l.SetValue(pos, val.l);
    }
    __aicore__ inline void AccurateFull(uint32_t offset, float scale) {
        float power = 1.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        for (uint32_t i = 0; i < t.cols; ++i) {
            Duplicate(x, 0.0f, s); PipeBarrier<PIPE_V>();
            if (t.m >= t.n) {
                DataCopyPad(gather, input[offset + i], DataCopyExtParams{static_cast<uint16_t>(t.m), 4, (t.n - 1) * 4, 0, 0}, DataCopyPadExtParams<float>{true, 0, 7, 0});
                SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0); VS();
                for (uint32_t j = 0; j < t.rows; ++j) x.SetValue(j, gather.GetValue(j * 8));
                SV();
            } else Load(x, input[offset + i * t.n], t.n);
            Muls(x, x, 1.0f / power, r8); PipeBarrier<PIPE_V>(); SaveRow(x, i);
        }
        for (uint32_t i = 0; i < t.cols; ++i) {
            ReadRow(x, i);
            for (uint32_t j = 0; j < t.cols; ++j) {
                ReadRow(y, j); VS(); Wide sum = {0, 0};
                for (uint32_t k = 0; k < t.rows; ++k) sum = Plus(sum, Times({x.GetValue(k), 0}, {y.GetValue(k), 0}));
                SaveEntry(gh, gl, i * k8 + j, sum);
                SaveEntry(ih, il, i * k8 + j, {i == j ? 1.0f : 0.0f, 0});
            }
        }
        for (uint32_t p = 0; p < t.cols; ++p) {
            uint32_t pivot = p; float best = AbsScalar(gh.GetValue(p * k8 + p));
            for (uint32_t i = p + 1; i < t.cols; ++i) {
                float value = AbsScalar(gh.GetValue(i * k8 + p));
                if (value > best) { best = value; pivot = i; }
            }
            if (pivot != p) for (uint32_t j = 0; j < t.cols; ++j) {
                Wide a = Entry(gh, gl, p * k8 + j), b = Entry(gh, gl, pivot * k8 + j);
                SaveEntry(gh, gl, p * k8 + j, b); SaveEntry(gh, gl, pivot * k8 + j, a);
                a = Entry(ih, il, p * k8 + j); b = Entry(ih, il, pivot * k8 + j);
                SaveEntry(ih, il, p * k8 + j, b); SaveEntry(ih, il, pivot * k8 + j, a);
            }
            Wide divisor = Entry(gh, gl, p * k8 + p);
            for (uint32_t j = 0; j < t.cols; ++j) {
                SaveEntry(gh, gl, p * k8 + j, Divide(Entry(gh, gl, p * k8 + j), divisor));
                SaveEntry(ih, il, p * k8 + j, Divide(Entry(ih, il, p * k8 + j), divisor));
            }
            for (uint32_t i = 0; i < t.cols; ++i) if (i != p) {
                Wide factor = Entry(gh, gl, i * k8 + p);
                for (uint32_t j = 0; j < t.cols; ++j) {
                    SaveEntry(gh, gl, i * k8 + j, Minus(Entry(gh, gl, i * k8 + j), Times(factor, Entry(gh, gl, p * k8 + j))));
                    SaveEntry(ih, il, i * k8 + j, Minus(Entry(ih, il, i * k8 + j), Times(factor, Entry(ih, il, p * k8 + j))));
                }
            }
        }
        for (uint32_t row = 0; row < t.n; ++row) {
            for (uint32_t col = 0; col < t.m; ++col) {
                Wide sum = {0, 0};
                for (uint32_t i = 0; i < t.cols; ++i) {
                    ReadRow(x, i); VS();
                    Wide inv = Entry(ih, il, (t.m >= t.n ? row : col) * k8 + i);
                    sum = Plus(sum, Times(inv, {x.GetValue(t.m >= t.n ? col : row), 0}));
                }
                output.SetValue(col, (sum.h + sum.l) / power);
            }
            Store(result[offset + row * t.m], output, t.m);
        }
    }
    __aicore__ inline void VectorEliminate(Wide factor) {
        auto qh = calc[0], ql = calc[1], bh = calc[2], bl = calc[3], e = calc[4], part = calc[5], product = calc[6], aux = calc[7];
        volatile float ac = factor.h * 4097.0f;
        volatile float az = ac - factor.h;
        volatile float ah = ac - az;
        volatile float al = factor.h - ah;
        Muls(bh, u, 4097.0f, s); PipeBarrier<PIPE_V>();
        Sub(bl, bh, u, s); PipeBarrier<PIPE_V>(); Sub(bh, bh, bl, s); PipeBarrier<PIPE_V>();
        Sub(bl, u, bh, s); Muls(product, u, factor.h, s); PipeBarrier<PIPE_V>();
        Muls(e, bh, ah, s); PipeBarrier<PIPE_V>(); Sub(e, e, product, s); PipeBarrier<PIPE_V>();
        Muls(part, bl, ah, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, bh, al, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, bl, al, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, v, factor.h, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, u, factor.l, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Add(qh, product, e, s); PipeBarrier<PIPE_V>(); Sub(ql, qh, product, s); PipeBarrier<PIPE_V>(); Sub(ql, e, ql, s); PipeBarrier<PIPE_V>();
        Sub(bh, x, qh, s); PipeBarrier<PIPE_V>(); Sub(bl, bh, x, s); PipeBarrier<PIPE_V>();
        Sub(e, bh, bl, s); PipeBarrier<PIPE_V>(); Sub(e, x, e, s); Muls(part, qh, -1.0f, s); PipeBarrier<PIPE_V>();
        Sub(part, part, bl, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Add(e, e, y, s); PipeBarrier<PIPE_V>(); Sub(e, e, ql, s); PipeBarrier<PIPE_V>();
        Add(x, bh, e, s); PipeBarrier<PIPE_V>(); Sub(aux, x, bh, s); PipeBarrier<PIPE_V>(); Sub(y, e, aux, s); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void AccurateSquare(uint32_t b, uint32_t offset, float scale, GlobalTensor<float> meta) {
        uint32_t core = GetBlockIdx(), cores = GetBlockNum();
        GlobalTensor<float> low;
        low.SetGlobalBuffer(workBase + uint64_t(t.batch + b) * t.cols * s);
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        for (uint32_t i = core; i < t.cols; i += cores) {
            Duplicate(x, 0.0f, s); Duplicate(y, 0.0f, s); PipeBarrier<PIPE_V>();
            Load(x, input[offset + i * t.n], t.n);
            Muls(x, x, 1.0f / power, r8); PipeBarrier<PIPE_V>(); VS();
            x.SetValue(r8 + i, 1.0f); SV(); SaveRow(x, i); Store(low[i * s], y, s);
        }
        SyncAll();
        for (uint32_t p = 0; p < t.cols; ++p) {
            float best = 0; uint32_t bestRow = p;
            for (uint32_t i = core; i < t.cols; i += cores) if (i >= p) {
                ReadRow(x, i); VS(); float value = AbsScalar(x.GetValue(p));
                if (value > best) { best = value; bestRow = i; }
            }
            red.SetValue(0, best); Store(meta[core * 8], red, 1);
            red.SetValue(0, static_cast<float>(static_cast<int32_t>(bestRow))); Store(meta[256 + core * 8], red, 1);
            SyncAll(); Load(gather, meta, cores * 8); best = 0; uint32_t bestCore = 0;
            for (uint32_t i = 0; i < cores; ++i) { float value = gather.GetValue(i * 8); if (value > best) { best = value; bestCore = i; } }
            Load(gather, meta[256], cores * 8);
            uint32_t pivot = static_cast<uint32_t>(static_cast<int32_t>(gather.GetValue(bestCore * 8)));
            if (core == 0) {
                if (pivot != p) {
                    ReadRow(x, pivot); ReadRow(y, p); Load(u, low[pivot * s], s); Load(v, low[p * s], s);
                    SaveRow(x, p); SaveRow(y, pivot); Store(low[p * s], u, s); Store(low[pivot * s], v, s);
                }
                ReadRow(x, p); Load(y, low[p * s], s); VS();
                Wide divisor = {x.GetValue(p), y.GetValue(p)};
                for (uint32_t j = 0; j < s; ++j) {
                    Wide value = Divide({x.GetValue(j), y.GetValue(j)}, divisor);
                    x.SetValue(j, value.h); y.SetValue(j, value.l);
                }
                SaveRow(x, p); Store(low[p * s], y, s);
            }
            SyncAll(); ReadRow(u, p); Load(v, low[p * s], s); VS();
            for (uint32_t i = core; i < t.cols; i += cores) if (i != p) {
                ReadRow(x, i); Load(y, low[i * s], s); VS();
                Wide factor = {x.GetValue(p), y.GetValue(p)};
                if (factor.h == 0 && factor.l == 0) continue;
                SV(); VectorEliminate(factor);
                SaveRow(x, i); Store(low[i * s], y, s);
            }
            SyncAll();
        }
        for (uint32_t row = core; row < t.n; row += cores) {
            ReadRow(x, row); Load(y, low[row * s], s); VS();
            for (uint32_t j = 0; j < t.m; ++j) output.SetValue(j, (x.GetValue(r8 + j) + y.GetValue(r8 + j)) / power);
            Store(result[offset + row * t.m], output, t.m);
        }
        SyncAll();
    }
    __aicore__ inline void Parallel() {
        uint32_t core = GetBlockIdx(), cores = GetBlockNum();
        uint32_t roundCols = (t.cols + 1) / 2 * 2;
        for (uint32_t b = 0; b < t.batch; ++b) {
            uint32_t offset = b * t.m * t.n;
            work.SetGlobalBuffer(workBase + uint64_t(b) * t.cols * s);
            GlobalTensor<float> meta;
            meta.SetGlobalBuffer(workBase + uint64_t(2) * t.batch * t.cols * s + uint64_t(b) * 8192);
            float scale = 0;
            for (uint32_t i = core; i < t.cols; i += cores) {
                Duplicate(x, 0.0f, s); PipeBarrier<PIPE_V>();
                if (t.m >= t.n) {
                    DataCopyPad(gather, input[offset + i], DataCopyExtParams{static_cast<uint16_t>(t.m), 4, (t.n - 1) * 4, 0, 0}, DataCopyPadExtParams<float>{true, 0, 7, 0});
                    SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0); VS();
                    for (uint32_t j = 0; j < t.rows; ++j) x.SetValue(j, gather.GetValue(j * 8));
                } else Load(x, input[offset + i * t.n], t.n);
                VS(); x.SetValue(r8 + i, 1.0f); SV();
                Abs(prod, x, r8); PipeBarrier<PIPE_V>(); ReduceMax(red, prod, tmp, r8, false); VS();
                float mx = red.GetValue(0); scale = mx > scale ? mx : scale; SaveRow(x, i);
            }
            red.SetValue(0, scale); Store(meta[core * 8], red, 1); SyncAll();
            Load(gather, meta, cores * 8); scale = 0;
            for (uint32_t i = 0; i < cores; ++i) { float v = gather.GetValue(i * 8); scale = v > scale ? v : scale; }
            if (scale == 0) {
                Duplicate(output, 0.0f, r8); PipeBarrier<PIPE_V>();
                for (uint32_t row = core; row < t.n; row += cores) Store(result[offset + row * t.m], output, t.m);
                SyncAll(); continue;
            }
            float total = 0;
            for (uint32_t i = core; i < t.cols; i += cores) {
                ReadRow(x, i); Muls(x, x, 1.0f / scale, r8); PipeBarrier<PIPE_V>(); total += Dot(x, x); SaveRow(x, i);
            }
            red.SetValue(0, total); Store(meta[core * 8], red, 1); SyncAll();
            Load(gather, meta, cores * 8); total = 0;
            for (uint32_t i = 0; i < cores; ++i) total += gather.GetValue(i * 8);
            float absTol = t.atol / scale, discard = absTol * absTol;
            float rel = t.rtol * t.rtol * total;
            discard = (discard > rel ? discard : rel) * 1.0e-8f;
            for (uint32_t sweep = 0; sweep < 40; ++sweep) {
                uint32_t changed = 0;
                for (uint32_t round = 0; round + 1 < roundCols; ++round) {
                    for (uint32_t slot = core; slot < roundCols / 2; slot += cores) {
                        uint32_t p = slot == 0 ? roundCols - 1 : (round + slot) % (roundCols - 1);
                        uint32_t q = slot == 0 ? round : (round + roundCols - 1 - slot) % (roundCols - 1);
                        if (p >= t.cols || q >= t.cols) continue;
                        ReadRow(x, p); ReadRow(y, q);
                        float a = Dot(x, x), d = Dot(y, y), h = Dot(x, y);
                        if (a <= discard || d <= discard || h == 0 || AbsScalar(h) <= 1.0e-7f * Root(a * d)) continue;
                        float tau = (d - a) / (2 * h), at = AbsScalar(tau);
                        float z = (tau < 0 ? -1.0f : 1.0f) / (at + Root(1 + at * at));
                        float c = 1.0f / Root(1 + z * z), sn = c * z;
                        Muls(u, x, c, s); Muls(v, y, -sn, s); PipeBarrier<PIPE_V>(); Add(u, u, v, s); PipeBarrier<PIPE_V>();
                        Muls(v, x, sn, s); Muls(y, y, c, s); PipeBarrier<PIPE_V>(); Add(y, y, v, s); PipeBarrier<PIPE_V>();
                        SaveRow(u, p); SaveRow(y, q); ++changed;
                    }
                    SyncAll();
                }
                red.SetValue(0, static_cast<float>(static_cast<int32_t>(changed))); Store(meta[core * 8], red, 1); SyncAll();
                Load(gather, meta, cores * 8); float sum = 0;
                for (uint32_t i = 0; i < cores; ++i) sum += gather.GetValue(i * 8);
                if (sum == 0) break;
            }
            for (uint32_t i = core; i < t.cols; i += cores) {
                ReadRow(x, i); float a = Dot(x, x); red.SetValue(0, a); Store(meta[512 + i * 8], red, 1);
            }
            SyncAll();
            DataCopyPad(gather, meta[512], DataCopyExtParams{static_cast<uint16_t>(t.cols), 4, 28, 0, 0}, DataCopyPadExtParams<float>{true, 0, 7, 0});
            SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
            float maxNorm = 0;
            for (uint32_t i = 0; i < t.cols; ++i) { float a = gather.GetValue(i * 8); norms.SetValue(i, a); maxNorm = a > maxNorm ? a : maxNorm; }
            float threshold = absTol * absTol, relative = maxNorm * t.rtol * t.rtol;
            threshold = threshold > relative ? threshold : relative;
            uint32_t kept = 0;
            for (uint32_t i = 0; i < t.cols; ++i) {
                float a = norms.GetValue(i); if (a > threshold) ++kept;
                norms.SetValue(i, a > threshold ? 1.0f / (a * scale) : 0.0f);
            }
            if (t.m == t.n && t.cols >= 256 && kept == t.cols) { AccurateSquare(b, offset, scale, meta); continue; }
            if (t.rows <= 32 && kept == t.cols && scale < 0.001f) {
                if (core == 0) AccurateFull(offset, scale);
                SyncAll(); continue;
            }
            for (uint32_t row = core; row < t.n; row += cores) {
                Duplicate(output, 0.0f, r8); PipeBarrier<PIPE_V>();
                for (uint32_t i = 0; i < t.cols; ++i) {
                    float inv = norms.GetValue(i); if (inv == 0) continue;
                    ReadRow(x, i); VS();
                    float factor = x.GetValue(t.m >= t.n ? r8 + row : row) * inv;
                    Muls(prod, t.m >= t.n ? x : x[r8], factor, t.m >= t.n ? r8 : k8); PipeBarrier<PIPE_V>();
                    Add(output, output, prod, t.m >= t.n ? r8 : k8); PipeBarrier<PIPE_V>();
                }
                Store(result[offset + row * t.m], output, t.m);
            }
            SyncAll();
        }
    }
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR out, GM_ADDR ws, const LinalgPinvTilingData &data, TPipe *pipe) {
        t = data; r8 = (t.rows + 7) / 8 * 8; k8 = (t.cols + 7) / 8 * 8; s = r8 + k8;
        uint32_t size = (t.local ? t.cols * s : 0) + 4 * s + 4 * r8 + 8 * t.rows + k8 + 32;
        pipe->InitBuffer(mem, size * 4);
        auto all = mem.Get<float>(); uint32_t off = 0;
        matrix = all; off += t.local ? t.cols * s : 0;
        x = all[off]; off += s; y = all[off]; off += s; u = all[off]; off += s; v = all[off]; off += s;
        prod = all[off]; off += r8; tmp = all[off]; off += r8; output = all[off]; off += r8;
        red = all[off]; off += 32; gather = all[off]; off += 8 * t.rows;
        norms = all[off];
        if (t.parallel) {
            pipe->InitBuffer(vectorPrecision, 8 * s * sizeof(float));
            auto area = vectorPrecision.Get<float>();
            for (uint32_t i = 0; i < 8; ++i) calc[i] = area[i * s];
        }
        if (t.rows <= 32) {
            pipe->InitBuffer(precision, 4 * k8 * k8 * sizeof(float));
            gh = precision.Get<float>(); gl = gh[k8 * k8]; ih = gl[k8 * k8]; il = ih[k8 * k8];
        }
        input.SetGlobalBuffer((__gm__ float *)a); result.SetGlobalBuffer((__gm__ float *)out);
        workBase = (__gm__ float *)ws; work.SetGlobalBuffer(workBase);
    }
    __aicore__ inline void Process() {
        if (t.parallel) { Parallel(); return; }
        for (uint32_t b = GetBlockIdx(); b < t.batch; b += GetBlockNum()) {
            uint32_t offset = b * t.m * t.n;
            if (!t.local) work.SetGlobalBuffer(workBase + uint64_t(b) * t.cols * s);
            float scale = 0;
            for (uint32_t i = 0; i < t.cols; ++i) {
                Duplicate(x, 0.0f, s); PipeBarrier<PIPE_V>();
                if (t.m >= t.n) {
                    DataCopyPad(gather, input[offset + i], DataCopyExtParams{static_cast<uint16_t>(t.m), 4, (t.n - 1) * 4, 0, 0}, DataCopyPadExtParams<float>{true, 0, 7, 0});
                    SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0); VS();
                    for (uint32_t j = 0; j < t.rows; ++j) x.SetValue(j, gather.GetValue(j * 8));
                } else Load(x, input[offset + i * t.n], t.n);
                VS(); x.SetValue(r8 + i, 1.0f); SV();
                Abs(prod, x, r8); PipeBarrier<PIPE_V>(); ReduceMax(red, prod, tmp, r8, false); VS();
                float mx = red.GetValue(0); scale = mx > scale ? mx : scale;
                SaveRow(x, i);
            }
            if (scale == 0) {
                Duplicate(output, 0.0f, r8); PipeBarrier<PIPE_V>();
                for (uint32_t i = 0; i < t.n; ++i) Store(result[offset + i * t.m], output, t.m);
                continue;
            }
            float total = 0;
            for (uint32_t i = 0; i < t.cols; ++i) {
                ReadRow(x, i); Muls(x, x, 1.0f / scale, r8); PipeBarrier<PIPE_V>();
                total += Dot(x, x); SaveRow(x, i);
            }
            float absTol = t.atol / scale;
            float discard = absTol * absTol;
            float rel = t.rtol * t.rtol * total;
            discard = (discard > rel ? discard : rel) * 1.0e-8f;
            for (uint32_t sweep = 0; sweep < 40; ++sweep) {
                uint32_t changed = 0;
                for (uint32_t p = 0; p < t.cols; ++p) {
                    ReadRow(x, p);
                    for (uint32_t q = p + 1; q < t.cols; ++q) {
                        ReadRow(y, q);
                        float a = Dot(x, x), d = Dot(y, y), h = Dot(x, y);
                        if (a <= discard || d <= discard || h == 0 || AbsScalar(h) <= 1.0e-7f * Root(a * d)) continue;
                        float tau = (d - a) / (2 * h);
                        float at = AbsScalar(tau);
                        float z = (tau < 0 ? -1.0f : 1.0f) / (at + Root(1 + at * at));
                        float c = 1.0f / Root(1 + z * z), sn = c * z;
                        Muls(u, x, c, s); Muls(v, y, -sn, s); PipeBarrier<PIPE_V>(); Add(u, u, v, s); PipeBarrier<PIPE_V>();
                        Muls(v, x, sn, s); Muls(y, y, c, s); PipeBarrier<PIPE_V>(); Add(y, y, v, s); PipeBarrier<PIPE_V>();
                        DataCopy(x, u, s); PipeBarrier<PIPE_V>(); SaveRow(y, q); ++changed;
                    }
                    SaveRow(x, p);
                }
                if (changed == 0) break;
            }
            float maxNorm = 0;
            for (uint32_t i = 0; i < t.cols; ++i) {
                ReadRow(x, i); float a = Dot(x, x); norms.SetValue(i, a); maxNorm = a > maxNorm ? a : maxNorm;
            }
            float threshold = absTol * absTol;
            float relative = maxNorm * t.rtol * t.rtol;
            threshold = threshold > relative ? threshold : relative;
            uint32_t kept = 0;
            for (uint32_t i = 0; i < t.cols; ++i) {
                float a = norms.GetValue(i); if (a > threshold) ++kept; norms.SetValue(i, a > threshold ? 1.0f / (a * scale) : 0.0f);
            }
            if (t.rows <= 32 && kept == t.cols && scale < 0.001f) { AccurateFull(offset, scale); continue; }
            for (uint32_t row = 0; row < t.n; ++row) {
                Duplicate(output, 0.0f, r8); PipeBarrier<PIPE_V>();
                for (uint32_t i = 0; i < t.cols; ++i) {
                    float inv = norms.GetValue(i); if (inv == 0) continue;
                    ReadRow(x, i); VS();
                    float factor = x.GetValue(t.m >= t.n ? r8 + row : row) * inv;
                    Muls(prod, t.m >= t.n ? x : x[r8], factor, t.m >= t.n ? r8 : k8); PipeBarrier<PIPE_V>();
                    Add(output, output, prod, t.m >= t.n ? r8 : k8); PipeBarrier<PIPE_V>();
                }
                Store(result[offset + row * t.m], output, t.m);
            }
        }
    }
};
template <typename DT_A>
__global__ __aicore__ void linalg_pinv(GM_ADDR A, GM_ADDR A_pinv, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(LinalgPinvTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData, data, tiling);
    TPipe pipe; Pinv op;
    op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
}
