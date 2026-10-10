#include "kernel_operator.h"
#include "linalg_pinv_tiling.h"
#include "tiling_key_linalg_pinv.h"
using namespace AscendC;
template <bool WAVE, bool LOCAL>
class StablePinv {
    TBuf<TPosition::VECCALC> mem, precision, vectorPrecision, inverseLow, batchPrecision, parallelElimination;
    LocalTensor<float> batchCalc[8], batchU, batchV, batchH, batchL;
    LocalTensor<float> tileHigh, tileLow;
    LocalTensor<float> lowMatrix;
    LocalTensor<float> calc[8];
    LocalTensor<float> gh, gl, ih, il;
    LocalTensor<float> matrix, x, y, u, v, prod, tmp, red, gather, norms, output;
    GlobalTensor<float> input, result, work;
    __gm__ float *workBase;
    LinalgPinvTilingData t;
    uint32_t r8, k8, s, stride, rowLength, lowElements, residentElements;
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
    __aicore__ inline LocalTensor<float> Row(uint32_t i) { return matrix[i * stride]; }
    __aicore__ inline void ReadRow(LocalTensor<float> dst, uint32_t i) {
        if (t.local) { DataCopy(dst, Row(i), rowLength); PipeBarrier<PIPE_V>(); }
        else Load(dst, work[i * stride], rowLength);
    }
    __aicore__ inline void SaveRow(LocalTensor<float> src, uint32_t i) {
        if (t.local) { DataCopy(Row(i), src, rowLength); PipeBarrier<PIPE_V>(); }
        else Store(work[i * stride], src, rowLength);
    }
    __aicore__ inline float Dot(LocalTensor<float> a, LocalTensor<float> b) {
        Mul(prod, a, b, r8); PipeBarrier<PIPE_V>();
        if (r8 <= 64) WholeReduceSum(red, prod, r8, 1, 1, 1, 8);
        else ReduceSum(red, prod, tmp, r8);
        VS(); return red.GetValue(0);
    }
    __aicore__ inline void CacheNorms() {
        uint32_t first = t.parallel ? GetBlockIdx() : 0;
        uint32_t step = t.parallel ? GetBlockNum() : 1;
        for (uint32_t i = first; i < t.cols; i += step) {
            ReadRow(x, i); float a = Dot(x, x); x.SetValue(s, a);
            if (t.local) SV();
            SaveRow(x, i);
        }
        if (t.parallel) SyncAll();
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
            Duplicate(x, 0.0f, rowLength); PipeBarrier<PIPE_V>();
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
        Muls(e, bh, static_cast<float>(ah), s); PipeBarrier<PIPE_V>(); Sub(e, e, product, s); PipeBarrier<PIPE_V>();
        Muls(part, bl, static_cast<float>(ah), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, bh, static_cast<float>(al), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, bl, static_cast<float>(al), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, v, factor.h, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, u, factor.l, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Add(qh, product, e, s); PipeBarrier<PIPE_V>(); Sub(ql, qh, product, s); PipeBarrier<PIPE_V>(); Sub(ql, e, ql, s); PipeBarrier<PIPE_V>();
        Sub(bh, x, qh, s); PipeBarrier<PIPE_V>(); Sub(bl, bh, x, s); PipeBarrier<PIPE_V>();
        Sub(e, bh, bl, s); PipeBarrier<PIPE_V>(); Sub(e, x, e, s); Muls(part, qh, -1.0f, s); PipeBarrier<PIPE_V>();
        Sub(part, part, bl, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Add(e, e, y, s); PipeBarrier<PIPE_V>(); Sub(e, e, ql, s); PipeBarrier<PIPE_V>();
        Add(x, bh, e, s); PipeBarrier<PIPE_V>(); Sub(aux, x, bh, s); PipeBarrier<PIPE_V>(); Sub(y, e, aux, s); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void VectorNormalize(Wide divisor) {
        float reciprocal = 1.0f / divisor.h;
        Muls(u, x, reciprocal, s); Duplicate(v, 0.0f, s); PipeBarrier<PIPE_V>();
        VectorEliminate(divisor);
        Muls(calc[0], x, reciprocal, s); PipeBarrier<PIPE_V>();
        Add(x, u, calc[0], s); PipeBarrier<PIPE_V>();
        Sub(y, x, u, s); PipeBarrier<PIPE_V>();
        Sub(y, calc[0], y, s); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void GlobalPivot(uint32_t cores, float &best, uint32_t &pivot) {
        if (cores == 0 || cores > 128 || (WAVE && cores < 8)) {
            for (uint32_t i = 0; i < cores; ++i) {
                float value = gather.GetValue(i * 8);
                if (value > best) { best = value; pivot = static_cast<uint32_t>(static_cast<int32_t>(gather.GetValue(i * 8 + 1))); }
            }
            return;
        }
        uint32_t groups = cores / 8, tail = cores % 8;
        uint64_t active[2] = {0x0101010101010101ULL, 0};
        WholeReduceMax(red, gather, active, groups, 1, 1, 8);
        if (tail) {
            PipeBarrier<PIPE_V>();
            active[0] &= (uint64_t(1) << (tail * 8)) - 1;
            WholeReduceMax(red[groups * 2], gather[groups * 64], active, 1, 1, 1, 8);
        }
        VS(); uint32_t selected = 0;
        for (uint32_t i = 0; i < groups + (tail != 0); ++i) {
            float value = red.GetValue(i * 2);
            if (value > best) { best = value; selected = i * 64 + red.ReinterpretCast<uint32_t>().GetValue(i * 2 + 1); }
        }
        if (best > 0.0f) pivot = static_cast<uint32_t>(static_cast<int32_t>(gather.GetValue(selected + 1)));
    }
    __aicore__ inline bool RefineResident(uint32_t b, uint32_t offset, float power,
                                          float total, float inverseTotal, GlobalTensor<float> meta,
                                          bool certify, float cutoff) {
        uint32_t core = GetBlockIdx(), cores = GetBlockNum();
        if constexpr (WAVE) if ((t.cols + cores - 1) / cores * r8 > lowElements) return false;
        uint32_t owned = (t.cols + cores - 1 - core) / cores;
        auto bx = x, by = y, bu = u, bv = v, cache = calc[0];
        GlobalTensor<float> residual, normWork;
        residual.SetGlobalBuffer(workBase + uint64_t(t.batch + b) * t.cols * stride);
        normWork.SetGlobalBuffer(workBase + uint64_t(b) * t.cols * stride);
        for (uint32_t row = 0; row < owned; ++row) {
            VS(); uint32_t target = static_cast<uint32_t>(static_cast<int32_t>(norms.GetValue(core + row * cores))) - 1;
            Store(result[offset + target * t.m], tileHigh[row * stride + r8], t.m);
        }
        SyncAll();
        for (uint32_t row = 0; row < owned; ++row) {
            Load(cache[row * r8], input[offset + (core + row * cores) * t.n], t.n);
            Muls(cache[row * r8], cache[row * r8], 1.0f / power, r8); PipeBarrier<PIPE_V>();
        }
        Duplicate(tileHigh, 0.0f, owned * r8); Duplicate(tileLow, 0.0f, owned * r8); PipeBarrier<PIPE_V>(); VS();
        for (uint32_t row = 0; row < owned; ++row) tileHigh.SetValue(row * r8 + core + row * cores, 1.0f);
        SV();
        uint32_t packedCount = t.inverseTile * r8;
        for (uint32_t k = 0; k < t.cols; ++k) {
            SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
            Load(u, result[offset + k * t.m], t.m);
            for (uint32_t row = 0; row < t.inverseTile; ++row) {
                DataCopy(batchU[row * r8], u, r8); Duplicate(batchV[row * r8], 0.0f, r8);
            }
            PipeBarrier<PIPE_V>();
            Muls(batchCalc[0], batchU, 4097.0f, packedCount); PipeBarrier<PIPE_V>();
            Sub(batchCalc[1], batchCalc[0], batchU, packedCount); PipeBarrier<PIPE_V>();
            Sub(batchCalc[0], batchCalc[0], batchCalc[1], packedCount); PipeBarrier<PIPE_V>();
            Sub(batchCalc[1], batchU, batchCalc[0], packedCount); PipeBarrier<PIPE_V>();
            for (uint32_t first = 0; first < owned; first += t.inverseTile) {
                uint32_t rows = owned - first < t.inverseTile ? owned - first : t.inverseTile;
                for (uint32_t row = 0; row < rows; ++row) {
                    float factor = cache.GetValue((first + row) * r8 + k);
                    Duplicate(batchH[row * r8], factor, r8); Duplicate(batchL[row * r8], 0.0f, r8);
                }
                PipeBarrier<PIPE_V>(); x = tileHigh[first * r8]; y = tileLow[first * r8];
                BatchEliminate(rows * r8, true); VS();
            }
        }
        x = bx; y = by; u = bu; v = bv;
        for (uint32_t row = 0; row < owned; ++row) {
            Add(tileHigh[row * r8], tileHigh[row * r8], tileLow[row * r8], r8); PipeBarrier<PIPE_V>();
            Store(residual[(core + row * cores) * r8], tileHigh[row * r8], t.n);
            Load(cache[row * r8], result[offset + (core + row * cores) * t.m], t.m);
        }
        Duplicate(x, 0.0f, r8); Duplicate(y, 0.0f, r8); PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < owned; ++row) {
            Mul(prod, tileHigh[row * r8], tileHigh[row * r8], r8); PipeBarrier<PIPE_V>(); Add(x, x, prod, r8); PipeBarrier<PIPE_V>();
            Mul(prod, cache[row * r8], cache[row * r8], r8); PipeBarrier<PIPE_V>(); Add(y, y, prod, r8); PipeBarrier<PIPE_V>();
        }
        Store(normWork[core * r8], x, r8); Store(normWork[(cores + core) * r8], y, r8); SyncAll();
        Duplicate(gather, 0.0f, r8); Duplicate(v, 0.0f, r8); PipeBarrier<PIPE_V>();
        for (uint32_t peer = 0; peer < cores; ++peer) {
            SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0); Load(output, normWork[peer * r8], r8);
            Add(gather, gather, output, r8); PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0); Load(output, normWork[(cores + peer) * r8], r8);
            Add(v, v, output, r8); PipeBarrier<PIPE_V>();
        }
        ReduceSum(red, gather, tmp, r8); VS(); float rf = Root(red.GetValue(0)) * 1.001f;
        float af = Root(total) * 1.001f, yf = Root(inverseTotal) * 1.001f;
        constexpr float unit = 5.9604644775390625e-8f;
        float gd = 32.0f * t.cols * unit * unit;
        float gamma = (2.0f * t.cols * unit) / (1.0f - 2.0f * t.cols * unit);
        float er = gd * af * yf + unit * rf, denominator = 1.0f - rf - er;
        if (!(denominator > 0.0f)) return false;
        Sqrt(gather, gather, r8); Sqrt(v, v, r8); PipeBarrier<PIPE_V>();
        Muls(gather, gather, 1.001f, r8); Muls(v, v, 1.001f, r8); PipeBarrier<PIPE_V>();
        Muls(output, v, gd * af, r8); Muls(prod, gather, unit, r8); PipeBarrier<PIPE_V>(); Add(output, output, prod, r8); PipeBarrier<PIPE_V>();
        Duplicate(tileLow, 0.0f, owned * r8); PipeBarrier<PIPE_V>();
        for (uint32_t k = 0; k < t.cols; ++k) {
            SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0); Load(u, residual[k * r8], t.n);
            for (uint32_t row = 0; row < owned; ++row) Duplicate(batchCalc[0][row * 8], cache.GetValue(row * r8 + k), 8);
            PipeBarrier<PIPE_V>();
            for (uint32_t col = 0; col < r8; col += 64) {
                uint32_t mask = r8 - col < 64 ? r8 - col : 64;
                MulAddDst(tileLow[col], u[col], batchCalc[0], static_cast<uint64_t>(mask), static_cast<uint8_t>(owned), {1, 1, 0, static_cast<uint8_t>(r8 / 8), 0, 1});
                PipeBarrier<PIPE_V>();
            }
        }
        bool valid = true; float correctedTotal = 0.0f;
        for (uint32_t row = 0; row < owned; ++row) {
            Add(tileHigh[row * r8], cache[row * r8], tileLow[row * r8], r8); PipeBarrier<PIPE_V>();
            float cn = Root(Dot(tileLow[row * r8], tileLow[row * r8])) * 1.001f;
            float yn = Root(Dot(cache[row * r8], cache[row * r8])) * 1.001f;
            correctedTotal += Dot(tileHigh[row * r8], tileHigh[row * r8]);
            float ct = cn + gamma * yn * rf + yn * er;
            Muls(x, gather, gamma * yn + ct / denominator, r8); Muls(y, output, yn + ct / denominator, r8); PipeBarrier<PIPE_V>(); Add(x, x, y, r8); PipeBarrier<PIPE_V>();
            Abs(y, cache[row * r8], r8); Abs(prod, tileLow[row * r8], r8); PipeBarrier<PIPE_V>(); Add(y, y, prod, r8); PipeBarrier<PIPE_V>(); Muls(y, y, unit, r8); PipeBarrier<PIPE_V>(); Add(x, x, y, r8); PipeBarrier<PIPE_V>(); Muls(x, x, 1.002f, r8); PipeBarrier<PIPE_V>();
            Abs(y, tileHigh[row * r8], r8); PipeBarrier<PIPE_V>(); Muls(y, y, 1.0e-4f, r8); PipeBarrier<PIPE_V>(); Adds(y, y, 1.0e-4f * power, r8); PipeBarrier<PIPE_V>(); Sub(x, x, y, r8); PipeBarrier<PIPE_V>();
            ReduceMax(red, x, tmp, r8, false); VS(); if (!(red.GetValue(0) <= 0.0f)) valid = false;
        }
        red.SetValue(0, correctedTotal); red.SetValue(1, valid ? 1.0f : 0.0f); Store(meta[core * 8], red, 2); SyncAll();
        Load(gather, meta, cores * 8); correctedTotal = 0.0f;
        for (uint32_t peer = 0; peer < cores; ++peer) {
            correctedTotal += gather.GetValue(peer * 8);
            if (gather.GetValue(peer * 8 + 1) < 0.5f) valid = false;
        }
        float scaledCutoff = cutoff / power;
        if (!(valid && correctedTotal > 0.0f && (!certify || correctedTotal * scaledCutoff * scaledCutoff < 81.0f))) return false;
        for (uint32_t row = 0; row < owned; ++row) {
            Muls(output, tileHigh[row * r8], 1.0f / power, r8); PipeBarrier<PIPE_V>();
            Store(result[offset + (core + row * cores) * t.m], output, t.m);
        }
        SyncAll(); return true;
    }
    __aicore__ inline bool QuickResidentSquare(uint32_t b, uint32_t offset, float scale,
                                               GlobalTensor<float> meta, bool certify, float cutoff) {
        if (!(scale >= 0.1f)) return false;
        uint32_t core = GetBlockIdx(), cores = GetBlockNum();
        uint32_t owned = (t.cols + cores - 1 - core) / cores;
        auto bx = x, by = y, bu = u, bv = v;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        Duplicate(tileHigh, 0.0f, owned * stride); Duplicate(norms, 0.0f, k8); PipeBarrier<PIPE_V>();
        float total = 0.0f;
        for (uint32_t row = 0; row < owned; ++row) {
            uint32_t i = core + row * cores;
            Load(tileHigh[row * stride], input[offset + i * t.n], t.n);
            Muls(tileHigh[row * stride], tileHigh[row * stride], 1.0f / power, r8); PipeBarrier<PIPE_V>();
            total += Dot(tileHigh[row * stride], tileHigh[row * stride]); VS();
            tileHigh.SetValue(row * stride + r8 + i, 1.0f); SV();
        }
        red.SetValue(0, total); Store(meta[core * 8], red, 1); SyncAll();
        Load(gather, meta, cores * 8); total = 0.0f;
        for (uint32_t i = 0; i < cores; ++i) total += gather.GetValue(i * 8);
        bool vectorPivot = owned <= 64;
        uint64_t available = owned < 64 ? (uint64_t(1) << owned) - 1 : ~uint64_t(0);
        auto pivotIndices = tmp.ReinterpretCast<uint32_t>();
        if (vectorPivot) {
            for (uint32_t row = 0; row < (owned + 7) / 8 * 8; ++row) pivotIndices.SetValue(row, row < owned ? row * stride * sizeof(float) : 0);
            SV();
        }
        for (uint32_t p = 0; p < t.cols; ++p) {
            if (p == 0) {
                VS(); float best = 0.0f; uint32_t bestRow = core;
                if (vectorPivot) {
                    Gather(prod, tileHigh, pivotIndices, 0, owned); PipeBarrier<PIPE_V>();
                    Abs(prod, prod, owned); PipeBarrier<PIPE_V>();
                    uint64_t active[2] = {available, 0};
                    WholeReduceMax(red, prod, active, 1, 1, 1, 8); VS();
                    best = red.GetValue(0); bestRow = core + red.ReinterpretCast<uint32_t>().GetValue(1) * cores;
                } else {
                    for (uint32_t row = 0; row < owned; ++row) {
                        float value = AbsScalar(tileHigh.GetValue(row * stride));
                        if (value > best) { best = value; bestRow = core + row * cores; }
                    }
                }
                red.SetValue(0, best); red.SetValue(1, static_cast<float>(static_cast<int32_t>(bestRow))); Store(meta[core * 8], red, 2); SyncAll();
            }
            Load(gather, meta, cores * 8); float best = 0.0f; uint32_t pivot = 0;
            GlobalPivot(cores, best, pivot);
            if (!(best > 0.0f)) return false;
            norms.SetValue(pivot, static_cast<float>(static_cast<int32_t>(p + 1))); SV();
            if (vectorPivot && core == pivot % cores) available &= ~(uint64_t(1) << (pivot / cores));
            if (core == pivot % cores) {
                uint32_t row = pivot / cores; VS();
                float divisor = tileHigh.GetValue(row * stride + p);
                Muls(tileHigh[row * stride], tileHigh[row * stride], 1.0f / divisor, s); PipeBarrier<PIPE_V>();
                Store(work, tileHigh[row * stride], rowLength);
            }
            SyncAll(); Load(u, work, rowLength); VS();
            float nextBest = 0.0f; uint32_t nextRow = core;
            if (owned <= 255) {
                for (uint32_t row = 0; row < owned; ++row) {
                    uint32_t i = core + row * cores;
                    float factor = i == pivot ? 0.0f : -tileHigh.GetValue(row * stride + p);
                    Duplicate(tileLow[row * stride], factor, 8);
                }
                PipeBarrier<PIPE_V>();
                for (uint32_t col = 0; col < s; col += 64) {
                    uint32_t mask = s - col < 64 ? s - col : 64;
                    MulAddDst(tileHigh[col], u[col], tileLow, static_cast<uint64_t>(mask), static_cast<uint8_t>(owned), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, static_cast<uint8_t>(stride / 8)});
                    PipeBarrier<PIPE_V>();
                }
                if (!vectorPivot) {
                    VS();
                    if (p + 1 < t.cols) for (uint32_t row = 0; row < owned; ++row) {
                        uint32_t i = core + row * cores;
                        float value = norms.GetValue(i) == 0.0f ? AbsScalar(tileHigh.GetValue(row * stride + p + 1)) : 0.0f;
                        if (value > nextBest) { nextBest = value; nextRow = i; }
                    }
                }
            } else {
                for (uint32_t row = 0; row < t.inverseTile; ++row) {
                    Duplicate(batchU[row * stride], 0.0f, stride); PipeBarrier<PIPE_V>();
                    DataCopy(batchU[row * stride], u, rowLength);
                }
                PipeBarrier<PIPE_V>();
                for (uint32_t first = 0; first < owned; first += t.inverseTile) {
                    uint32_t rows = owned - first < t.inverseTile ? owned - first : t.inverseTile;
                    for (uint32_t row = 0; row < rows; ++row) {
                        uint32_t i = core + (first + row) * cores;
                        float factor = i == pivot ? 0.0f : -tileHigh.GetValue((first + row) * stride + p);
                        Duplicate(batchH[row * stride], factor, stride);
                    }
                    PipeBarrier<PIPE_V>();
                    MulAddDst(tileHigh[first * stride], batchU, batchH, rows * stride); PipeBarrier<PIPE_V>(); VS();
                    if (!vectorPivot && p + 1 < t.cols) for (uint32_t row = 0; row < rows; ++row) {
                        uint32_t i = core + (first + row) * cores;
                        float value = norms.GetValue(i) == 0.0f ? AbsScalar(tileHigh.GetValue((first + row) * stride + p + 1)) : 0.0f;
                        if (value > nextBest) { nextBest = value; nextRow = i; }
                    }
                }
            }
            if (p + 1 < t.cols && vectorPivot && available) {
                Gather(prod, tileHigh, pivotIndices, (p + 1) * sizeof(float), owned); PipeBarrier<PIPE_V>();
                Abs(prod, prod, owned); PipeBarrier<PIPE_V>();
                uint64_t active[2] = {available, 0};
                WholeReduceMax(red, prod, active, 1, 1, 1, 8); VS();
                nextBest = red.GetValue(0); nextRow = core + red.ReinterpretCast<uint32_t>().GetValue(1) * cores;
            }
            if (p + 1 < t.cols) { red.SetValue(0, nextBest); red.SetValue(1, static_cast<float>(static_cast<int32_t>(nextRow))); Store(meta[core * 8], red, 2); }
            SyncAll();
        }
        float inverseTotal = 0.0f;
        for (uint32_t row = 0; row < owned; ++row) inverseTotal += Dot(tileHigh[row * stride + r8], tileHigh[row * stride + r8]);
        red.SetValue(0, inverseTotal); Store(meta[core * 8], red, 1); SyncAll();
        Load(gather, meta, cores * 8); inverseTotal = 0.0f;
        for (uint32_t i = 0; i < cores; ++i) inverseTotal += gather.GetValue(i * 8);
        float scaledCutoff = cutoff / power;
        if (!(inverseTotal > 0.0f)) return false;
        if (certify && !(inverseTotal * scaledCutoff * scaledCutoff < 81.0f)) return false;
        if (!(total * inverseTotal < 3.6e9f)) {
            if (total * inverseTotal < 4.0e11f && t.resident * r8 <= 8 * s &&
                owned <= 255 && t.inverseTile * stride >= owned * 8 && 2 * cores * r8 <= t.cols * stride)
                return RefineResident(b, offset, power, total, inverseTotal, meta, certify, cutoff);
            return false;
        }
        for (uint32_t row = 0; row < owned; ++row) {
            VS(); uint32_t target = static_cast<uint32_t>(static_cast<int32_t>(norms.GetValue(core + row * cores))) - 1;
            Muls(output, tileHigh[row * stride + r8], 1.0f / power, r8); PipeBarrier<PIPE_V>();
            Store(result[offset + target * t.m], output, t.m);
        }
        x = bx; y = by; u = bu; v = bv;
        SyncAll(); return true;
    }
    __aicore__ inline bool ResidentSquare(uint32_t b, uint32_t offset, float scale,
                                          GlobalTensor<float> meta, bool certify, float cutoff, bool skipQuick = false) {
        if constexpr (WAVE) if ((t.cols + GetBlockNum() - 1) / GetBlockNum() * stride > lowElements) return false;
        if (!skipQuick && QuickResidentSquare(b, offset, scale, meta, certify, cutoff)) return true;
        uint32_t core = GetBlockIdx(), cores = GetBlockNum();
        uint32_t owned = (t.cols + cores - 1 - core) / cores;
        GlobalTensor<float> low;
        low.SetGlobalBuffer(workBase + uint64_t(t.batch + b) * t.cols * stride);
        auto bx = x, by = y, bu = u, bv = v;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        Duplicate(tileHigh, 0.0f, owned * stride); Duplicate(tileLow, 0.0f, owned * stride);
        Duplicate(norms, 0.0f, k8); PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < owned; ++row) {
            uint32_t i = core + row * cores;
            Load(tileHigh[row * stride], input[offset + i * t.n], t.n);
            Muls(tileHigh[row * stride], tileHigh[row * stride], 1.0f / power, r8); PipeBarrier<PIPE_V>(); VS();
            tileHigh.SetValue(row * stride + r8 + i, 1.0f); SV();
        }
        for (uint32_t p = 0; p < t.cols; ++p) {
            if (p == 0) {
                VS(); float best = 0.0f; uint32_t bestRow = core;
                for (uint32_t row = 0; row < owned; ++row) {
                    float value = AbsScalar(tileHigh.GetValue(row * stride));
                    if (value > best) { best = value; bestRow = core + row * cores; }
                }
                red.SetValue(0, best); red.SetValue(1, static_cast<float>(static_cast<int32_t>(bestRow))); Store(meta[core * 8], red, 2); SyncAll();
            }
            Load(gather, meta, cores * 8); float best = 0.0f; uint32_t pivot = 0;
            GlobalPivot(cores, best, pivot);
            if (!(best > 0.0f)) { x = bx; y = by; u = bu; v = bv; return false; }
            norms.SetValue(pivot, static_cast<float>(static_cast<int32_t>(p + 1))); SV();
            if (core == pivot % cores) {
                uint32_t row = pivot / cores;
                x = tileHigh[row * stride]; y = tileLow[row * stride]; u = bu; v = bv; VS();
                Wide divisor = {x.GetValue(p), y.GetValue(p)}; SV(); VectorNormalize(divisor);
                Store(work, x, rowLength); Store(low, y, rowLength);
                x = bx; y = by; u = bu; v = bv;
            }
            SyncAll(); Load(u, work, rowLength); Load(v, low, rowLength);
            for (uint32_t row = 0; row < t.inverseTile; ++row) {
                Duplicate(batchU[row * stride], 0.0f, stride); Duplicate(batchV[row * stride], 0.0f, stride); PipeBarrier<PIPE_V>();
                DataCopy(batchU[row * stride], u, rowLength); DataCopy(batchV[row * stride], v, rowLength);
            }
            PipeBarrier<PIPE_V>();
            uint32_t cachedCount = t.inverseTile * stride;
            Muls(batchCalc[0], batchU, 4097.0f, cachedCount); PipeBarrier<PIPE_V>();
            Sub(batchCalc[1], batchCalc[0], batchU, cachedCount); PipeBarrier<PIPE_V>();
            Sub(batchCalc[0], batchCalc[0], batchCalc[1], cachedCount); PipeBarrier<PIPE_V>();
            Sub(batchCalc[1], batchU, batchCalc[0], cachedCount); PipeBarrier<PIPE_V>(); VS();
            float nextBest = 0.0f; uint32_t nextRow = core;
            for (uint32_t first = 0; first < owned; first += t.inverseTile) {
                uint32_t rows = owned - first < t.inverseTile ? owned - first : t.inverseTile;
                for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t i = core + (first + row) * cores;
                    float fh = i == pivot ? 0.0f : tileHigh.GetValue((first + row) * stride + p);
                    float fl = i == pivot ? 0.0f : tileLow.GetValue((first + row) * stride + p);
                    Duplicate(batchH[row * stride], fh, stride); Duplicate(batchL[row * stride], fl, stride);
                }
                PipeBarrier<PIPE_V>(); x = tileHigh[first * stride]; y = tileLow[first * stride];
                BatchEliminate(rows * stride, true); VS();
                if (p + 1 < t.cols) for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t i = core + (first + row) * cores;
                    float value = norms.GetValue(i) == 0.0f ? AbsScalar(tileHigh.GetValue((first + row) * stride + p + 1)) : 0.0f;
                    if (value > nextBest) { nextBest = value; nextRow = i; }
                }
            }
            x = bx; y = by;
            if (p + 1 < t.cols) { red.SetValue(0, nextBest); red.SetValue(1, static_cast<float>(static_cast<int32_t>(nextRow))); Store(meta[core * 8], red, 2); }
            SyncAll();
        }
        if (certify) {
            float inverseTotal = 0.0f;
            for (uint32_t row = 0; row < owned; ++row) inverseTotal += Dot(tileHigh[row * stride + r8], tileHigh[row * stride + r8]);
            red.SetValue(0, inverseTotal); Store(meta[core * 8], red, 1); SyncAll();
            Load(gather, meta, cores * 8); inverseTotal = 0.0f;
            for (uint32_t i = 0; i < cores; ++i) inverseTotal += gather.GetValue(i * 8);
            float scaledCutoff = cutoff / power;
            if (!(inverseTotal > 0.0f && inverseTotal * scaledCutoff * scaledCutoff < 81.0f)) return false;
        }
        for (uint32_t row = 0; row < owned; ++row) {
            VS(); uint32_t target = static_cast<uint32_t>(static_cast<int32_t>(norms.GetValue(core + row * cores))) - 1;
            Add(output, tileHigh[row * stride + r8], tileLow[row * stride + r8], r8); PipeBarrier<PIPE_V>();
            Muls(output, output, 1.0f / power, r8); PipeBarrier<PIPE_V>();
            Store(result[offset + target * t.m], output, t.m);
        }
        SyncAll(); return true;
    }
    __aicore__ inline bool AccurateSquare(uint32_t b, uint32_t offset, float scale, GlobalTensor<float> meta,
                                         bool certify = false, float cutoff = 0.0f, bool skipQuick = false) {
        if (t.resident) {
            if constexpr (WAVE) {
                if ((t.cols + GetBlockNum() - 1) / GetBlockNum() * stride <= lowElements)
                    return ResidentSquare(b, offset, scale, meta, certify, cutoff, skipQuick);
            } else return ResidentSquare(b, offset, scale, meta, certify, cutoff, skipQuick);
        }
        uint32_t core = GetBlockIdx(), cores = GetBlockNum();
        GlobalTensor<float> low;
        low.SetGlobalBuffer(workBase + uint64_t(t.batch + b) * t.cols * stride);
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        for (uint32_t i = core; i < t.cols; i += cores) {
            Duplicate(x, 0.0f, rowLength); Duplicate(y, 0.0f, rowLength); PipeBarrier<PIPE_V>();
            Load(x, input[offset + i * t.n], t.n);
            Muls(x, x, 1.0f / power, r8); PipeBarrier<PIPE_V>(); VS();
            x.SetValue(r8 + i, 1.0f); SV(); SaveRow(x, i); Store(low[i * stride], y, rowLength);
        }
        SyncAll();
        for (uint32_t p = 0; p < t.cols; ++p) {
            float best = 0; uint32_t bestRow = p;
            if (p == 0) {
                for (uint32_t i = core; i < t.cols; i += cores) {
                    ReadRow(x, i); VS(); float value = AbsScalar(x.GetValue(p));
                    if (value > best) { best = value; bestRow = i; }
                }
                red.SetValue(0, best); Store(meta[core * 8], red, 1);
                red.SetValue(0, static_cast<float>(static_cast<int32_t>(bestRow))); Store(meta[256 + core * 8], red, 1);
                SyncAll();
            }
            Load(gather, meta, cores * 8); best = 0; uint32_t bestCore = 0;
            for (uint32_t i = 0; i < cores; ++i) { float value = gather.GetValue(i * 8); if (value > best) { best = value; bestCore = i; } }
            if (certify && !(best > 0.0f)) return false;
            Load(gather, meta[256], cores * 8);
            uint32_t pivot = static_cast<uint32_t>(static_cast<int32_t>(gather.GetValue(bestCore * 8)));
            if (core == 0) {
                if (pivot != p) {
                    ReadRow(x, pivot); ReadRow(y, p); Load(u, low[pivot * stride], rowLength); Load(v, low[p * stride], rowLength);
                    SaveRow(x, p); SaveRow(y, pivot); Store(low[p * stride], u, rowLength); Store(low[pivot * stride], v, rowLength);
                }
                ReadRow(x, p); Load(y, low[p * stride], rowLength); VS();
                Wide divisor = {x.GetValue(p), y.GetValue(p)};
                SV(); VectorNormalize(divisor);
                SaveRow(x, p); Store(low[p * stride], y, rowLength);
            }
            SyncAll(); ReadRow(u, p); Load(v, low[p * stride], rowLength); VS();
            float nextBest = 0.0f; uint32_t nextRow = p + 1;
            auto bx = x, by = y;
            for (uint32_t row = 0; row < t.inverseTile; ++row) {
                Duplicate(batchU[row * stride], 0.0f, stride); Duplicate(batchV[row * stride], 0.0f, stride); PipeBarrier<PIPE_V>();
                DataCopy(batchU[row * stride], u, rowLength); DataCopy(batchV[row * stride], v, rowLength);
            }
            PipeBarrier<PIPE_V>();
            uint32_t owned = (t.cols + cores - 1 - core) / cores;
            for (uint32_t first = 0; first < owned; first += t.inverseTile) {
                uint32_t rows = owned - first < t.inverseTile ? owned - first : t.inverseTile;
                uint32_t index = core + first * cores;
                Duplicate(tileHigh, 0.0f, rows * stride); Duplicate(tileLow, 0.0f, rows * stride); PipeBarrier<PIPE_V>();
                DataCopyExtParams inParams{static_cast<uint16_t>(rows), rowLength * 4, (cores * stride - rowLength) * 4, (stride - rowLength) / 8, 0};
                SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
                DataCopyPad(tileHigh, work[index * stride], inParams, DataCopyPadExtParams<float>{false, 0, 0, 0});
                DataCopyPad(tileLow, low[index * stride], inParams, DataCopyPadExtParams<float>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
                SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
                for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t i = index + row * cores;
                    float fh = i == p ? 0.0f : tileHigh.GetValue(row * stride + p);
                    float fl = i == p ? 0.0f : tileLow.GetValue(row * stride + p);
                    Duplicate(batchH[row * stride], fh, stride); Duplicate(batchL[row * stride], fl, stride);
                }
                PipeBarrier<PIPE_V>(); x = tileHigh; y = tileLow;
                BatchEliminate(rows * stride); VS();
                for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t i = index + row * cores;
                    if (i > p && p + 1 < t.cols) {
                        float value = AbsScalar(tileHigh.GetValue(row * stride + p + 1));
                        if (value > nextBest) { nextBest = value; nextRow = i; }
                    }
                }
                DataCopyExtParams outParams{static_cast<uint16_t>(rows), rowLength * 4, (stride - rowLength) / 8, (cores * stride - rowLength) * 4, 0};
                SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
                SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
                DataCopyPad(work[index * stride], tileHigh, outParams); DataCopyPad(low[index * stride], tileLow, outParams);
                SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
                SetFlag<HardEvent::MTE3_MTE2>(0); WaitFlag<HardEvent::MTE3_MTE2>(0);
            }
            x = bx; y = by;
            if (p + 1 < t.cols) {
                red.SetValue(0, nextBest); Store(meta[core * 8], red, 1);
                red.SetValue(0, static_cast<float>(static_cast<int32_t>(nextRow))); Store(meta[256 + core * 8], red, 1);
            }
            SyncAll();
        }
        if (certify) {
            float inverseTotal = 0.0f;
            for (uint32_t row = core; row < t.n; row += cores) {
                ReadRow(x, row);
                inverseTotal += Dot(x[r8], x[r8]);
            }
            red.SetValue(0, inverseTotal); Store(meta[core * 8], red, 1); SyncAll();
            Load(gather, meta, cores * 8); inverseTotal = 0.0f;
            for (uint32_t i = 0; i < cores; ++i) inverseTotal += gather.GetValue(i * 8);
            float scaledCutoff = cutoff / power;
            // The promised spectral gap bounds a discarded nonzero value by 0.1 times the threshold.
            // Such a value forces the inverse norm times this upper threshold to be at least ten.
            if (!(inverseTotal > 0.0f && inverseTotal * scaledCutoff * scaledCutoff < 81.0f)) return false;
        }
        for (uint32_t row = core; row < t.n; row += cores) {
            ReadRow(x, row); Load(y, low[row * stride], rowLength); VS();
            for (uint32_t j = 0; j < t.m; ++j) output.SetValue(j, (x.GetValue(r8 + j) + y.GetValue(r8 + j)) / power);
            Store(result[offset + row * t.m], output, t.m);
        }
        SyncAll();
        return true;
    }
    __aicore__ inline void RestoreJacobi(uint32_t offset, float scale) {
        for (uint32_t i = GetBlockIdx(); i < t.cols; i += GetBlockNum()) {
            Duplicate(x, 0.0f, rowLength); PipeBarrier<PIPE_V>();
            DataCopyPad(gather, input[offset + i], DataCopyExtParams{static_cast<uint16_t>(t.m), 4, (t.n - 1) * 4, 0, 0}, DataCopyPadExtParams<float>{true, 0, 7, 0});
            SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0); VS();
            for (uint32_t j = 0; j < t.rows; ++j) x.SetValue(j, gather.GetValue(j * 8));
            x.SetValue(r8 + i, 1.0f); SV();
            Muls(x, x, 1.0f / scale, r8); PipeBarrier<PIPE_V>(); SaveRow(x, i);
        }
        SyncAll();
    }
    __aicore__ inline void BatchEliminate(uint32_t count, bool cached = false) {
        auto qh = batchCalc[0], ql = batchCalc[1], bh = batchCalc[2], bl = batchCalc[3], e = batchCalc[4], part = batchCalc[5], product = batchCalc[6], aux = batchCalc[7];
        Muls(bh, batchH, 4097.0f, count); PipeBarrier<PIPE_V>();
        Sub(bl, bh, batchH, count); PipeBarrier<PIPE_V>(); Sub(bh, bh, bl, count); PipeBarrier<PIPE_V>();
        Sub(bl, batchH, bh, count);
        if (!cached) {
            Muls(qh, batchU, 4097.0f, count); PipeBarrier<PIPE_V>();
            Sub(ql, qh, batchU, count); PipeBarrier<PIPE_V>(); Sub(qh, qh, ql, count); PipeBarrier<PIPE_V>();
            Sub(ql, batchU, qh, count);
        }
        Mul(product, batchU, batchH, count); PipeBarrier<PIPE_V>();
        Mul(e, qh, bh, count); PipeBarrier<PIPE_V>(); Sub(e, e, product, count); PipeBarrier<PIPE_V>();
        Mul(part, ql, bh, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Mul(part, qh, bl, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Mul(part, ql, bl, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Mul(part, batchV, batchH, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Mul(part, batchU, batchL, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        auto ph = cached ? aux : qh, pl = cached ? product : ql;
        Add(ph, product, e, count); PipeBarrier<PIPE_V>(); Sub(part, ph, product, count); PipeBarrier<PIPE_V>(); Sub(pl, e, part, count); PipeBarrier<PIPE_V>();
        Sub(bh, x, ph, count); PipeBarrier<PIPE_V>(); Sub(bl, bh, x, count); PipeBarrier<PIPE_V>();
        Sub(e, bh, bl, count); PipeBarrier<PIPE_V>(); Sub(e, x, e, count); Muls(part, ph, -1.0f, count); PipeBarrier<PIPE_V>();
        Sub(part, part, bl, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Add(e, e, y, count); PipeBarrier<PIPE_V>(); Sub(e, e, pl, count); PipeBarrier<PIPE_V>();
        Add(x, bh, e, count); PipeBarrier<PIPE_V>(); Sub(aux, x, bh, count); PipeBarrier<PIPE_V>(); Sub(y, e, aux, count); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool LocalInverseBody(uint32_t offset, LocalTensor<float> bx, LocalTensor<float> by,
                                            LocalTensor<float> bu, LocalTensor<float> bv) {
        float scale = 0.0f;
        for (uint32_t i = 0; i < t.cols; ++i) {
            x = Row(i); y = lowMatrix[i * stride];
            Duplicate(x, 0.0f, rowLength); Duplicate(y, 0.0f, rowLength); PipeBarrier<PIPE_V>();
            Load(x, input[offset + i * t.n], t.n);
            Abs(prod, x, r8); PipeBarrier<PIPE_V>(); ReduceMax(red, prod, tmp, r8, false); VS();
            float mx = red.GetValue(0); scale = mx > scale ? mx : scale;
            x.SetValue(r8 + i, 1.0f); SV();
        }
        if (!(scale > 0.0f)) return false;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        float total = 0.0f;
        for (uint32_t i = 0; i < t.cols; ++i) {
            x = Row(i); Muls(x, x, 1.0f / power, r8); PipeBarrier<PIPE_V>(); total += Dot(x, x);
        }
        for (uint32_t p = 0; p < t.cols; ++p) {
            VS(); uint32_t pivot = p; float best = 0.0f;
            for (uint32_t i = p; i < t.cols; ++i) {
                float value = AbsScalar(Row(i).GetValue(p));
                if (value > best) { best = value; pivot = i; }
            }
            if (!(best > 0.0f)) return false;
            if (pivot != p) {
                DataCopy(bx, Row(p), rowLength); DataCopy(by, lowMatrix[p * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(p), Row(pivot), rowLength); DataCopy(lowMatrix[p * stride], lowMatrix[pivot * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(pivot), bx, rowLength); DataCopy(lowMatrix[pivot * stride], by, rowLength); PipeBarrier<PIPE_V>();
            }
            x = Row(p); y = lowMatrix[p * stride]; u = bu; v = bv; VS();
            Wide divisor = {x.GetValue(p), y.GetValue(p)}; SV(); VectorNormalize(divisor);
            VS(); u = x; v = y;
            for (uint32_t first = 0; first < t.cols; first += t.inverseTile) {
                uint32_t rows = t.cols - first < t.inverseTile ? t.cols - first : t.inverseTile;
                for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t i = first + row;
                    float fh = i == p ? 0.0f : Row(i).GetValue(p);
                    float fl = i == p ? 0.0f : lowMatrix[i * stride].GetValue(p);
                    Duplicate(batchH[row * stride], fh, stride); Duplicate(batchL[row * stride], fl, stride);
                    DataCopy(batchU[row * stride], u, stride); DataCopy(batchV[row * stride], v, stride);
                }
                PipeBarrier<PIPE_V>();
                x = Row(first); y = lowMatrix[first * stride];
                BatchEliminate(rows * stride); VS();
            }
        }
        float inverseTotal = 0.0f;
        for (uint32_t i = 0; i < t.cols; ++i) { x = Row(i); inverseTotal += Dot(x[r8], x[r8]); }
        float cutoff = Root(total) * t.rtol;
        float absolute = t.atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        if (!(inverseTotal > 0.0f && inverseTotal * cutoff * cutoff < 0.81f)) return false;
        for (uint32_t i = 0; i < t.n; ++i) {
            x = Row(i); y = lowMatrix[i * stride];
            Add(output, x[r8], y[r8], r8); PipeBarrier<PIPE_V>(); Muls(output, output, 1.0f / power, r8); PipeBarrier<PIPE_V>();
            Store(result[offset + i * t.m], output, t.m);
        }
        return true;
    }
    __aicore__ inline bool QuickLocalInverseLegacy(uint32_t offset) {
        if (t.cols <= 32 || t.cols > 96) return false;
        uint32_t count = t.cols * stride;
        Duplicate(matrix, 0.0f, count); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(matrix, input[offset], DataCopyExtParams{static_cast<uint16_t>(t.cols), t.n * 4, 0, (stride - r8) / 8, 0}, DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(r8 - t.n), 0});
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        Abs(lowMatrix, matrix, count); PipeBarrier<PIPE_V>(); ReduceMax(red, lowMatrix, tmp, count, false); VS();
        float scale = red.GetValue(0);
        if (!(scale >= 0.1f)) return false;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
        Mul(lowMatrix, matrix, matrix, count); PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS();
        float total = red.GetValue(0);
        auto data = reinterpret_cast<__ubuf__ float *>(matrix.GetPhyAddr());
        for (uint32_t i = 0; i < t.cols; ++i) data[i * stride + r8 + i] = 1.0f;
        auto indices = gather.ReinterpretCast<uint32_t>();
        for (uint32_t i = 0; i < k8; ++i) indices.SetValue(i, i < t.cols ? i * stride * sizeof(float) : 0);
        Duplicate(norms, 0.0f, k8); PipeBarrier<PIPE_V>(); SV();
        for (uint32_t p = 0; p < t.cols; ++p) {
            Gather(norms, matrix, indices, p * sizeof(float), t.cols); PipeBarrier<PIPE_V>();
            Abs(prod, norms, k8); PipeBarrier<PIPE_V>();
            if (t.cols <= 64) {
                uint64_t limit = t.cols < 64 ? (uint64_t(1) << t.cols) - 1 : ~uint64_t(0);
                uint64_t active[2] = {limit & (~uint64_t(0) << p), 0};
                WholeReduceMax(red, prod, active, 1, 1, 1, 8);
            } else {
                if (p) { Duplicate(prod, 0.0f, p); PipeBarrier<PIPE_V>(); }
                ReduceMax(red, prod, tmp, t.cols, true);
            }
            VS();
            float best = red.GetValue(0);
            uint32_t pivot = red.ReinterpretCast<uint32_t>().GetValue(1);
            if (!(best > 0.0f)) return false;
            if (pivot != p) {
                DataCopy(x, Row(p), rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(p), Row(pivot), rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(pivot), x, rowLength); PipeBarrier<PIPE_V>(); VS();
            }
            Muls(Row(p), Row(p), 1.0f / data[p * stride + p], s); PipeBarrier<PIPE_V>();
            DataCopy(u, Row(p), rowLength); PipeBarrier<PIPE_V>();
            if (pivot != p) norms.SetValue(pivot, norms.GetValue(p));
            norms.SetValue(p, 0.0f); SV();
            Muls(norms, norms, -1.0f, k8); PipeBarrier<PIPE_V>();
            Brcb(lowMatrix, norms, k8 / 8, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)});
            PipeBarrier<PIPE_V>();
            for (uint32_t col = 0; col < s; col += 64) {
                uint32_t mask = s - col < 64 ? s - col : 64;
                MulAddDst(matrix[col], u[col], lowMatrix, static_cast<uint64_t>(mask), static_cast<uint8_t>(t.cols), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, static_cast<uint8_t>(stride / 8)});
            }
            PipeBarrier<PIPE_V>();
        }
        Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
        for (uint32_t col = 0; col < r8; col += 64) {
            uint32_t mask = r8 - col < 64 ? r8 - col : 64;
            Mul(lowMatrix[col], matrix[r8 + col], matrix[r8 + col], static_cast<uint64_t>(mask), static_cast<uint8_t>(t.cols), {1, 1, 1, static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride / 8)});
        }
        PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS();
        float inverseTotal = red.GetValue(0), cutoff = Root(total) * t.rtol;
        float absolute = t.atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        if (!(inverseTotal > 0.0f && inverseTotal * cutoff * cutoff < 0.81f)) return false;
        Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        DataCopyPad(result[offset], matrix[r8], DataCopyExtParams{static_cast<uint16_t>(t.n), t.m * 4, (stride - r8) / 8, 0, 0});
        SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0); WaitFlag<HardEvent::MTE3_MTE2>(0);
        return true;
    }
    template <uint32_t PAD, bool ALIGNED, bool PACKED = true>
    __aicore__ inline bool QuickLocalInverseLogicalGeometry(uint32_t offset) {
        constexpr uint32_t r8 = PAD, k8 = PAD, s = 2 * PAD, stride = 2 * PAD + 8, rowLength = stride;
        const uint32_t n = ALIGNED ? PAD : t.cols;
        if (n <= 32 || n > 64) return false;
        uint32_t count = n * stride;
        Duplicate(matrix, 0.0f, count); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(matrix, input[offset], DataCopyExtParams{static_cast<uint16_t>(n), n * 4, 0, (stride - r8) / 8, 0}, DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(r8 - n), 0});
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        Abs(lowMatrix, matrix, count); PipeBarrier<PIPE_V>(); ReduceMax(red, lowMatrix, tmp, count, false); VS();
        float scale = red.GetValue(0);
        if (!(scale >= 0.1f)) return false;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
        Mul(lowMatrix, matrix, matrix, count); PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS();
        float total = red.GetValue(0);
        auto data = reinterpret_cast<__ubuf__ float *>(matrix.GetPhyAddr());
        for (uint32_t i = 0; i < n; ++i) data[i * stride + r8 + i] = 1.0f;
        auto indices = gather.ReinterpretCast<uint32_t>();
        auto pivotOrder = output.ReinterpretCast<uint32_t>();
        uint64_t available = n < 64 ? (uint64_t(1) << n) - 1 : ~uint64_t(0);
        for (uint32_t i = 0; i < k8; ++i) indices.SetValue(i, i < n ? i * stride * sizeof(float) : 0);
        Duplicate(norms, 0.0f, k8); PipeBarrier<PIPE_V>(); SV();
        for (uint32_t p = 0; p < n; ++p) {
            Gather(norms, matrix, indices, p * sizeof(float), n); PipeBarrier<PIPE_V>();
            Abs(prod, norms, k8); PipeBarrier<PIPE_V>();
            uint64_t active[2] = {available, 0};
            WholeReduceMax(red, prod, active, 1, 1, 1, 8);
            VS();
            float best = red.GetValue(0);
            uint32_t pivot = red.ReinterpretCast<uint32_t>().GetValue(1);
            if (!(best > 0.0f)) return false;
            available &= ~(uint64_t(1) << pivot);
            pivotOrder.SetValue(pivot, p);
            Muls(matrix[pivot * stride], matrix[pivot * stride], 1.0f / data[pivot * stride + p], s); PipeBarrier<PIPE_V>();
            DataCopy(u, matrix[pivot * stride], rowLength); PipeBarrier<PIPE_V>();
            norms.SetValue(pivot, 0.0f); SV();
            Muls(norms, norms, -1.0f, k8); PipeBarrier<PIPE_V>();
            if constexpr (PACKED) Brcb(batchH, norms, k8 / 8, {2, 16});
            else Brcb(lowMatrix, norms, k8 / 8, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)});
            PipeBarrier<PIPE_V>();
            for (uint32_t col = 0; col < s; col += 64) {
                uint32_t mask = s - col < 64 ? s - col : 64;
                MulAddDst(matrix[col], u[col], PACKED ? batchH : lowMatrix, static_cast<uint64_t>(mask), static_cast<uint8_t>(n), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, PACKED ? uint8_t(2) : static_cast<uint8_t>(stride / 8)});
            }
            PipeBarrier<PIPE_V>();
        }
        Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
        for (uint32_t col = 0; col < r8; col += 64) {
            uint32_t mask = r8 - col < 64 ? r8 - col : 64;
            Mul(lowMatrix[col], matrix[r8 + col], matrix[r8 + col], static_cast<uint64_t>(mask), static_cast<uint8_t>(n), {1, 1, 1, static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride / 8)});
        }
        PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS();
        float inverseTotal = red.GetValue(0), cutoff = Root(total) * t.rtol;
        float absolute = t.atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        if (!(inverseTotal > 0.0f && inverseTotal * cutoff * cutoff < 0.81f)) return false;
        Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < n; ++row) {
            uint32_t target = pivotOrder.GetValue(row);
            DataCopy(lowMatrix[target * stride], matrix[row * stride], stride);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        DataCopyPad(result[offset], lowMatrix[r8], DataCopyExtParams{static_cast<uint16_t>(n), n * 4, (stride - r8) / 8, 0, 0});
        SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0); WaitFlag<HardEvent::MTE3_MTE2>(0);
        return true;
    }
    template <uint32_t PAD>
    __aicore__ inline bool QuickLocalInverseLogicalCapacity(uint32_t offset) {
        if (t.cols % 8 == 0) return QuickLocalInverseLogicalGeometry<PAD, true>(offset);
        return QuickLocalInverseLogicalGeometry<PAD, false>(offset);
    }
    __aicore__ inline bool QuickLocalInverseLogical(uint32_t offset) {
        if (t.cols <= 32 || t.cols > 64) return false;
        if (t.cols <= 40) return QuickLocalInverseLogicalCapacity<40>(offset);
        if (t.cols <= 48) return QuickLocalInverseLogicalCapacity<48>(offset);
        if (t.cols <= 56) return QuickLocalInverseLogicalCapacity<56>(offset);
        return QuickLocalInverseLogicalCapacity<64>(offset);
    }
    __aicore__ inline bool QuickLocalInverse(uint32_t offset) {
        if (t.cols <= 64) return QuickLocalInverseLogical(offset);
        return QuickLocalInverseLegacy(offset);
    }
    __aicore__ inline bool LocalInverse(uint32_t offset) {
        auto bx = x, by = y, bu = u, bv = v;
        bool accepted = QuickLocalInverse(offset);
        if (!accepted) accepted = LocalInverseBody(offset, bx, by, bu, bv);
        x = bx; y = by; u = bu; v = bv;
        return accepted;
    }
    template <bool PACKETS>
    __aicore__ inline void PrepareGroupedCandidate(uint32_t core, uint32_t cores, uint32_t row, uint32_t column, float best) {
        constexpr uint32_t header = PACKETS ? 8 : 0;
        if (best > 0.0f) {
            uint32_t localRow = row / cores;
            float divisor = tileHigh.GetValue(localRow * stride + column);
            Muls(u[header], tileHigh[localRow * stride], 1.0f / divisor, s); PipeBarrier<PIPE_V>();
            if constexpr (!PACKETS) Store(work[((column & 1) * cores + core) * stride], u, rowLength);
        }
        if constexpr (PACKETS) {
            u.SetValue(0, best); u.SetValue(1, static_cast<float>(static_cast<int32_t>(row)));
            Store(work[((column & 1) * cores + core) * (rowLength + 8)], u, rowLength + 8);
        }
    }
    template <bool PACKETS>
    __aicore__ inline uint32_t GroupedResidentPath() {
        uint32_t physical = GetBlockIdx(), allCores = GetBlockNum(), groups = t.groupCount;
        uint32_t base = allCores / groups, extra = allCores % groups;
        uint32_t group, core, cores;
        if (physical < extra * (base + 1)) { group = physical / (base + 1); core = physical % (base + 1); cores = base + 1; }
        else { uint32_t rest = physical - extra * (base + 1); group = extra + rest / base; core = rest % base; cores = base; }
        uint32_t b = group, offset = b * t.m * t.n, owned = (t.cols + cores - 1 - core) / cores;
        GlobalTensor<float> meta;
        meta.SetGlobalBuffer(workBase + uint64_t(2) * t.batch * t.cols * stride + uint64_t(b) * 8192);
        work.SetGlobalBuffer(workBase + uint64_t(b) * t.cols * stride);
        Duplicate(tileHigh, 0.0f, owned * stride); Duplicate(norms, 0.0f, k8); PipeBarrier<PIPE_V>();
        float scale = 0.0f;
        for (uint32_t row = 0; row < owned; ++row) {
            Load(tileHigh[row * stride], input[offset + (core + row * cores) * t.n], t.n);
            Abs(prod, tileHigh[row * stride], r8); PipeBarrier<PIPE_V>(); ReduceMax(red, prod, tmp, r8, false); VS();
            float value = red.GetValue(0); scale = value > scale ? value : scale;
        }
        red.SetValue(0, scale); Store(meta[core * 8], red, 1); SyncAll();
        Load(gather, meta, cores * 8); scale = 0.0f;
        for (uint32_t peer = 0; peer < cores; ++peer) { float value = gather.GetValue(peer * 8); scale = value > scale ? value : scale; }
        bool valid = scale >= 0.1f && scale < 1.7014118e38f;
        float power = 1.0f;
        if (valid) { while (power < scale) power *= 2.0f; while (power * 0.5f >= scale) power *= 0.5f; }
        float total = 0.0f;
        if (valid) for (uint32_t row = 0; row < owned; ++row) {
            Muls(tileHigh[row * stride], tileHigh[row * stride], 1.0f / power, r8); PipeBarrier<PIPE_V>();
            total += Dot(tileHigh[row * stride], tileHigh[row * stride]);
            tileHigh.SetValue(row * stride + r8 + core + row * cores, 1.0f); SV();
        }
        red.SetValue(0, total); Store(meta[core * 8], red, 1); SyncAll(); Load(gather, meta, cores * 8); total = 0.0f;
        for (uint32_t peer = 0; peer < cores; ++peer) total += gather.GetValue(peer * 8);
        float cutoff = Root(total) * t.rtol;
        float absolute = t.atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        uint64_t available = owned < 64 ? (uint64_t(1) << owned) - 1 : ~uint64_t(0);
        auto pivotIndices = tmp.ReinterpretCast<uint32_t>();
        uint32_t factorRows = (owned + 7) / 8 * 8;
        bool packed = PACKETS || t.inverseTile * stride >= factorRows * 8;
        for (uint32_t row = 0; row < factorRows; ++row) pivotIndices.SetValue(row, row < owned ? row * stride * sizeof(float) : 0);
        SV(); Duplicate(u, 0.0f, rowLength + (PACKETS ? 8 : 0)); PipeBarrier<PIPE_V>();
        for (uint32_t p = 0; p < t.cols; ++p) {
            if (p == 0) {
                float best = 0.0f; uint32_t selected = core;
                if (valid) {
                    Gather(prod, tileHigh, pivotIndices, 0, owned); PipeBarrier<PIPE_V>(); Abs(prod, prod, owned); PipeBarrier<PIPE_V>();
                    uint64_t active[2] = {available, 0}; WholeReduceMax(red, prod, active, 1, 1, 1, 8); VS();
                    best = red.GetValue(0); selected = core + red.ReinterpretCast<uint32_t>().GetValue(1) * cores;
                }
                PrepareGroupedCandidate<PACKETS>(core, cores, selected, 0, best);
                if constexpr (!PACKETS) { red.SetValue(0, best); red.SetValue(1, static_cast<float>(static_cast<int32_t>(selected))); Store(meta[core * 8], red, 2); }
                SyncAll();
            }
            if constexpr (PACKETS) {
                Load(tileLow, work[(p & 1) * cores * (rowLength + 8)], cores * (rowLength + 8));
                Copy(gather, tileLow, uint64_t(8), static_cast<uint8_t>(cores), {1, 1, 1, static_cast<uint16_t>((rowLength + 8) / 8)}); VS();
            } else Load(gather, meta, cores * 8);
            float best = 0.0f; uint32_t pivot = 0;
            GlobalPivot(cores, best, pivot); valid = valid && best > 0.0f;
            if (valid) {
                norms.SetValue(pivot, static_cast<float>(static_cast<int32_t>(p + 1))); SV();
                if (core != pivot % cores) {
                    if constexpr (PACKETS) { DataCopy(u[8], tileLow[(pivot % cores) * (rowLength + 8) + 8], rowLength); PipeBarrier<PIPE_V>(); }
                    else DataCopyPad(u, work[((p & 1) * cores + pivot % cores) * stride], DataCopyExtParams{1, rowLength * 4, 0, 0, 0}, DataCopyPadExtParams<float>{true, 0, 0, 0});
                }
                if (core == pivot % cores) {
                    available &= ~(uint64_t(1) << (pivot / cores));
                    DataCopy(tileHigh[(pivot / cores) * stride], u[PACKETS ? 8 : 0], rowLength); PipeBarrier<PIPE_V>();
                }
            }
            float nextBest = 0.0f; uint32_t nextRow = core;
            if (valid) {
                if (packed) {
                    Gather(v, tileHigh, pivotIndices, p * sizeof(float), factorRows); PipeBarrier<PIPE_V>();
                    if (core == pivot % cores) { VS(); v.SetValue(pivot / cores, 0.0f); SV(); }
                    Muls(v, v, -1.0f, factorRows); PipeBarrier<PIPE_V>();
                    Brcb(batchH, v, factorRows / 8, {1, 8});
                } else {
                    VS();
                    for (uint32_t row = 0; row < owned; ++row) {
                        uint32_t i = core + row * cores;
                        float factor = i == pivot ? 0.0f : -tileHigh.GetValue(row * stride + p);
                        Duplicate(tileLow[row * stride], factor, 8);
                    }
                }
                PipeBarrier<PIPE_V>();
                auto factors = packed ? batchH : tileLow;
                uint8_t factorStride = packed ? 1 : static_cast<uint8_t>(stride / 8);
                if constexpr (!PACKETS) if (core != pivot % cores) { SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0); }
                for (uint32_t col = (p + 1) / 64 * 64; col < s; col += 64) {
                    uint32_t mask = s - col < 64 ? s - col : 64;
                    MulAddDst(tileHigh[col], u[col + (PACKETS ? 8 : 0)], factors, static_cast<uint64_t>(mask), static_cast<uint8_t>(owned), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, factorStride});
                    PipeBarrier<PIPE_V>();
                }
                if (p + 1 < t.cols && available) {
                    Gather(prod, tileHigh, pivotIndices, (p + 1) * sizeof(float), owned); PipeBarrier<PIPE_V>(); Abs(prod, prod, owned); PipeBarrier<PIPE_V>();
                    uint64_t active[2] = {available, 0}; WholeReduceMax(red, prod, active, 1, 1, 1, 8); VS();
                    nextBest = red.GetValue(0); nextRow = core + red.ReinterpretCast<uint32_t>().GetValue(1) * cores;
                }
            }
            if (p + 1 < t.cols) {
                PrepareGroupedCandidate<PACKETS>(core, cores, nextRow, p + 1, nextBest);
                if constexpr (!PACKETS) { red.SetValue(0, nextBest); red.SetValue(1, static_cast<float>(static_cast<int32_t>(nextRow))); Store(meta[core * 8], red, 2); }
            }
            SyncAll();
        }
        float inverseTotal = 0.0f;
        if (valid) for (uint32_t row = 0; row < owned; ++row) inverseTotal += Dot(tileHigh[row * stride + r8], tileHigh[row * stride + r8]);
        red.SetValue(0, inverseTotal); Store(meta[core * 8], red, 1); SyncAll(); Load(gather, meta, cores * 8); inverseTotal = 0.0f;
        for (uint32_t peer = 0; peer < cores; ++peer) inverseTotal += gather.GetValue(peer * 8);
        valid = valid && inverseTotal > 0.0f && total * inverseTotal < 3.6e9f && inverseTotal * cutoff * cutoff < 81.0f;
        if (valid) for (uint32_t row = 0; row < owned; ++row) {
            VS(); uint32_t target = static_cast<uint32_t>(static_cast<int32_t>(norms.GetValue(core + row * cores))) - 1;
            Muls(output, tileHigh[row * stride + r8], 1.0f / power, r8); PipeBarrier<PIPE_V>(); Store(result[offset + target * t.m], output, t.m);
        }
        if (core == 0) { red.SetValue(0, valid ? 1.0f : 0.0f); Store(meta[1024], red, 1); }
        SyncAll();
        uint32_t accepted = 0;
        for (uint32_t peer = 0; peer < groups; ++peer) {
            meta.SetGlobalBuffer(workBase + uint64_t(2) * t.batch * t.cols * stride + uint64_t(peer) * 8192);
            Load(gather, meta[1024], 1); if (gather.GetValue(0) > 0.5f) accepted |= uint32_t(1) << peer;
        }
        t.resident = (t.cols + allCores - 1) / allCores;
        return accepted;
    }
    template <uint32_t PANEL>
    __aicore__ inline uint32_t GroupedResidentPanels() {
        uint32_t physical = GetBlockIdx(), allCores = GetBlockNum(), groups = t.groupCount;
        uint32_t base = allCores / groups, extra = allCores % groups;
        uint32_t group, core, cores;
        if (physical < extra * (base + 1)) { group = physical / (base + 1); core = physical % (base + 1); cores = base + 1; }
        else { uint32_t rest = physical - extra * (base + 1); group = extra + rest / base; core = rest % base; cores = base; }
        uint32_t b = group, offset = b * t.m * t.n, owned = (t.cols + cores - 1 - core) / cores;
        GlobalTensor<float> meta;
        meta.SetGlobalBuffer(workBase + uint64_t(2) * t.batch * t.cols * stride + uint64_t(b) * 8192);
        work.SetGlobalBuffer(workBase + uint64_t(b) * t.cols * stride);
        Duplicate(tileHigh, 0.0f, owned * stride); Duplicate(norms, 0.0f, k8); PipeBarrier<PIPE_V>();
        float scale = 0.0f;
        for (uint32_t row = 0; row < owned; ++row) {
            Load(tileHigh[row * stride], input[offset + (core + row * cores) * t.n], t.n);
            Abs(prod, tileHigh[row * stride], r8); PipeBarrier<PIPE_V>(); ReduceMax(red, prod, tmp, r8, false); VS();
            float value = red.GetValue(0); scale = value > scale ? value : scale;
        }
        red.SetValue(0, scale); Store(meta[core * 8], red, 1); SyncAll();
        Load(gather, meta, cores * 8); scale = 0.0f;
        for (uint32_t peer = 0; peer < cores; ++peer) { float value = gather.GetValue(peer * 8); scale = value > scale ? value : scale; }
        bool valid = scale >= 0.1f && scale < 1.7014118e38f;
        float power = 1.0f;
        if (valid) { while (power < scale) power *= 2.0f; while (power * 0.5f >= scale) power *= 0.5f; }
        float total = 0.0f;
        if (valid) for (uint32_t row = 0; row < owned; ++row) {
            Muls(tileHigh[row * stride], tileHigh[row * stride], 1.0f / power, r8); PipeBarrier<PIPE_V>();
            total += Dot(tileHigh[row * stride], tileHigh[row * stride]);
            tileHigh.SetValue(row * stride + r8 + core + row * cores, 1.0f); SV();
        }
        red.SetValue(0, total); Store(meta[core * 8], red, 1); SyncAll(); Load(gather, meta, cores * 8); total = 0.0f;
        for (uint32_t peer = 0; peer < cores; ++peer) total += gather.GetValue(peer * 8);
        float cutoff = Root(total) * t.rtol;
        float absolute = t.atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        uint32_t factorRows = (owned + 7) / 8 * 8;
        auto rowIndices = tmp.ReinterpretCast<uint32_t>();
        auto compactIndices = gather.ReinterpretCast<uint32_t>();
        auto panelIndices = compactIndices[k8];
        auto compactColumnIndices = compactIndices[k8 + PANEL];
        for (uint32_t column = 0; column < PANEL; ++column) compactColumnIndices.SetValue(column, column * k8 * sizeof(float));
        for (uint32_t row = 0; row < t.cols; ++row) compactIndices.SetValue(row, row * PANEL * sizeof(float));
        Duplicate(norms, 1.0f, k8); PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < factorRows; ++row) rowIndices.SetValue(row, row < owned ? row * stride * sizeof(float) : 0);
        for (uint32_t row = 0; row < PANEL; ++row) panelIndices.SetValue(row, row * stride * sizeof(float));
        Duplicate(u, 0.0f, rowLength); PipeBarrier<PIPE_V>(); SV();
        for (uint32_t first = 0; first < t.cols; first += PANEL) {
            uint32_t rows = t.cols - first < PANEL ? t.cols - first : PANEL;
            if (valid) DataCopyPad(work[core * PANEL], tileHigh[first], DataCopyExtParams{static_cast<uint16_t>(owned), PANEL * 4, (stride - PANEL) / 8, (cores - 1) * PANEL * 4, 0});
            SyncAll();
            if (valid) {
                Load(tileLow, work, t.cols * PANEL);
                auto columns = tileLow[t.cols * PANEL];
                for (uint32_t column = 0; column < PANEL; ++column) Gather(columns[column * k8], tileLow, compactIndices, column * sizeof(float), t.cols);
                PipeBarrier<PIPE_V>();
                for (uint32_t column = 0; column < rows; ++column) {
                    DataCopy(v, columns[column * k8], k8); PipeBarrier<PIPE_V>();
                    Abs(x, v, t.cols);
                    Maxs(prod, norms, 0.0f, k8); PipeBarrier<PIPE_V>();
                    Mul(prod, prod, x, t.cols); PipeBarrier<PIPE_V>();
                    uint32_t chunks = t.cols / 64, tail = t.cols % 64;
                    WholeReduceMax(red, prod, uint64_t(64), static_cast<uint8_t>(chunks), 1, 1, 8);
                    if (tail) {
                        PipeBarrier<PIPE_V>();
                        WholeReduceMax(red[chunks * 2], prod[chunks * 64], static_cast<uint64_t>(tail), 1, 1, 1, 8);
                    }
                    VS(); float best = 0.0f; uint32_t selected = 0;
                    for (uint32_t chunk = 0; chunk < chunks + (tail != 0); ++chunk) {
                        float value = red.GetValue(chunk * 2);
                        if (value > best) { best = value; selected = chunk * 64 + red.ReinterpretCast<uint32_t>().GetValue(chunk * 2 + 1); }
                    }
                    if (!(best > 1.0e-6f)) { valid = false; break; }
                    float divisor = v.GetValue(selected);
                    norms.SetValue(selected, -static_cast<float>(static_cast<int32_t>(first + column + 1)));
                    output.SetValue(column, static_cast<float>(static_cast<int32_t>(selected)));
                    Gather(u, columns, compactColumnIndices, selected * sizeof(float), PANEL); PipeBarrier<PIPE_V>();
                    Muls(u, u, 1.0f / divisor, PANEL); PipeBarrier<PIPE_V>();
                    v.SetValue(selected, 0.0f); SV();
                    Muls(v, v, -1.0f, k8); PipeBarrier<PIPE_V>();
                    Brcb(batchH, u, (PANEL + 7) / 8, {1, 8}); PipeBarrier<PIPE_V>();
                    for (uint32_t begin = 0; begin < t.cols; begin += 64) {
                        uint32_t count = t.cols - begin < 64 ? t.cols - begin : 64;
                        MulAddDst(columns[begin], v[begin], batchH, static_cast<uint64_t>(count), static_cast<uint8_t>(PANEL), {1, 1, 0, static_cast<uint8_t>(k8 / 8), 0, 1});
                    }
                    PipeBarrier<PIPE_V>();
                }
                if (valid) for (uint32_t column = 0; column < rows; ++column) {
                    uint32_t selected = static_cast<uint32_t>(static_cast<int32_t>(output.GetValue(column)));
                    if (selected % cores == core) Store(work[t.cols * PANEL + column * stride], tileHigh[(selected / cores) * stride], rowLength);
                }
            }
            SyncAll();
            if (valid) {
                Load(tileLow, work[t.cols * PANEL], rows * stride);
                for (uint32_t column = 0; column < rows; ++column) {
                    Gather(prod, tileLow, panelIndices, (first + column) * sizeof(float), rows); PipeBarrier<PIPE_V>();
                    VS(); float divisor = prod.GetValue(column);
                    if (!(AbsScalar(divisor) > 1.0e-6f)) { valid = false; break; }
                    Muls(u, tileLow[column * stride], 1.0f / divisor, s); PipeBarrier<PIPE_V>();
                    DataCopy(tileLow[column * stride], u, rowLength); PipeBarrier<PIPE_V>();
                    prod.SetValue(column, 0.0f); SV();
                    Muls(prod, prod, -1.0f, (PANEL + 7) / 8 * 8); PipeBarrier<PIPE_V>();
                    Brcb(batchH, prod, (PANEL + 7) / 8, {1, 8}); PipeBarrier<PIPE_V>();
                    for (uint32_t col = 0; col < s; col += 64) {
                        uint32_t mask = s - col < 64 ? s - col : 64;
                        MulAddDst(tileLow[col], u[col], batchH, static_cast<uint64_t>(mask), static_cast<uint8_t>(rows), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, 1});
                    }
                    PipeBarrier<PIPE_V>();
                }
            }
            if (valid) {
                for (uint32_t column = 0; column < rows; ++column) {
                    DataCopy(u, tileLow[column * stride], rowLength);
                    Gather(v, tileHigh, rowIndices, (first + column) * sizeof(float), factorRows); PipeBarrier<PIPE_V>();
                    Muls(v, v, -1.0f, factorRows); PipeBarrier<PIPE_V>();
                    Brcb(batchH, v, factorRows / 8, {1, 8}); PipeBarrier<PIPE_V>();
                    for (uint32_t col = (first + column + 1) / 64 * 64; col < s; col += 64) {
                        uint32_t mask = s - col < 64 ? s - col : 64;
                        MulAddDst(tileHigh[col], u[col], batchH, static_cast<uint64_t>(mask), static_cast<uint8_t>(owned), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, 1});
                        PipeBarrier<PIPE_V>();
                    }
                }
                VS();
                for (uint32_t column = 0; column < rows; ++column) {
                    uint32_t selected = static_cast<uint32_t>(static_cast<int32_t>(output.GetValue(column)));
                    if (selected % cores == core) DataCopy(tileHigh[(selected / cores) * stride], tileLow[column * stride], rowLength);
                }
                PipeBarrier<PIPE_V>();
            }
        }
        float inverseTotal = 0.0f;
        if (valid) for (uint32_t row = 0; row < owned; ++row) inverseTotal += Dot(tileHigh[row * stride + r8], tileHigh[row * stride + r8]);
        red.SetValue(0, inverseTotal); Store(meta[core * 8], red, 1); SyncAll(); Load(gather, meta, cores * 8); inverseTotal = 0.0f;
        for (uint32_t peer = 0; peer < cores; ++peer) inverseTotal += gather.GetValue(peer * 8);
        valid = valid && inverseTotal > 0.0f && total * inverseTotal < 3.6e9f && inverseTotal * cutoff * cutoff < 0.81f;
        if (valid) for (uint32_t row = 0; row < owned; ++row) {
            VS(); uint32_t target = static_cast<uint32_t>(static_cast<int32_t>(-norms.GetValue(core + row * cores))) - 1;
            Muls(output, tileHigh[row * stride + r8], 1.0f / power, r8); PipeBarrier<PIPE_V>(); Store(result[offset + target * t.m], output, t.m);
        }
        if (core == 0) { red.SetValue(0, valid ? 1.0f : 0.0f); Store(meta[1024], red, 1); }
        SyncAll();
        uint32_t accepted = 0;
        for (uint32_t peer = 0; peer < groups; ++peer) {
            meta.SetGlobalBuffer(workBase + uint64_t(2) * t.batch * t.cols * stride + uint64_t(peer) * 8192);
            Load(gather, meta[1024], 1); if (gather.GetValue(0) > 0.5f) accepted |= uint32_t(1) << peer;
        }
        t.resident = (t.cols + allCores - 1) / allCores;
        return accepted;
    }
    __aicore__ inline uint32_t GroupedResident() {
        constexpr uint32_t panelRows = 8;
        uint32_t factorRows = (t.resident + 7) / 8 * 8;
        if (t.resident * stride >= panelRows * stride
            && t.resident * stride >= t.cols * panelRows + k8 * 8
            && t.cols * panelRows + panelRows * stride <= t.cols * stride
            && t.inverseTile * stride >= factorRows * 8) return GroupedResidentPanels<panelRows>();
        uint32_t cores = (GetBlockNum() + t.groupCount - 1) / t.groupCount;
        uint32_t factors = (t.resident + 7) / 8 * 8;
        bool fits = t.resident * stride >= cores * (rowLength + 8)
                    && 2 * cores * (rowLength + 8) <= t.cols * stride
                    && t.inverseTile * stride >= factors * 8;
        if (fits) return GroupedResidentPath<true>();
        return GroupedResidentPath<false>();
    }
    __aicore__ inline void RebudgetWaveScratch() {
        uint32_t active = (t.cols + GetBlockNum() - 1) / GetBlockNum();
        if (t.singleResident) {
            tileLow = tileHigh[active * stride];
            uint32_t remaining = residentElements > active * stride ? residentElements - active * stride : 0;
            lowElements = remaining < active * stride ? remaining : active * stride;
            uint32_t unused = t.resident > 2 * active ? t.resident - 2 * active : 0;
            uint32_t initial = t.inverseTile;
            uint32_t candidate = (12 * initial + unused) / 12;
            candidate = candidate < active ? candidate : active;
            while (candidate > initial && 12 * initial / candidate + unused / candidate < 12) --candidate;
            if (candidate > initial) {
                LocalTensor<float> regions[2] = {batchPrecision.Get<float>(), tileHigh[2 * active * stride]};
                uint32_t sizes[2] = {12 * initial, unused};
                LocalTensor<float> slots[12]; uint32_t count = 0;
                for (uint32_t region = 0; region < 2 && count < 12; ++region) {
                    for (uint32_t row = 0; row + candidate <= sizes[region] && count < 12; row += candidate)
                        slots[count++] = regions[region][row * stride];
                }
                for (uint32_t i = 0; i < 8; ++i) batchCalc[i] = slots[i];
                batchU = slots[8]; batchV = slots[9]; batchH = slots[10]; batchL = slots[11];
                t.inverseTile = candidate;
            }
            if (t.inverseTile * stride > lowElements) t.inverseTile = lowElements / stride;
            t.resident = active;
            return;
        }
        lowElements = active * stride;
        uint32_t unused = t.resident > active ? t.resident - active : 0;
        uint32_t initial = t.inverseTile;
        uint32_t candidate = (12 * initial + 2 * unused) / 12;
        candidate = candidate < active ? candidate : active;
        while (candidate > initial && 12 * initial / candidate + 2 * (unused / candidate) < 12) --candidate;
        if (candidate > initial) {
            LocalTensor<float> regions[3] = {batchPrecision.Get<float>(), tileHigh[active * stride], tileLow[active * stride]};
            uint32_t sizes[3] = {12 * initial, unused, unused};
            LocalTensor<float> slots[12]; uint32_t count = 0;
            for (uint32_t region = 0; region < 3 && count < 12; ++region) {
                for (uint32_t row = 0; row + candidate <= sizes[region] && count < 12; row += candidate)
                    slots[count++] = regions[region][row * stride];
            }
            for (uint32_t i = 0; i < 8; ++i) batchCalc[i] = slots[i];
            batchU = slots[8]; batchV = slots[9]; batchH = slots[10]; batchL = slots[11];
            t.inverseTile = candidate;
        }
        t.resident = active;
    }
    __aicore__ inline uint32_t WaveGroupedResident() {
        uint32_t physical = GetBlockIdx(), allCores = GetBlockNum(), groups = t.groupCount;
        uint32_t base = allCores / groups, extra = allCores % groups;
        uint32_t group, core, cores;
        if (physical < extra * (base + 1)) { group = physical / (base + 1); core = physical % (base + 1); cores = base + 1; }
        else { uint32_t rest = physical - extra * (base + 1); group = extra + rest / base; core = rest % base; cores = base; }
        uint32_t accepted = 0;
        for (uint32_t round = 0; round < t.batch / groups; ++round) {
        uint32_t b = round * groups + group, offset = b * t.m * t.n, owned = (t.cols + cores - 1 - core) / cores;
        GlobalTensor<float> meta;
        meta.SetGlobalBuffer(workBase + uint64_t(2) * t.batch * t.cols * stride + uint64_t(b) * 8192);
        work.SetGlobalBuffer(workBase + uint64_t(b) * t.cols * stride);
        Duplicate(tileHigh, 0.0f, owned * stride); Duplicate(norms, 0.0f, k8); PipeBarrier<PIPE_V>();
        float scale = 0.0f;
        for (uint32_t row = 0; row < owned; ++row) {
            Load(tileHigh[row * stride], input[offset + (core + row * cores) * t.n], t.n);
            Abs(prod, tileHigh[row * stride], r8); PipeBarrier<PIPE_V>(); ReduceMax(red, prod, tmp, r8, false); VS();
            float value = red.GetValue(0); scale = value > scale ? value : scale;
        }
        red.SetValue(0, scale); Store(meta[core * 8], red, 1); SyncAll();
        Load(gather, meta, cores * 8); scale = 0.0f;
        for (uint32_t peer = 0; peer < cores; ++peer) { float value = gather.GetValue(peer * 8); scale = value > scale ? value : scale; }
        bool valid = scale >= 0.1f && scale < 1.7014118e38f;
        float power = 1.0f;
        if (valid) { while (power < scale) power *= 2.0f; while (power * 0.5f >= scale) power *= 0.5f; }
        float total = 0.0f;
        if (valid) for (uint32_t row = 0; row < owned; ++row) {
            Muls(tileHigh[row * stride], tileHigh[row * stride], 1.0f / power, r8); PipeBarrier<PIPE_V>();
            total += Dot(tileHigh[row * stride], tileHigh[row * stride]);
            tileHigh.SetValue(row * stride + r8 + core + row * cores, 1.0f); SV();
        }
        red.SetValue(0, total); Store(meta[core * 8], red, 1); SyncAll(); Load(gather, meta, cores * 8); total = 0.0f;
        for (uint32_t peer = 0; peer < cores; ++peer) total += gather.GetValue(peer * 8);
        float cutoff = Root(total) * t.rtol;
        float absolute = t.atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        uint64_t available = owned < 64 ? (uint64_t(1) << owned) - 1 : ~uint64_t(0);
        auto pivotIndices = tmp.ReinterpretCast<uint32_t>();
        for (uint32_t row = 0; row < (owned + 7) / 8 * 8; ++row) pivotIndices.SetValue(row, row < owned ? row * stride * sizeof(float) : 0);
        SV(); Duplicate(u, 0.0f, rowLength); PipeBarrier<PIPE_V>();
        for (uint32_t p = 0; p < t.cols; ++p) {
            if (p == 0) {
                float best = 0.0f; uint32_t selected = core;
                if (valid) {
                    Gather(prod, tileHigh, pivotIndices, 0, owned); PipeBarrier<PIPE_V>(); Abs(prod, prod, owned); PipeBarrier<PIPE_V>();
                    uint64_t active[2] = {available, 0}; WholeReduceMax(red, prod, active, 1, 1, 1, 8); VS();
                    best = red.GetValue(0); selected = core + red.ReinterpretCast<uint32_t>().GetValue(1) * cores;
                }
                PrepareGroupedCandidate<false>(core, cores, selected, 0, best);
                red.SetValue(0, best); red.SetValue(1, static_cast<float>(static_cast<int32_t>(selected))); Store(meta[core * 8], red, 2); SyncAll();
            }
            Load(gather, meta, cores * 8); float best = 0.0f; uint32_t pivot = 0;
            GlobalPivot(cores, best, pivot); valid = valid && best > 0.0f;
            if (valid) {
                norms.SetValue(pivot, static_cast<float>(static_cast<int32_t>(p + 1))); SV();
                if (core != pivot % cores) {
                    DataCopyPad(u, work[((p & 1) * cores + pivot % cores) * stride], DataCopyExtParams{1, rowLength * 4, 0, 0, 0}, DataCopyPadExtParams<float>{true, 0, 0, 0});
                }
                if (core == pivot % cores) {
                    available &= ~(uint64_t(1) << (pivot / cores));
                    DataCopy(tileHigh[(pivot / cores) * stride], u, rowLength); PipeBarrier<PIPE_V>();
                }
            }
            float nextBest = 0.0f; uint32_t nextRow = core;
            if (valid) {
                uint32_t factorRows = (owned + 7) / 8 * 8;
                bool packed = t.inverseTile * stride >= factorRows * 8;
                if (packed) {
                    Gather(v, tileHigh, pivotIndices, p * sizeof(float), factorRows); PipeBarrier<PIPE_V>();
                    if (core == pivot % cores) { VS(); v.SetValue(pivot / cores, 0.0f); SV(); }
                    Muls(v, v, -1.0f, factorRows); PipeBarrier<PIPE_V>();
                    Brcb(batchH, v, factorRows / 8, {1, 8});
                } else {
                    VS();
                    for (uint32_t row = 0; row < owned; ++row) {
                        uint32_t i = core + row * cores;
                        float factor = i == pivot ? 0.0f : -tileHigh.GetValue(row * stride + p);
                        Duplicate(tileLow[row * stride], factor, 8);
                    }
                }
                PipeBarrier<PIPE_V>();
                auto factors = packed ? batchH : tileLow;
                uint8_t factorStride = packed ? 1 : static_cast<uint8_t>(stride / 8);
                if (core != pivot % cores) { SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0); }
                for (uint32_t col = 0; col < s; col += 64) {
                    uint32_t mask = s - col < 64 ? s - col : 64;
                    MulAddDst(tileHigh[col], u[col], factors, static_cast<uint64_t>(mask), static_cast<uint8_t>(owned), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, factorStride});
                    PipeBarrier<PIPE_V>();
                }
                if (p + 1 < t.cols && available) {
                    Gather(prod, tileHigh, pivotIndices, (p + 1) * sizeof(float), owned); PipeBarrier<PIPE_V>(); Abs(prod, prod, owned); PipeBarrier<PIPE_V>();
                    uint64_t active[2] = {available, 0}; WholeReduceMax(red, prod, active, 1, 1, 1, 8); VS();
                    nextBest = red.GetValue(0); nextRow = core + red.ReinterpretCast<uint32_t>().GetValue(1) * cores;
                }
            }
            if (p + 1 < t.cols) {
                PrepareGroupedCandidate<false>(core, cores, nextRow, p + 1, nextBest);
                red.SetValue(0, nextBest); red.SetValue(1, static_cast<float>(static_cast<int32_t>(nextRow))); Store(meta[core * 8], red, 2);
            }
            SyncAll();
        }
        float inverseTotal = 0.0f;
        if (valid) for (uint32_t row = 0; row < owned; ++row) inverseTotal += Dot(tileHigh[row * stride + r8], tileHigh[row * stride + r8]);
        red.SetValue(0, inverseTotal); Store(meta[core * 8], red, 1); SyncAll(); Load(gather, meta, cores * 8); inverseTotal = 0.0f;
        for (uint32_t peer = 0; peer < cores; ++peer) inverseTotal += gather.GetValue(peer * 8);
        valid = valid && inverseTotal > 0.0f && inverseTotal * cutoff * cutoff < 81.0f;
        uint32_t normOwned = (t.cols + allCores - 1) / allCores;
        bool prepared = valid && total * inverseTotal >= 3.6e9f && total * inverseTotal < 4.0e11f &&
            normOwned * r8 <= 8 * s && t.inverseTile * stride >= normOwned * 8 && 2 * allCores * r8 <= t.cols * stride;
        valid = valid && total * inverseTotal < 3.6e9f;
        if (valid || prepared) for (uint32_t row = 0; row < owned; ++row) {
            VS(); uint32_t target = static_cast<uint32_t>(static_cast<int32_t>(norms.GetValue(core + row * cores))) - 1;
            Muls(output, tileHigh[row * stride + r8], valid ? 1.0f / power : 1.0f, r8); PipeBarrier<PIPE_V>(); Store(result[offset + target * t.m], output, t.m);
        }
        if (core == 0) {
            red.SetValue(0, total); red.SetValue(1, inverseTotal); red.SetValue(2, power); Store(meta[1032], red, 3);
            red.SetValue(0, valid ? 1.0f : (prepared ? 2.0f : 0.0f)); Store(meta[1024], red, 1);
        }
        SyncAll();
        for (uint32_t peer = 0; peer < groups; ++peer) {
            uint32_t item = round * groups + peer;
            meta.SetGlobalBuffer(workBase + uint64_t(2) * t.batch * t.cols * stride + uint64_t(item) * 8192);
            Load(gather, meta[1024], 1); float state = gather.GetValue(0);
            if (state > 1.5f) accepted |= uint32_t(1) << (item + 4);
            else if (state > 0.5f) accepted |= uint32_t(1) << item;
        }
        }
        RebudgetWaveScratch();
        return accepted;
    }
    __aicore__ inline void Parallel() {
        uint32_t core = GetBlockIdx(), cores = GetBlockNum();
        uint32_t accepted = 0, prepared = 0;
        if constexpr (WAVE) {
            uint32_t grouped = WaveGroupedResident(); accepted = grouped & 15; prepared = grouped >> 4;
        } else if (t.groupCount) accepted = GroupedResident();
        uint32_t roundCols = (t.cols + 1) / 2 * 2;
        for (uint32_t b = 0; b < t.batch; ++b) {
            if (accepted & (uint32_t(1) << b)) continue;
            uint32_t offset = b * t.m * t.n;
            work.SetGlobalBuffer(workBase + uint64_t(b) * t.cols * stride);
            GlobalTensor<float> meta;
            meta.SetGlobalBuffer(workBase + uint64_t(2) * t.batch * t.cols * stride + uint64_t(b) * 8192);
            float scale = 0;
            for (uint32_t i = core; i < t.cols; i += cores) {
                Duplicate(x, 0.0f, rowLength); PipeBarrier<PIPE_V>();
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
                ReadRow(x, i); Muls(x, x, 1.0f / scale, r8); PipeBarrier<PIPE_V>();
                total += Dot(x, x); SaveRow(x, i);
            }
            red.SetValue(0, total); Store(meta[core * 8], red, 1); SyncAll();
            Load(gather, meta, cores * 8); total = 0;
            for (uint32_t i = 0; i < cores; ++i) total += gather.GetValue(i * 8);
            if (t.m == t.n) {
                float cutoff = scale * Root(total) * t.rtol;
                cutoff = cutoff > t.atol ? cutoff : t.atol;
                if (prepared & (uint32_t(1) << b)) {
                    Load(gather, meta[1032], 3);
                    float cachedTotal = gather.GetValue(0), cachedInverse = gather.GetValue(1), power = gather.GetValue(2);
                    VS(); for (uint32_t i = 0; i < t.cols; ++i) norms.SetValue(i, static_cast<float>(static_cast<int32_t>(i + 1))); SV();
                    uint32_t owned = (t.cols + cores - 1 - core) / cores;
                    for (uint32_t row = 0; row < owned; ++row) Load(tileHigh[row * stride + r8], result[offset + (core + row * cores) * t.m], t.m);
                    if (RefineResident(b, offset, power, cachedTotal, cachedInverse, meta, true, cutoff)) continue;
                    if (AccurateSquare(b, offset, scale, meta, true, cutoff, true)) continue;
                } else if (AccurateSquare(b, offset, scale, meta, true, cutoff)) continue;
                RestoreJacobi(offset, scale);
            }
            float absTol = t.atol / scale, discard = absTol * absTol;
            float rel = t.rtol * t.rtol * total;
            discard = (discard > rel ? discard : rel) * 1.0e-8f;
            float roundoffFloor = total * 1.0e-14f;
            discard = discard < roundoffFloor ? discard : roundoffFloor;
            bool cached = false;
            for (uint32_t sweep = 0; sweep < 40; ++sweep) {
                uint32_t changed = 0;
                for (uint32_t round = 0; round + 1 < roundCols; ++round) {
                    for (uint32_t slot = core; slot < roundCols / 2; slot += cores) {
                        uint32_t p = slot == 0 ? roundCols - 1 : (round + slot) % (roundCols - 1);
                        uint32_t q = slot == 0 ? round : (round + roundCols - 1 - slot) % (roundCols - 1);
                        if (p >= t.cols || q >= t.cols) continue;
                        ReadRow(x, p); ReadRow(y, q);
                        float a, d, h;
                        if (cached) { h = Dot(x, y); a = x.GetValue(s); d = y.GetValue(s); }
                        else { a = Dot(x, x); d = Dot(y, y); h = Dot(x, y); }
                        if (a <= discard || d <= discard || h == 0 || h * h <= 1.0e-12f * (a * d)) continue;
                        float tau = (d - a) / (2 * h), at = AbsScalar(tau);
                        float z = (tau < 0 ? -1.0f : 1.0f) / (at + Root(1 + at * at));
                        float c = 1.0f / Root(1 + z * z), sn = c * z;
                        Muls(u, x, c, s); Muls(v, y, -sn, s); PipeBarrier<PIPE_V>(); Add(u, u, v, s); PipeBarrier<PIPE_V>();
                        Muls(v, x, sn, s); Muls(y, y, c, s); PipeBarrier<PIPE_V>(); Add(y, y, v, s); PipeBarrier<PIPE_V>();
                        if (cached) {
                            float na = Dot(u, u), nd = Dot(y, y);
                            u.SetValue(s, na); y.SetValue(s, nd);
                        }
                        SaveRow(u, p); SaveRow(y, q); ++changed;
                    }
                    SyncAll();
                }
                red.SetValue(0, static_cast<float>(static_cast<int32_t>(changed))); Store(meta[core * 8], red, 1); SyncAll();
                Load(gather, meta, cores * 8); float sum = 0;
                for (uint32_t i = 0; i < cores; ++i) sum += gather.GetValue(i * 8);
                if (sum == 0) break;
                if (!cached && sum < 0.35f * t.cols * (t.cols - 1)) { CacheNorms(); cached = true; }
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
            float smallest = maxNorm;
            for (uint32_t i = 0; i < t.cols; ++i) {
                float a = gather.GetValue(i * 8);
                if (a > threshold && a < smallest) smallest = a;
            }
            if (t.m == t.n && t.cols >= 256 && kept == t.cols && (smallest < maxNorm * 1.0e-4f || scale * Root(smallest) < 0.02f)) { AccurateSquare(b, offset, scale, meta); continue; }
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
        t = data; r8 = (t.rows + 7) / 8 * 8; k8 = (t.cols + 7) / 8 * 8; s = r8 + k8; stride = t.stride; rowLength = s + 8;
        lowElements = 0; residentElements = 0;
        uint32_t bufferStride = t.parallel ? (rowLength + 63) / 64 * 64 : stride;
        uint32_t size = (t.local ? t.cols * stride : 0) + 4 * bufferStride + 4 * r8 + 8 * t.rows + k8 + 32;
        pipe->InitBuffer(mem, size * 4);
        auto all = mem.Get<float>(); uint32_t off = 0;
        matrix = all; off += t.local ? t.cols * stride : 0;
        x = all[off]; off += bufferStride; y = all[off]; off += bufferStride; u = all[off]; off += bufferStride; v = all[off]; off += bufferStride;
        prod = all[off]; off += r8; tmp = all[off]; off += r8; output = all[off]; off += r8;
        red = all[off]; off += 32; gather = all[off]; off += 8 * t.rows;
        norms = all[off];
        if (t.parallel || t.localInverse) {
            pipe->InitBuffer(vectorPrecision, 8 * s * sizeof(float));
            auto area = vectorPrecision.Get<float>();
            for (uint32_t i = 0; i < 8; ++i) calc[i] = area[i * s];
        }
        if (t.localInverse) {
            uint32_t lowRows = t.cols > 32 && t.cols <= 96 ? k8 : t.cols;
            pipe->InitBuffer(inverseLow, lowRows * stride * sizeof(float));
            lowMatrix = inverseLow.Get<float>();
        }
        if (t.localInverse || (t.parallel && t.m == t.n)) {
            uint32_t areaSize = t.inverseTile * stride;
            pipe->InitBuffer(batchPrecision, 12 * areaSize * sizeof(float));
            auto area = batchPrecision.Get<float>();
            for (uint32_t i = 0; i < 8; ++i) batchCalc[i] = area[i * areaSize];
            batchU = area[8 * areaSize]; batchV = area[9 * areaSize]; batchH = area[10 * areaSize]; batchL = area[11 * areaSize];
            if (t.parallel) {
                uint32_t residentSize = (t.resident ? t.resident : t.inverseTile) * stride;
                uint32_t residentBuffers = 2, lowOffset = residentSize;
                if constexpr (WAVE) if (t.singleResident) {
                    residentBuffers = 1;
                    lowOffset = (t.cols + GetBlockNum() - 1) / GetBlockNum() * stride;
                }
                residentElements = residentBuffers * residentSize;
                lowElements = residentElements - lowOffset;
                pipe->InitBuffer(parallelElimination, residentElements * sizeof(float));
                tileHigh = parallelElimination.Get<float>(); tileLow = tileHigh[lowOffset];
            }
        }
        if (t.rows <= 32) {
            pipe->InitBuffer(precision, 4 * k8 * k8 * sizeof(float));
            gh = precision.Get<float>(); gl = gh[k8 * k8]; ih = gl[k8 * k8]; il = ih[k8 * k8];
        }
        input.SetGlobalBuffer((__gm__ float *)a); result.SetGlobalBuffer((__gm__ float *)out);
        workBase = (__gm__ float *)ws; work.SetGlobalBuffer(workBase);
    }
    __aicore__ inline void Process() {
        if constexpr (!LOCAL) { if (t.parallel) { Parallel(); return; } }
        for (uint32_t b = GetBlockIdx(); b < t.batch; b += GetBlockNum()) {
            uint32_t offset = b * t.m * t.n;
            if (t.localInverse && LocalInverse(offset)) continue;
            if (!t.local) work.SetGlobalBuffer(workBase + uint64_t(b) * t.cols * stride);
            float scale = 0;
            for (uint32_t i = 0; i < t.cols; ++i) {
                Duplicate(x, 0.0f, rowLength); PipeBarrier<PIPE_V>();
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
            float roundoffFloor = total * 1.0e-14f;
            discard = discard < roundoffFloor ? discard : roundoffFloor;
            bool cached = false;
            for (uint32_t sweep = 0; sweep < 40; ++sweep) {
                uint32_t changed = 0;
                for (uint32_t p = 0; p < t.cols; ++p) {
                    ReadRow(x, p);
                    for (uint32_t q = p + 1; q < t.cols; ++q) {
                        ReadRow(y, q);
                        float a, d, h;
                        if (cached) { h = Dot(x, y); a = x.GetValue(s); d = y.GetValue(s); }
                        else { a = Dot(x, x); d = Dot(y, y); h = Dot(x, y); }
                        if (a <= discard || d <= discard || h == 0 || h * h <= 1.0e-12f * (a * d)) continue;
                        float tau = (d - a) / (2 * h);
                        float at = AbsScalar(tau);
                        float z = (tau < 0 ? -1.0f : 1.0f) / (at + Root(1 + at * at));
                        float c = 1.0f / Root(1 + z * z), sn = c * z;
                        Muls(u, x, c, s); Muls(v, y, -sn, s); PipeBarrier<PIPE_V>(); Add(u, u, v, s); PipeBarrier<PIPE_V>();
                        Muls(v, x, sn, s); Muls(y, y, c, s); PipeBarrier<PIPE_V>(); Add(y, y, v, s); PipeBarrier<PIPE_V>();
                        if (cached) {
                            float na = Dot(u, u), nd = Dot(y, y);
                            u.SetValue(s, na); y.SetValue(s, nd); SV();
                        }
                        DataCopy(x, u, stride); PipeBarrier<PIPE_V>(); SaveRow(y, q); ++changed;
                    }
                    SaveRow(x, p);
                }
                if (changed == 0) break;
                if (!cached && changed < 0.35f * t.cols * (t.cols - 1)) { CacheNorms(); cached = true; }
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
template <bool TAIL, uint32_t PAD = 0>
class Pinv {
    TBuf<TPosition::VECCALC> mem, precision, vectorPrecision, inverseLow, batchPrecision, parallelElimination;
    LocalTensor<float> batchCalc[8], batchU, batchV, batchH, batchL;
    LocalTensor<float> tileHigh, tileLow;
    LocalTensor<float> lowMatrix;
    LocalTensor<float> calc[8];
    LocalTensor<float> gh, gl, ih, il;
    LocalTensor<float> matrix, x, y, u, v, prod, tmp, red, gather, norms, output;
    GlobalTensor<float> input, result, work;
    __gm__ float *workBase;
    LinalgPinvTilingData t;
    uint32_t r8, k8, s, stride, rowLength;
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
    __aicore__ inline LocalTensor<float> Row(uint32_t i) { return matrix[i * stride]; }
    __aicore__ inline void ReadRow(LocalTensor<float> dst, uint32_t i) {
        if (t.local) { DataCopy(dst, Row(i), rowLength); PipeBarrier<PIPE_V>(); }
        else Load(dst, work[i * stride], rowLength);
    }
    __aicore__ inline void SaveRow(LocalTensor<float> src, uint32_t i) {
        if (t.local) { DataCopy(Row(i), src, rowLength); PipeBarrier<PIPE_V>(); }
        else Store(work[i * stride], src, rowLength);
    }
    __aicore__ inline float Dot(LocalTensor<float> a, LocalTensor<float> b) {
        Mul(prod, a, b, r8); PipeBarrier<PIPE_V>();
        if (r8 <= 64) WholeReduceSum(red, prod, r8, 1, 1, 1, 8);
        else ReduceSum(red, prod, tmp, r8);
        VS(); return red.GetValue(0);
    }
    __aicore__ inline void CacheNorms() {
        uint32_t first = t.parallel ? GetBlockIdx() : 0;
        uint32_t step = t.parallel ? GetBlockNum() : 1;
        for (uint32_t i = first; i < t.cols; i += step) {
            ReadRow(x, i); float a = Dot(x, x); x.SetValue(s, a);
            if (t.local) SV();
            SaveRow(x, i);
        }
        if (t.parallel) SyncAll();
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
            Duplicate(x, 0.0f, rowLength); PipeBarrier<PIPE_V>();
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
        Muls(e, bh, static_cast<float>(ah), s); PipeBarrier<PIPE_V>(); Sub(e, e, product, s); PipeBarrier<PIPE_V>();
        Muls(part, bl, static_cast<float>(ah), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, bh, static_cast<float>(al), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, bl, static_cast<float>(al), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, v, factor.h, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, u, factor.l, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Add(qh, product, e, s); PipeBarrier<PIPE_V>(); Sub(ql, qh, product, s); PipeBarrier<PIPE_V>(); Sub(ql, e, ql, s); PipeBarrier<PIPE_V>();
        Sub(bh, x, qh, s); PipeBarrier<PIPE_V>(); Sub(bl, bh, x, s); PipeBarrier<PIPE_V>();
        Sub(e, bh, bl, s); PipeBarrier<PIPE_V>(); Sub(e, x, e, s); Muls(part, qh, -1.0f, s); PipeBarrier<PIPE_V>();
        Sub(part, part, bl, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Add(e, e, y, s); PipeBarrier<PIPE_V>(); Sub(e, e, ql, s); PipeBarrier<PIPE_V>();
        Add(x, bh, e, s); PipeBarrier<PIPE_V>(); Sub(aux, x, bh, s); PipeBarrier<PIPE_V>(); Sub(y, e, aux, s); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void VectorNormalize(Wide divisor) {
        float reciprocal = 1.0f / divisor.h;
        Muls(u, x, reciprocal, s); Duplicate(v, 0.0f, s); PipeBarrier<PIPE_V>();
        VectorEliminate(divisor);
        Muls(calc[0], x, reciprocal, s); PipeBarrier<PIPE_V>();
        Add(x, u, calc[0], s); PipeBarrier<PIPE_V>();
        Sub(y, x, u, s); PipeBarrier<PIPE_V>();
        Sub(y, calc[0], y, s); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool AccurateSquare(uint32_t b, uint32_t offset, float scale, GlobalTensor<float> meta,
                                         bool certify = false, float cutoff = 0.0f) {
        uint32_t core = GetBlockIdx(), cores = GetBlockNum();
        GlobalTensor<float> low;
        low.SetGlobalBuffer(workBase + uint64_t(t.batch + b) * t.cols * stride);
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        for (uint32_t i = core; i < t.cols; i += cores) {
            Duplicate(x, 0.0f, rowLength); Duplicate(y, 0.0f, rowLength); PipeBarrier<PIPE_V>();
            Load(x, input[offset + i * t.n], t.n);
            Muls(x, x, 1.0f / power, r8); PipeBarrier<PIPE_V>(); VS();
            x.SetValue(r8 + i, 1.0f); SV(); SaveRow(x, i); Store(low[i * stride], y, rowLength);
        }
        SyncAll();
        for (uint32_t p = 0; p < t.cols; ++p) {
            float best = 0; uint32_t bestRow = p;
            if (p == 0) {
                for (uint32_t i = core; i < t.cols; i += cores) {
                    ReadRow(x, i); VS(); float value = AbsScalar(x.GetValue(p));
                    if (value > best) { best = value; bestRow = i; }
                }
                red.SetValue(0, best); Store(meta[core * 8], red, 1);
                red.SetValue(0, static_cast<float>(static_cast<int32_t>(bestRow))); Store(meta[256 + core * 8], red, 1);
                SyncAll();
            }
            Load(gather, meta, cores * 8); best = 0; uint32_t bestCore = 0;
            for (uint32_t i = 0; i < cores; ++i) { float value = gather.GetValue(i * 8); if (value > best) { best = value; bestCore = i; } }
            if (certify && !(best > 0.0f)) return false;
            Load(gather, meta[256], cores * 8);
            uint32_t pivot = static_cast<uint32_t>(static_cast<int32_t>(gather.GetValue(bestCore * 8)));
            if (core == 0) {
                if (pivot != p) {
                    ReadRow(x, pivot); ReadRow(y, p); Load(u, low[pivot * stride], rowLength); Load(v, low[p * stride], rowLength);
                    SaveRow(x, p); SaveRow(y, pivot); Store(low[p * stride], u, rowLength); Store(low[pivot * stride], v, rowLength);
                }
                ReadRow(x, p); Load(y, low[p * stride], rowLength); VS();
                Wide divisor = {x.GetValue(p), y.GetValue(p)};
                SV(); VectorNormalize(divisor);
                SaveRow(x, p); Store(low[p * stride], y, rowLength);
            }
            SyncAll(); ReadRow(u, p); Load(v, low[p * stride], rowLength); VS();
            float nextBest = 0.0f; uint32_t nextRow = p + 1;
            auto bx = x, by = y;
            for (uint32_t row = 0; row < t.inverseTile; ++row) {
                Duplicate(batchU[row * stride], 0.0f, stride); Duplicate(batchV[row * stride], 0.0f, stride); PipeBarrier<PIPE_V>();
                DataCopy(batchU[row * stride], u, rowLength); DataCopy(batchV[row * stride], v, rowLength);
            }
            PipeBarrier<PIPE_V>();
            uint32_t owned = (t.cols + cores - 1 - core) / cores;
            for (uint32_t first = 0; first < owned; first += t.inverseTile) {
                uint32_t rows = owned - first < t.inverseTile ? owned - first : t.inverseTile;
                uint32_t index = core + first * cores;
                Duplicate(tileHigh, 0.0f, rows * stride); Duplicate(tileLow, 0.0f, rows * stride); PipeBarrier<PIPE_V>();
                DataCopyExtParams inParams{static_cast<uint16_t>(rows), rowLength * 4, (cores * stride - rowLength) * 4, (stride - rowLength) / 8, 0};
                SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
                DataCopyPad(tileHigh, work[index * stride], inParams, DataCopyPadExtParams<float>{false, 0, 0, 0});
                DataCopyPad(tileLow, low[index * stride], inParams, DataCopyPadExtParams<float>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
                SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
                for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t i = index + row * cores;
                    float fh = i == p ? 0.0f : tileHigh.GetValue(row * stride + p);
                    float fl = i == p ? 0.0f : tileLow.GetValue(row * stride + p);
                    Duplicate(batchH[row * stride], fh, stride); Duplicate(batchL[row * stride], fl, stride);
                }
                PipeBarrier<PIPE_V>(); x = tileHigh; y = tileLow;
                BatchEliminate(rows * stride); VS();
                for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t i = index + row * cores;
                    if (i > p && p + 1 < t.cols) {
                        float value = AbsScalar(tileHigh.GetValue(row * stride + p + 1));
                        if (value > nextBest) { nextBest = value; nextRow = i; }
                    }
                }
                DataCopyExtParams outParams{static_cast<uint16_t>(rows), rowLength * 4, (stride - rowLength) / 8, (cores * stride - rowLength) * 4, 0};
                SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
                SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
                DataCopyPad(work[index * stride], tileHigh, outParams); DataCopyPad(low[index * stride], tileLow, outParams);
                SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
                SetFlag<HardEvent::MTE3_MTE2>(0); WaitFlag<HardEvent::MTE3_MTE2>(0);
            }
            x = bx; y = by;
            if (p + 1 < t.cols) {
                red.SetValue(0, nextBest); Store(meta[core * 8], red, 1);
                red.SetValue(0, static_cast<float>(static_cast<int32_t>(nextRow))); Store(meta[256 + core * 8], red, 1);
            }
            SyncAll();
        }
        if (certify) {
            float inverseTotal = 0.0f;
            for (uint32_t row = core; row < t.n; row += cores) {
                ReadRow(x, row);
                inverseTotal += Dot(x[r8], x[r8]);
            }
            red.SetValue(0, inverseTotal); Store(meta[core * 8], red, 1); SyncAll();
            Load(gather, meta, cores * 8); inverseTotal = 0.0f;
            for (uint32_t i = 0; i < cores; ++i) inverseTotal += gather.GetValue(i * 8);
            float scaledCutoff = cutoff / power;
            // The promised spectral gap bounds a discarded nonzero value by 0.1 times the threshold.
            // Such a value forces the inverse norm times this upper threshold to be at least ten.
            if (!(inverseTotal > 0.0f && inverseTotal * scaledCutoff * scaledCutoff < 81.0f)) return false;
        }
        for (uint32_t row = core; row < t.n; row += cores) {
            ReadRow(x, row); Load(y, low[row * stride], rowLength); VS();
            for (uint32_t j = 0; j < t.m; ++j) output.SetValue(j, (x.GetValue(r8 + j) + y.GetValue(r8 + j)) / power);
            Store(result[offset + row * t.m], output, t.m);
        }
        SyncAll();
        return true;
    }
    __aicore__ inline void RestoreJacobi(uint32_t offset, float scale) {
        for (uint32_t i = GetBlockIdx(); i < t.cols; i += GetBlockNum()) {
            Duplicate(x, 0.0f, rowLength); PipeBarrier<PIPE_V>();
            DataCopyPad(gather, input[offset + i], DataCopyExtParams{static_cast<uint16_t>(t.m), 4, (t.n - 1) * 4, 0, 0}, DataCopyPadExtParams<float>{true, 0, 7, 0});
            SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0); VS();
            for (uint32_t j = 0; j < t.rows; ++j) x.SetValue(j, gather.GetValue(j * 8));
            x.SetValue(r8 + i, 1.0f); SV();
            Muls(x, x, 1.0f / scale, r8); PipeBarrier<PIPE_V>(); SaveRow(x, i);
        }
        SyncAll();
    }
    __aicore__ inline void BatchEliminate(uint32_t count) {
        auto qh = batchCalc[0], ql = batchCalc[1], bh = batchCalc[2], bl = batchCalc[3], e = batchCalc[4], part = batchCalc[5], product = batchCalc[6], aux = batchCalc[7];
        Muls(bh, batchH, 4097.0f, count); PipeBarrier<PIPE_V>();
        Sub(bl, bh, batchH, count); PipeBarrier<PIPE_V>(); Sub(bh, bh, bl, count); PipeBarrier<PIPE_V>();
        Sub(bl, batchH, bh, count); Muls(qh, batchU, 4097.0f, count); PipeBarrier<PIPE_V>();
        Sub(ql, qh, batchU, count); PipeBarrier<PIPE_V>(); Sub(qh, qh, ql, count); PipeBarrier<PIPE_V>();
        Sub(ql, batchU, qh, count); Mul(product, batchU, batchH, count); PipeBarrier<PIPE_V>();
        Mul(e, qh, bh, count); PipeBarrier<PIPE_V>(); Sub(e, e, product, count); PipeBarrier<PIPE_V>();
        Mul(part, ql, bh, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Mul(part, qh, bl, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Mul(part, ql, bl, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Mul(part, batchV, batchH, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Mul(part, batchU, batchL, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Add(qh, product, e, count); PipeBarrier<PIPE_V>(); Sub(ql, qh, product, count); PipeBarrier<PIPE_V>(); Sub(ql, e, ql, count); PipeBarrier<PIPE_V>();
        Sub(bh, x, qh, count); PipeBarrier<PIPE_V>(); Sub(bl, bh, x, count); PipeBarrier<PIPE_V>();
        Sub(e, bh, bl, count); PipeBarrier<PIPE_V>(); Sub(e, x, e, count); Muls(part, qh, -1.0f, count); PipeBarrier<PIPE_V>();
        Sub(part, part, bl, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Add(e, e, y, count); PipeBarrier<PIPE_V>(); Sub(e, e, ql, count); PipeBarrier<PIPE_V>();
        Add(x, bh, e, count); PipeBarrier<PIPE_V>(); Sub(aux, x, bh, count); PipeBarrier<PIPE_V>(); Sub(y, e, aux, count); PipeBarrier<PIPE_V>();
    }
    template <uint32_t LANES>
    __aicore__ inline bool CompactInverseLegacyLanes(uint32_t offset, LocalTensor<float> bx, LocalTensor<float> by,
                                            LocalTensor<float> bu, LocalTensor<float> bv, bool tryQuick = false) {
        const uint32_t n = LANES ? LANES : t.cols;
        auto highData = reinterpret_cast<__ubuf__ float *>(matrix.GetPhyAddr());
        auto lowData = reinterpret_cast<__ubuf__ float *>(lowMatrix.GetPhyAddr());
        bool compact = true;
        uint32_t count = n * stride;
        float scale = 0.0f;
        if (compact) {
            Duplicate(matrix, 0.0f, count); Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
            DataCopyPad(matrix, input[offset], DataCopyExtParams{static_cast<uint16_t>(n), n * 4, 0, (stride - r8) / 8, 0}, DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(r8 - n), 0});
            SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
            Abs(lowMatrix, matrix, count); PipeBarrier<PIPE_V>(); ReduceMax(red, lowMatrix, tmp, count, false); VS();
            scale = red.GetValue(0);
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                x = Row(i); y = lowMatrix[i * stride];
                Duplicate(x, 0.0f, rowLength); Duplicate(y, 0.0f, rowLength); PipeBarrier<PIPE_V>();
                Load(x, input[offset + i * n], n);
                Abs(prod, x, r8); PipeBarrier<PIPE_V>(); ReduceMax(red, prod, tmp, r8, false); VS();
                float mx = red.GetValue(0); scale = mx > scale ? mx : scale;
                x.SetValue(r8 + i, 1.0f); SV();
            }
        }
        if (!(scale > 0.0f)) return false;
        bool quick = scale >= 0.1f;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        float total = 0.0f;
        if (compact) {
            Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
            Mul(lowMatrix, matrix, matrix, count); PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS(); total = red.GetValue(0);
            Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < n; ++i) highData[i * stride + r8 + i] = 1.0f;
            SV();
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                x = Row(i); Muls(x, x, 1.0f / power, r8); PipeBarrier<PIPE_V>(); total += Dot(x, x);
            }
        }
        bool vectorQuick = quick && n >= 8;
        auto indices = gather.ReinterpretCast<uint32_t>();
        if (vectorQuick) {
            for (uint32_t i = 0; i < k8; ++i) indices.SetValue(i, i < n ? i * stride * sizeof(float) : 0);
            Duplicate(norms, 0.0f, k8); PipeBarrier<PIPE_V>(); SV();
        }
        for (uint32_t p = 0; p < n; ++p) {
            uint32_t pivot = p; float best = 0.0f;
            if (vectorQuick) {
                Gather(norms, matrix, indices, p * sizeof(float), n); PipeBarrier<PIPE_V>();
                Abs(prod, norms, k8); PipeBarrier<PIPE_V>();
                uint64_t pivotMask[2] = {((uint64_t(1) << n) - 1) & ~((uint64_t(1) << p) - 1), 0};
                WholeReduceMax(red, prod, pivotMask, 1, 1, 1, 8); VS();
                best = red.GetValue(0); pivot = red.ReinterpretCast<uint32_t>().GetValue(1);
            } else {
                if (p == 0 || !quick) VS();
                for (uint32_t i = p; i < n; ++i) {
                    float value = AbsScalar(highData[i * stride + p]);
                    if (value > best) { best = value; pivot = i; }
                }
            }
            if (!(best > 0.0f)) return false;
            if (pivot != p) {
                DataCopy(bx, Row(p), rowLength); if (!quick) DataCopy(by, lowMatrix[p * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(p), Row(pivot), rowLength); if (!quick) DataCopy(lowMatrix[p * stride], lowMatrix[pivot * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(pivot), bx, rowLength); if (!quick) DataCopy(lowMatrix[pivot * stride], by, rowLength); PipeBarrier<PIPE_V>();
            }
            if (quick) {
                if (pivot != p) VS();
                auto pivotLocal = matrix[p * stride];
                float divisor = highData[p * stride + p];
                Muls(pivotLocal, pivotLocal, 1.0f / divisor, s); PipeBarrier<PIPE_V>();
                DataCopy(bu, pivotLocal, stride); PipeBarrier<PIPE_V>();
                if (vectorQuick) {
                    if (pivot != p) norms.SetValue(pivot, norms.GetValue(p));
                    norms.SetValue(p, 0.0f); SV(); Muls(norms, norms, -1.0f, k8); PipeBarrier<PIPE_V>();
                    if constexpr (TAIL) {
                        Brcb(lowMatrix, norms, k8 / 8, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)});
                    } else {
                        Brcb(lowMatrix, norms, n / 8, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)});
                    }
                    PipeBarrier<PIPE_V>();
                    MulAddDst(matrix, bu, lowMatrix, static_cast<uint64_t>(stride), static_cast<uint8_t>(n), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, static_cast<uint8_t>(stride / 8)});
                    PipeBarrier<PIPE_V>();
                    continue;
                }
                for (uint32_t first = 0; first < n; first += t.inverseTile) {
                    uint32_t rows = n - first < t.inverseTile ? n - first : t.inverseTile;
                    for (uint32_t row = 0; row < 8; ++row) {
                        uint32_t i = first + row;
                        float factor = row >= rows || i == p ? 0.0f : highData[i * stride + p];
                        red.SetValue(row, -factor);
                    }
                    SV(); Brcb(batchH, red, 1, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)}); PipeBarrier<PIPE_V>();
                    auto target = matrix[first * stride];
                    MulAddDst(target, bu, batchH, static_cast<uint64_t>(stride), static_cast<uint8_t>(rows), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, static_cast<uint8_t>(stride / 8)});
                    PipeBarrier<PIPE_V>(); VS();
                }
                continue;
            }
            x = Row(p); y = lowMatrix[p * stride]; u = bu; v = bv; VS();
            Wide divisor = {highData[p * stride + p], lowData[p * stride + p]}; SV();
            VectorNormalize(divisor); VS(); u = x; v = y;
            for (uint32_t first = 0; first < n; first += t.inverseTile) {
                uint32_t rows = n - first < t.inverseTile ? n - first : t.inverseTile;
                for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t i = first + row;
                    float fh = i == p ? 0.0f : highData[i * stride + p];
                    float fl = i == p ? 0.0f : lowData[i * stride + p];
                    Duplicate(batchH[row * stride], fh, stride);
                    DataCopy(batchU[row * stride], u, stride);
                    Duplicate(batchL[row * stride], fl, stride); DataCopy(batchV[row * stride], v, stride);
                }
                PipeBarrier<PIPE_V>();
                x = Row(first); y = lowMatrix[first * stride];
                BatchEliminate(rows * stride); VS();
            }
        }
        float inverseTotal = 0.0f;
        if (compact) {
            if (!vectorQuick) { Add(matrix, matrix, lowMatrix, count); PipeBarrier<PIPE_V>(); }
            Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
            Mul(lowMatrix, matrix[r8], matrix[r8], static_cast<uint64_t>(r8), static_cast<uint8_t>(n), {1, 1, 1, static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride / 8)});
            PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS(); inverseTotal = red.GetValue(0);
        } else {
            for (uint32_t i = 0; i < n; ++i) { x = Row(i); inverseTotal += Dot(x[r8], x[r8]); }
        }
        float cutoff = Root(total) * t.rtol;
        float absolute = t.atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        if (!(inverseTotal > 0.0f && inverseTotal * cutoff * cutoff < 0.81f)) return false;
        if (compact) {
            Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
            bool packedOutput = n * n <= 8 * s;
            auto stored = matrix[r8];
            if (packedOutput) {
                auto pattern = red.ReinterpretCast<uint32_t>();
                pattern.SetValue(0, (uint32_t(1) << n) - 1);
                for (uint32_t i = 1; i < 8; ++i) pattern.SetValue(i, 0);
                SV();
                auto packed = vectorPrecision.Get<float>();
                uint64_t retained = 0;
                GatherMask(packed, matrix[r8], pattern, false, 0,
                           {1, static_cast<uint8_t>(n), static_cast<uint16_t>(stride / 8), 0}, retained);
                PipeBarrier<PIPE_V>(); stored = packed;
            }
            SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
            SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
            if (packedOutput) DataCopyPad(result[offset], stored, DataCopyExtParams{1, n * n * 4, 0, 0, 0});
            else DataCopyPad(result[offset], matrix[r8], DataCopyExtParams{static_cast<uint16_t>(n), n * 4, (stride - r8) / 8, 0, 0});
            SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
            SetFlag<HardEvent::MTE3_MTE2>(0); WaitFlag<HardEvent::MTE3_MTE2>(0);
            return true;
        }
        for (uint32_t i = 0; i < n; ++i) {
            x = Row(i); y = lowMatrix[i * stride];
            Add(output, x[r8], y[r8], r8); PipeBarrier<PIPE_V>(); Muls(output, output, 1.0f / power, r8); PipeBarrier<PIPE_V>();
            Store(result[offset + i * n], output, n);
        }
        return true;
    }
    __aicore__ inline bool CompactInverseLegacyBody(uint32_t offset, LocalTensor<float> bx, LocalTensor<float> by,
                                            LocalTensor<float> bu, LocalTensor<float> bv, bool tryQuick = false) {
        if constexpr (PAD > 0) {
            if (t.cols % 8 == 7) return CompactInverseLegacyLanes<PAD - 1>(offset, bx, by, bu, bv, tryQuick);
        }
        return CompactInverseLegacyLanes<0>(offset, bx, by, bu, bv, tryQuick);
    }
    __aicore__ inline bool CompactInverseBody(uint32_t offset, LocalTensor<float> bx, LocalTensor<float> by,
                                            LocalTensor<float> bu, LocalTensor<float> bv, bool tryQuick = false) {
        auto highData = reinterpret_cast<__ubuf__ float *>(matrix.GetPhyAddr());
        auto lowData = reinterpret_cast<__ubuf__ float *>(lowMatrix.GetPhyAddr());
        bool compact = true;
        uint32_t count = t.cols * stride;
        float scale = 0.0f;
        if (compact) {
            Duplicate(matrix, 0.0f, count); Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
            DataCopyPad(matrix, input[offset], DataCopyExtParams{static_cast<uint16_t>(t.cols), t.n * 4, 0, (stride - r8) / 8, 0}, DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(r8 - t.n), 0});
            SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
            Abs(lowMatrix, matrix, count); PipeBarrier<PIPE_V>(); ReduceMax(red, lowMatrix, tmp, count, false); VS();
            scale = red.GetValue(0);
        } else {
            for (uint32_t i = 0; i < t.cols; ++i) {
                x = Row(i); y = lowMatrix[i * stride];
                Duplicate(x, 0.0f, rowLength); Duplicate(y, 0.0f, rowLength); PipeBarrier<PIPE_V>();
                Load(x, input[offset + i * t.n], t.n);
                Abs(prod, x, r8); PipeBarrier<PIPE_V>(); ReduceMax(red, prod, tmp, r8, false); VS();
                float mx = red.GetValue(0); scale = mx > scale ? mx : scale;
                x.SetValue(r8 + i, 1.0f); SV();
            }
        }
        if (!(scale > 0.0f)) return false;
        bool quick = scale >= 0.1f;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        float total = 0.0f;
        if (compact) {
            Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
            Mul(lowMatrix, matrix, matrix, count); PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS(); total = red.GetValue(0);
            Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < t.cols; ++i) highData[i * stride + r8 + i] = 1.0f;
            SV();
        } else {
            for (uint32_t i = 0; i < t.cols; ++i) {
                x = Row(i); Muls(x, x, 1.0f / power, r8); PipeBarrier<PIPE_V>(); total += Dot(x, x);
            }
        }
        bool vectorQuick = quick && t.cols >= 8;
        bool logicalPivot = vectorQuick && !TAIL;
        auto indices = gather.ReinterpretCast<uint32_t>();
        auto pivotOrder = output.ReinterpretCast<uint32_t>();
        uint64_t available = (uint64_t(1) << t.cols) - 1;
        if (vectorQuick) {
            for (uint32_t i = 0; i < k8; ++i) indices.SetValue(i, i < t.cols ? i * stride * sizeof(float) : 0);
            Duplicate(norms, 0.0f, k8); PipeBarrier<PIPE_V>(); SV();
        }
        for (uint32_t p = 0; p < t.cols; ++p) {
            uint32_t pivot = p; float best = 0.0f;
            if (vectorQuick) {
                Gather(norms, matrix, indices, p * sizeof(float), t.cols); PipeBarrier<PIPE_V>();
                Abs(prod, norms, k8); PipeBarrier<PIPE_V>();
                uint64_t pivotMask[2] = {logicalPivot ? available : (((uint64_t(1) << t.cols) - 1) & ~((uint64_t(1) << p) - 1)), 0};
                WholeReduceMax(red, prod, pivotMask, 1, 1, 1, 8); VS();
                best = red.GetValue(0); pivot = red.ReinterpretCast<uint32_t>().GetValue(1);
            } else {
                if (p == 0 || !quick) VS();
                for (uint32_t i = p; i < t.cols; ++i) {
                    float value = AbsScalar(highData[i * stride + p]);
                    if (value > best) { best = value; pivot = i; }
                }
            }
            if (!(best > 0.0f)) return false;
            if (!logicalPivot && pivot != p) {
                DataCopy(bx, Row(p), rowLength); if (!quick) DataCopy(by, lowMatrix[p * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(p), Row(pivot), rowLength); if (!quick) DataCopy(lowMatrix[p * stride], lowMatrix[pivot * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(pivot), bx, rowLength); if (!quick) DataCopy(lowMatrix[pivot * stride], by, rowLength); PipeBarrier<PIPE_V>();
            }
            if (quick) {
                if (!logicalPivot && pivot != p) VS();
                uint32_t physicalPivot = logicalPivot ? pivot : p;
                auto pivotLocal = matrix[physicalPivot * stride];
                float divisor = highData[physicalPivot * stride + p];
                Muls(pivotLocal, pivotLocal, 1.0f / divisor, s); PipeBarrier<PIPE_V>();
                DataCopy(bu, pivotLocal, stride); PipeBarrier<PIPE_V>();
                if (vectorQuick) {
                    if (logicalPivot) {
                        available &= ~(uint64_t(1) << pivot);
                        pivotOrder.SetValue(pivot, p);
                        norms.SetValue(pivot, 0.0f);
                    } else {
                        if (pivot != p) norms.SetValue(pivot, norms.GetValue(p));
                        norms.SetValue(p, 0.0f);
                    }
                    SV(); Muls(norms, norms, -1.0f, k8); PipeBarrier<PIPE_V>();
                    if constexpr (TAIL) {
                        Brcb(lowMatrix, norms, k8 / 8, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)});
                    } else {
                        Brcb(lowMatrix, norms, t.cols / 8, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)});
                    }
                    PipeBarrier<PIPE_V>();
                    MulAddDst(matrix, bu, lowMatrix, static_cast<uint64_t>(stride), static_cast<uint8_t>(t.cols), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, static_cast<uint8_t>(stride / 8)});
                    PipeBarrier<PIPE_V>();
                    continue;
                }
                for (uint32_t first = 0; first < t.cols; first += t.inverseTile) {
                    uint32_t rows = t.cols - first < t.inverseTile ? t.cols - first : t.inverseTile;
                    for (uint32_t row = 0; row < 8; ++row) {
                        uint32_t i = first + row;
                        float factor = row >= rows || i == p ? 0.0f : highData[i * stride + p];
                        red.SetValue(row, -factor);
                    }
                    SV(); Brcb(batchH, red, 1, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)}); PipeBarrier<PIPE_V>();
                    auto target = matrix[first * stride];
                    MulAddDst(target, bu, batchH, static_cast<uint64_t>(stride), static_cast<uint8_t>(rows), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, static_cast<uint8_t>(stride / 8)});
                    PipeBarrier<PIPE_V>(); VS();
                }
                continue;
            }
            x = Row(p); y = lowMatrix[p * stride]; u = bu; v = bv; VS();
            Wide divisor = {highData[p * stride + p], lowData[p * stride + p]}; SV();
            VectorNormalize(divisor); VS(); u = x; v = y;
            for (uint32_t first = 0; first < t.cols; first += t.inverseTile) {
                uint32_t rows = t.cols - first < t.inverseTile ? t.cols - first : t.inverseTile;
                for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t i = first + row;
                    float fh = i == p ? 0.0f : highData[i * stride + p];
                    float fl = i == p ? 0.0f : lowData[i * stride + p];
                    Duplicate(batchH[row * stride], fh, stride);
                    DataCopy(batchU[row * stride], u, stride);
                    Duplicate(batchL[row * stride], fl, stride); DataCopy(batchV[row * stride], v, stride);
                }
                PipeBarrier<PIPE_V>();
                x = Row(first); y = lowMatrix[first * stride];
                BatchEliminate(rows * stride); VS();
            }
        }
        float inverseTotal = 0.0f;
        if (compact) {
            if (!vectorQuick) { Add(matrix, matrix, lowMatrix, count); PipeBarrier<PIPE_V>(); }
            Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
            Mul(lowMatrix, matrix[r8], matrix[r8], static_cast<uint64_t>(r8), static_cast<uint8_t>(t.cols), {1, 1, 1, static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride / 8)});
            PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS(); inverseTotal = red.GetValue(0);
        } else {
            for (uint32_t i = 0; i < t.cols; ++i) { x = Row(i); inverseTotal += Dot(x[r8], x[r8]); }
        }
        float cutoff = Root(total) * t.rtol;
        float absolute = t.atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        if (!(inverseTotal > 0.0f && inverseTotal * cutoff * cutoff < 0.81f)) return false;
        if (compact) {
            Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
            auto ordered = matrix;
            if (logicalPivot) {
                for (uint32_t row = 0; row < t.cols; ++row) {
                    uint32_t target = pivotOrder.GetValue(row);
                    DataCopy(lowMatrix[target * stride], matrix[row * stride], stride);
                }
                PipeBarrier<PIPE_V>(); ordered = lowMatrix;
            }
            SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
            SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
            DataCopyPad(result[offset], ordered[r8], DataCopyExtParams{static_cast<uint16_t>(t.n), t.m * 4, (stride - r8) / 8, 0, 0});
            SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
            SetFlag<HardEvent::MTE3_MTE2>(0); WaitFlag<HardEvent::MTE3_MTE2>(0);
            return true;
        }
        for (uint32_t i = 0; i < t.n; ++i) {
            x = Row(i); y = lowMatrix[i * stride];
            Add(output, x[r8], y[r8], r8); PipeBarrier<PIPE_V>(); Muls(output, output, 1.0f / power, r8); PipeBarrier<PIPE_V>();
            Store(result[offset + i * t.m], output, t.m);
        }
        return true;
    }
    __aicore__ inline bool AccurateLocalInverseBody(uint32_t offset, LocalTensor<float> bx, LocalTensor<float> by,
                                            LocalTensor<float> bu, LocalTensor<float> bv) {
        float scale = 0.0f;
        for (uint32_t i = 0; i < t.cols; ++i) {
            x = Row(i); y = lowMatrix[i * stride];
            Duplicate(x, 0.0f, rowLength); Duplicate(y, 0.0f, rowLength); PipeBarrier<PIPE_V>();
            Load(x, input[offset + i * t.n], t.n);
            Abs(prod, x, r8); PipeBarrier<PIPE_V>(); ReduceMax(red, prod, tmp, r8, false); VS();
            float mx = red.GetValue(0); scale = mx > scale ? mx : scale;
            x.SetValue(r8 + i, 1.0f); SV();
        }
        if (!(scale > 0.0f)) return false;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        float total = 0.0f;
        for (uint32_t i = 0; i < t.cols; ++i) {
            x = Row(i); Muls(x, x, 1.0f / power, r8); PipeBarrier<PIPE_V>(); total += Dot(x, x);
        }
        for (uint32_t p = 0; p < t.cols; ++p) {
            VS(); uint32_t pivot = p; float best = 0.0f;
            for (uint32_t i = p; i < t.cols; ++i) {
                float value = AbsScalar(Row(i).GetValue(p));
                if (value > best) { best = value; pivot = i; }
            }
            if (!(best > 0.0f)) return false;
            if (pivot != p) {
                DataCopy(bx, Row(p), rowLength); DataCopy(by, lowMatrix[p * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(p), Row(pivot), rowLength); DataCopy(lowMatrix[p * stride], lowMatrix[pivot * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(pivot), bx, rowLength); DataCopy(lowMatrix[pivot * stride], by, rowLength); PipeBarrier<PIPE_V>();
            }
            x = Row(p); y = lowMatrix[p * stride]; u = bu; v = bv; VS();
            Wide divisor = {x.GetValue(p), y.GetValue(p)}; SV(); VectorNormalize(divisor);
            VS(); u = x; v = y;
            for (uint32_t first = 0; first < t.cols; first += t.inverseTile) {
                uint32_t rows = t.cols - first < t.inverseTile ? t.cols - first : t.inverseTile;
                for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t i = first + row;
                    float fh = i == p ? 0.0f : Row(i).GetValue(p);
                    float fl = i == p ? 0.0f : lowMatrix[i * stride].GetValue(p);
                    Duplicate(batchH[row * stride], fh, stride); Duplicate(batchL[row * stride], fl, stride);
                    DataCopy(batchU[row * stride], u, stride); DataCopy(batchV[row * stride], v, stride);
                }
                PipeBarrier<PIPE_V>();
                x = Row(first); y = lowMatrix[first * stride];
                BatchEliminate(rows * stride); VS();
            }
        }
        float inverseTotal = 0.0f;
        for (uint32_t i = 0; i < t.cols; ++i) { x = Row(i); inverseTotal += Dot(x[r8], x[r8]); }
        float cutoff = Root(total) * t.rtol;
        float absolute = t.atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        if (!(inverseTotal > 0.0f && inverseTotal * cutoff * cutoff < 0.81f)) return false;
        for (uint32_t i = 0; i < t.n; ++i) {
            x = Row(i); y = lowMatrix[i * stride];
            Add(output, x[r8], y[r8], r8); PipeBarrier<PIPE_V>(); Muls(output, output, 1.0f / power, r8); PipeBarrier<PIPE_V>();
            Store(result[offset + i * t.m], output, t.m);
        }
        return true;
    }
    __aicore__ inline bool LocalInverse(uint32_t offset) {
        auto bx = x, by = y, bu = u, bv = v;
        bool accepted;
        if constexpr (TAIL) accepted = stride <= 64 ? CompactInverseLegacyBody(offset, bx, by, bu, bv) : AccurateLocalInverseBody(offset, bx, by, bu, bv);
        else accepted = stride <= 64 ? CompactInverseBody(offset, bx, by, bu, bv) : AccurateLocalInverseBody(offset, bx, by, bu, bv);
        x = bx; y = by; u = bu; v = bv;
        return accepted;
    }
    __aicore__ inline void Parallel() {
        uint32_t core = GetBlockIdx(), cores = GetBlockNum();
        uint32_t roundCols = (t.cols + 1) / 2 * 2;
        for (uint32_t b = 0; b < t.batch; ++b) {
            uint32_t offset = b * t.m * t.n;
            work.SetGlobalBuffer(workBase + uint64_t(b) * t.cols * stride);
            GlobalTensor<float> meta;
            meta.SetGlobalBuffer(workBase + uint64_t(2) * t.batch * t.cols * stride + uint64_t(b) * 8192);
            float scale = 0;
            for (uint32_t i = core; i < t.cols; i += cores) {
                Duplicate(x, 0.0f, rowLength); PipeBarrier<PIPE_V>();
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
                ReadRow(x, i); Muls(x, x, 1.0f / scale, r8); PipeBarrier<PIPE_V>();
                total += Dot(x, x); SaveRow(x, i);
            }
            red.SetValue(0, total); Store(meta[core * 8], red, 1); SyncAll();
            Load(gather, meta, cores * 8); total = 0;
            for (uint32_t i = 0; i < cores; ++i) total += gather.GetValue(i * 8);
            if (t.m == t.n) {
                float cutoff = scale * Root(total) * t.rtol;
                cutoff = cutoff > t.atol ? cutoff : t.atol;
                if (AccurateSquare(b, offset, scale, meta, true, cutoff)) continue;
                RestoreJacobi(offset, scale);
            }
            float absTol = t.atol / scale, discard = absTol * absTol;
            float rel = t.rtol * t.rtol * total;
            discard = (discard > rel ? discard : rel) * 1.0e-8f;
            float roundoffFloor = total * 1.0e-14f;
            discard = discard < roundoffFloor ? discard : roundoffFloor;
            bool cached = false;
            for (uint32_t sweep = 0; sweep < 40; ++sweep) {
                uint32_t changed = 0;
                for (uint32_t round = 0; round + 1 < roundCols; ++round) {
                    for (uint32_t slot = core; slot < roundCols / 2; slot += cores) {
                        uint32_t p = slot == 0 ? roundCols - 1 : (round + slot) % (roundCols - 1);
                        uint32_t q = slot == 0 ? round : (round + roundCols - 1 - slot) % (roundCols - 1);
                        if (p >= t.cols || q >= t.cols) continue;
                        ReadRow(x, p); ReadRow(y, q);
                        float a, d, h;
                        if (cached) { h = Dot(x, y); a = x.GetValue(s); d = y.GetValue(s); }
                        else { a = Dot(x, x); d = Dot(y, y); h = Dot(x, y); }
                        if (a <= discard || d <= discard || h == 0 || h * h <= 1.0e-12f * (a * d)) continue;
                        float tau = (d - a) / (2 * h), at = AbsScalar(tau);
                        float z = (tau < 0 ? -1.0f : 1.0f) / (at + Root(1 + at * at));
                        float c = 1.0f / Root(1 + z * z), sn = c * z;
                        Muls(u, x, c, s); Muls(v, y, -sn, s); PipeBarrier<PIPE_V>(); Add(u, u, v, s); PipeBarrier<PIPE_V>();
                        Muls(v, x, sn, s); Muls(y, y, c, s); PipeBarrier<PIPE_V>(); Add(y, y, v, s); PipeBarrier<PIPE_V>();
                        if (cached) {
                            float na = Dot(u, u), nd = Dot(y, y);
                            u.SetValue(s, na); y.SetValue(s, nd);
                        }
                        SaveRow(u, p); SaveRow(y, q); ++changed;
                    }
                    SyncAll();
                }
                red.SetValue(0, static_cast<float>(static_cast<int32_t>(changed))); Store(meta[core * 8], red, 1); SyncAll();
                Load(gather, meta, cores * 8); float sum = 0;
                for (uint32_t i = 0; i < cores; ++i) sum += gather.GetValue(i * 8);
                if (sum == 0) break;
                if (!cached && sum < 0.35f * t.cols * (t.cols - 1)) { CacheNorms(); cached = true; }
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
            float smallest = maxNorm;
            for (uint32_t i = 0; i < t.cols; ++i) {
                float a = gather.GetValue(i * 8);
                if (a > threshold && a < smallest) smallest = a;
            }
            if (t.m == t.n && t.cols >= 256 && kept == t.cols && (smallest < maxNorm * 1.0e-4f || scale * Root(smallest) < 0.02f)) { AccurateSquare(b, offset, scale, meta); continue; }
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
        t = data; r8 = (t.rows + 7) / 8 * 8; k8 = (t.cols + 7) / 8 * 8; s = r8 + k8; stride = t.stride; rowLength = s + 8;
        if constexpr (PAD > 0) {
            r8 = PAD; k8 = PAD; s = 2 * PAD; stride = 2 * PAD + 8; rowLength = stride;
            if constexpr (!TAIL) { t.m = PAD; t.n = PAD; t.rows = PAD; t.cols = PAD; }
        }
        uint32_t bufferStride = t.parallel ? (rowLength + 63) / 64 * 64 : stride;
        uint32_t size = (t.local ? t.cols * stride : 0) + 4 * bufferStride + 4 * r8 + 8 * t.rows + k8 + 32;
        pipe->InitBuffer(mem, size * 4);
        auto all = mem.Get<float>(); uint32_t off = 0;
        matrix = all; off += t.local ? t.cols * stride : 0;
        x = all[off]; off += bufferStride; y = all[off]; off += bufferStride; u = all[off]; off += bufferStride; v = all[off]; off += bufferStride;
        prod = all[off]; off += r8; tmp = all[off]; off += r8; output = all[off]; off += r8;
        red = all[off]; off += 32; gather = all[off]; off += 8 * t.rows;
        norms = all[off];
        if (t.parallel || t.localInverse) {
            pipe->InitBuffer(vectorPrecision, 8 * s * sizeof(float));
            auto area = vectorPrecision.Get<float>();
            for (uint32_t i = 0; i < 8; ++i) calc[i] = area[i * s];
        }
        if (t.localInverse) {
            pipe->InitBuffer(inverseLow, (TAIL ? k8 : t.cols) * stride * sizeof(float));
            lowMatrix = inverseLow.Get<float>();
        }
        if (t.localInverse || (t.parallel && t.m == t.n)) {
            uint32_t areaSize = t.inverseTile * stride;
            pipe->InitBuffer(batchPrecision, 12 * areaSize * sizeof(float));
            auto area = batchPrecision.Get<float>();
            for (uint32_t i = 0; i < 8; ++i) batchCalc[i] = area[i * areaSize];
            batchU = area[8 * areaSize]; batchV = area[9 * areaSize]; batchH = area[10 * areaSize]; batchL = area[11 * areaSize];
            if (t.parallel) {
                pipe->InitBuffer(parallelElimination, 2 * areaSize * sizeof(float));
                tileHigh = parallelElimination.Get<float>(); tileLow = tileHigh[areaSize];
            }
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
            if (t.localInverse && LocalInverse(offset)) continue;
            if (!t.local) work.SetGlobalBuffer(workBase + uint64_t(b) * t.cols * stride);
            float scale = 0;
            for (uint32_t i = 0; i < t.cols; ++i) {
                Duplicate(x, 0.0f, rowLength); PipeBarrier<PIPE_V>();
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
            float roundoffFloor = total * 1.0e-14f;
            discard = discard < roundoffFloor ? discard : roundoffFloor;
            bool cached = false;
            for (uint32_t sweep = 0; sweep < 40; ++sweep) {
                uint32_t changed = 0;
                for (uint32_t p = 0; p < t.cols; ++p) {
                    ReadRow(x, p);
                    for (uint32_t q = p + 1; q < t.cols; ++q) {
                        ReadRow(y, q);
                        float a, d, h;
                        if (cached) { h = Dot(x, y); a = x.GetValue(s); d = y.GetValue(s); }
                        else { a = Dot(x, x); d = Dot(y, y); h = Dot(x, y); }
                        if (a <= discard || d <= discard || h == 0 || h * h <= 1.0e-12f * (a * d)) continue;
                        float tau = (d - a) / (2 * h);
                        float at = AbsScalar(tau);
                        float z = (tau < 0 ? -1.0f : 1.0f) / (at + Root(1 + at * at));
                        float c = 1.0f / Root(1 + z * z), sn = c * z;
                        Muls(u, x, c, s); Muls(v, y, -sn, s); PipeBarrier<PIPE_V>(); Add(u, u, v, s); PipeBarrier<PIPE_V>();
                        Muls(v, x, sn, s); Muls(y, y, c, s); PipeBarrier<PIPE_V>(); Add(y, y, v, s); PipeBarrier<PIPE_V>();
                        if (cached) {
                            float na = Dot(u, u), nd = Dot(y, y);
                            u.SetValue(s, na); y.SetValue(s, nd); SV();
                        }
                        DataCopy(x, u, stride); PipeBarrier<PIPE_V>(); SaveRow(y, q); ++changed;
                    }
                    SaveRow(x, p);
                }
                if (changed == 0) break;
                if (!cached && changed < 0.35f * t.cols * (t.cols - 1)) { CacheNorms(); cached = true; }
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
template <typename DT_A, bool COMPACT, bool TAIL, bool WAVE, bool LOCAL>
__global__ __aicore__ void linalg_pinv(GM_ADDR A, GM_ADDR A_pinv, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(LinalgPinvTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData, data, tiling);
    TPipe pipe;
    if constexpr (COMPACT) {
        data.parallel = 0; data.local = 1; data.localInverse = 1;
        if (data.rows > 8 && data.rows <= 16) {
            Pinv<TAIL, 16> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
        } else {
            Pinv<TAIL> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
        }
    } else {
        if constexpr (WAVE) { data.parallel = 1; data.local = 0; data.localInverse = 0; }
        if constexpr (LOCAL) { data.parallel = 0; data.groupCount = 0; data.localInverse = 1; }
        StablePinv<WAVE, LOCAL> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
    }
}
