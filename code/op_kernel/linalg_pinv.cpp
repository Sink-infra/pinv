#include "kernel_operator.h"
#include "linalg_pinv_tiling.h"
#include "tiling_key_linalg_pinv.h"
using namespace AscendC;
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
            for (uint32_t i = 0; i < cores; ++i) {
                float value = gather.GetValue(i * 8);
                if (value > best) { best = value; pivot = static_cast<uint32_t>(static_cast<int32_t>(gather.GetValue(i * 8 + 1))); }
            }
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
            for (uint32_t row = 0; row < t.inverseTile; ++row) {
                Duplicate(batchU[row * stride], 0.0f, stride); PipeBarrier<PIPE_V>();
                DataCopy(batchU[row * stride], u, rowLength);
            }
            PipeBarrier<PIPE_V>();
            float nextBest = 0.0f; uint32_t nextRow = core;
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
        if (!(inverseTotal > 0.0f && total * inverseTotal < 3.6e9f)) return false;
        if (certify && !(inverseTotal * scaledCutoff * scaledCutoff < 81.0f)) return false;
        for (uint32_t row = 0; row < owned; ++row) {
            VS(); uint32_t target = static_cast<uint32_t>(static_cast<int32_t>(norms.GetValue(core + row * cores))) - 1;
            Muls(output, tileHigh[row * stride + r8], 1.0f / power, r8); PipeBarrier<PIPE_V>();
            Store(result[offset + target * t.m], output, t.m);
        }
        x = bx; y = by; u = bu; v = bv;
        SyncAll(); return true;
    }
    __aicore__ inline bool ResidentSquare(uint32_t b, uint32_t offset, float scale,
                                          GlobalTensor<float> meta, bool certify, float cutoff) {
        if (QuickResidentSquare(b, offset, scale, meta, certify, cutoff)) return true;
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
            for (uint32_t i = 0; i < cores; ++i) {
                float value = gather.GetValue(i * 8);
                if (value > best) { best = value; pivot = static_cast<uint32_t>(static_cast<int32_t>(gather.GetValue(i * 8 + 1))); }
            }
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
                                         bool certify = false, float cutoff = 0.0f) {
        if (t.resident) return ResidentSquare(b, offset, scale, meta, certify, cutoff);
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
    __aicore__ inline bool QuickLocalInverse(uint32_t offset) {
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
            if (p) { Duplicate(prod, 0.0f, p); PipeBarrier<PIPE_V>(); }
            ReduceMax(red, prod, tmp, t.cols, true); VS();
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
            Muls(norms, norms, -1.0f, k8); PipeBarrier<PIPE_V>(); VS();
            Brcb(lowMatrix, norms, t.cols / 8, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)});
            for (uint32_t i = t.cols / 8 * 8; i < t.cols; ++i) Duplicate(lowMatrix[i * stride], norms.GetValue(i), 8);
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
    __aicore__ inline bool LocalInverse(uint32_t offset) {
        auto bx = x, by = y, bu = u, bv = v;
        bool accepted = QuickLocalInverse(offset);
        if (!accepted) accepted = LocalInverseBody(offset, bx, by, bu, bv);
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
            pipe->InitBuffer(inverseLow, t.cols * stride * sizeof(float));
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
                pipe->InitBuffer(parallelElimination, 2 * residentSize * sizeof(float));
                tileHigh = parallelElimination.Get<float>(); tileLow = tileHigh[residentSize];
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
template <bool TAIL>
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
        auto indices = gather.ReinterpretCast<uint32_t>();
        if (vectorQuick) {
            for (uint32_t i = 0; i < k8; ++i) indices.SetValue(i, i < t.cols ? i * stride * sizeof(float) : 0);
            Duplicate(norms, 0.0f, k8); PipeBarrier<PIPE_V>(); SV();
        }
        for (uint32_t p = 0; p < t.cols; ++p) {
            uint32_t pivot = p; float best = 0.0f;
            if (vectorQuick) {
                Gather(norms, matrix, indices, p * sizeof(float), t.cols); PipeBarrier<PIPE_V>();
                Abs(prod, norms, k8); PipeBarrier<PIPE_V>();
                uint64_t pivotMask[2] = {((uint64_t(1) << t.cols) - 1) & ~((uint64_t(1) << p) - 1), 0};
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
                        VS();
                        Brcb(lowMatrix, norms, t.cols / 8, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)});
                        for (uint32_t row = t.cols / 8 * 8; row < t.cols; ++row) Duplicate(lowMatrix[row * stride], norms.GetValue(row), 8);
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
        if (vectorQuick) { Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>(); }
        if (compact) {
            Add(matrix, matrix, lowMatrix, count); PipeBarrier<PIPE_V>(); Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
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
            SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
            SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
            DataCopyPad(result[offset], matrix[r8], DataCopyExtParams{static_cast<uint16_t>(t.n), t.m * 4, (stride - r8) / 8, 0, 0});
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
        bool accepted = stride <= 64 ? CompactInverseBody(offset, bx, by, bu, bv) : AccurateLocalInverseBody(offset, bx, by, bu, bv);
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
template <typename DT_A, bool COMPACT, bool TAIL>
__global__ __aicore__ void linalg_pinv(GM_ADDR A, GM_ADDR A_pinv, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(LinalgPinvTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData, data, tiling);
    TPipe pipe;
    if constexpr (COMPACT) {
        data.parallel = 0; data.local = 1; data.localInverse = 1;
        Pinv<TAIL> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
    } else {
        StablePinv op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
    }
}
