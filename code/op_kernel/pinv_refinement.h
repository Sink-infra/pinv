#pragma once
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & 32768UL)
    __aicore__ inline bool RefineResident(uint32_t b, uint32_t offset, float power,
                                          float total, float inverseTotal, GlobalTensor<float> meta,
                                          bool certify, float cutoff) {
        uint32_t core = GetBlockIdx(), cores = GetBlockNum();
        if constexpr (WAVE) if ((t.cols + cores - 1) / cores * r8 > lowElements) return false;
        uint32_t owned = (t.cols + cores - 1 - core) / cores;
        LocalTensor<float> savedCalc[8];
        for (uint32_t i = 0; i < 8; ++i) savedCalc[i] = batchCalc[i];
        LocalTensor<float> savedU = batchU, savedV = batchV, savedH = batchH, savedL = batchL;
        uint32_t savedTile = t.inverseTile;
        uint32_t repacked = InstallRefinePacking((t.cols + cores - 1) / cores);
        if (repacked) t.inverseTile = repacked;
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
        // The inverse rows are read once per column; batching the reads turns t.cols
        // single-row transfers (each with its own pipeline flush) into t.cols/span of them.
        uint32_t active = (t.cols + cores - 1) / cores;
        uint32_t cacheSpan = (active * r8 + 7) / 8 * 8;
        uint32_t span = 8 * s > cacheSpan ? (8 * s - cacheSpan) / r8 : 0;
        if (span > 8) span = 8;
        auto block = span > 1 ? cache[cacheSpan] : u;
        if (span < 1) span = 1;
        // The factor column is gathered and expanded on the vector pipe. Reading those
        // factors with the scalar unit cost one UB access per owned row per column plus
        // the pipeline flush that made the access legal, which profiling showed to be a
        // third of this kernel's time.
        auto rowIndex = prod.ReinterpretCast<uint32_t>();
        VS();
        for (uint32_t row = 0; row < t.inverseTile; ++row) rowIndex.SetValue(row, row * r8 * sizeof(float));
        SV();
        bool batchedSplit = r8 % 64 == 0 && t.inverseTile > 1;
        uint32_t sourceSpan = batchedSplit && span > t.inverseTile ? t.inverseTile : span;
        auto sourceHigh = batchU, sourceLow = batchV;
        uint32_t prepRows = (owned + 7) / 8 * 8;
        bool cachedFactors = batchedSplit && t.inverseTile >= owned
            && sourceSpan * prepRows <= r8 && sourceSpan * prepRows * 8 <= t.inverseTile * r8;
        auto fullFactors = batchH, highFactors = batchL, lowFactors = batchCalc[1];
        auto prepIndices = gather.ReinterpretCast<uint32_t>();
        if (cachedFactors) {
            VS();
            for (uint32_t entry = 0; entry < sourceSpan * prepRows; ++entry) {
                uint32_t row = entry % prepRows, column = entry / prepRows;
                prepIndices.SetValue(entry, ((row < owned ? row : 0) * r8 + column) * sizeof(float));
            }
            SV();
        }
        float cubeQuantizationBound=0.0f;
        bool cubeResidual=false;
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & 32768UL)
        if constexpr(true){
            if(t.cols==r8 && r8%32==0 && owned>0 && owned<=8 && t.inverseTile>=owned
                && total>0.0f && total<1.0e20f && inverseTotal>0.0f && inverseTotal<1.0e20f){
                float inverseNorm=Root(inverseTotal)*1.001f;
                float inputNorm=Root(total)*1.001f,inverseScale=1.0f;
                while(inverseScale<2.0f*inverseNorm)inverseScale*=2.0f;
                float sumErrorA,sumErrorX;
                PinvCubeBuildResidual(reinterpret_cast<GM_ADDR>(workBase),t,result,offset,cache,
                    tileHigh,tileLow,batchCalc,owned,core,cores,inverseScale,sumErrorA,sumErrorX);
                float deltaA=Root(sumErrorA)*4.0f*1.7763568394002505e-15f*1.002f;
                float deltaX=Root(sumErrorX)*inverseScale*1.7763568394002505e-15f*1.002f;
                cubeQuantizationBound=deltaA*inverseNorm+deltaX*inputNorm+deltaA*deltaX;
                cubeResidual=true;
            }
        }
#endif
        if(!cubeResidual){
        for (uint32_t base = 0; base < t.cols; base += sourceSpan) {
            uint32_t chunk = t.cols - base < sourceSpan ? t.cols - base : sourceSpan;
            SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
            if (span > 1) {
                DataCopyPad(block, result[offset + base * t.m],
                            DataCopyExtParams{static_cast<uint16_t>(chunk), t.m * 4, 0, 0, 0},
                            DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - t.m % 8) % 8), 0});
                SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
            } else Load(block, result[offset + base * t.m], t.m);
            if (cachedFactors) {
                uint32_t entries = chunk * prepRows;
                Gather(v, cache, prepIndices, base * sizeof(float), entries); PipeBarrier<PIPE_V>();
                Muls(prod, v, 4097.0f, entries); PipeBarrier<PIPE_V>();
                Sub(tmp, prod, v, entries); PipeBarrier<PIPE_V>();
                Sub(prod, prod, tmp, entries); PipeBarrier<PIPE_V>();
                Sub(tmp, v, prod, entries); PipeBarrier<PIPE_V>();
                Brcb(fullFactors, v, entries / 8, {1, 8});
                Brcb(highFactors, prod, entries / 8, {1, 8});
                Brcb(lowFactors, tmp, entries / 8, {1, 8}); PipeBarrier<PIPE_V>();
            }
            if (batchedSplit) {
                uint32_t elements = chunk * r8;
                Muls(sourceHigh, block, 4097.0f, elements); PipeBarrier<PIPE_V>();
                Sub(sourceLow, sourceHigh, block, elements); PipeBarrier<PIPE_V>();
                Sub(sourceHigh, sourceHigh, sourceLow, elements); PipeBarrier<PIPE_V>();
                Sub(sourceLow, block, sourceHigh, elements); PipeBarrier<PIPE_V>();
            }
            for (uint32_t j = 0; j < chunk; ++j) {
                uint32_t k = base + j;
                auto source = block[j * r8];
                if (batchedSplit) { batchU = sourceHigh[j * r8]; batchV = sourceLow[j * r8]; }
                else {
                Muls(batchU, source, 4097.0f, r8); PipeBarrier<PIPE_V>();
                Sub(batchV, batchU, source, r8); PipeBarrier<PIPE_V>();
                Sub(batchU, batchU, batchV, r8); PipeBarrier<PIPE_V>();
                Sub(batchV, source, batchU, r8); PipeBarrier<PIPE_V>();
                }
                if (cachedFactors) {
                    batchH = fullFactors[j * prepRows * 8];
                    batchL = highFactors[j * prepRows * 8];
                    batchCalc[1] = lowFactors[j * prepRows * 8];
                    x = tileHigh; y = tileLow;
                    RefineBroadcastEliminateFixed(owned, source, (k & 7) == 7 || k + 1 == t.cols, true);
                } else {
                for (uint32_t first = 0; first < owned; first += t.inverseTile) {
                    uint32_t rows = owned - first < t.inverseTile ? owned - first : t.inverseTile;
                    Gather(v, cache, rowIndex, (first * r8 + k) * sizeof(float), rows); PipeBarrier<PIPE_V>();
                    x = tileHigh[first * r8]; y = tileLow[first * r8];
                    RefineBroadcastEliminate(rows, source, (k & 7) == 7 || k + 1 == t.cols);
                }
                }
            }
        }
        }
        batchU = sourceHigh; batchV = sourceLow;
        batchH = fullFactors; batchL = highFactors; batchCalc[1] = lowFactors;
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
        float gd = (r8 % 64 == 0 ? 128.0f : 64.0f) * t.cols * unit * unit;
        float gamma = (2.0f * t.cols * unit) / (1.0f - 2.0f * t.cols * unit);
        float er = gd * af * yf + 4.0f * unit * rf + cubeQuantizationBound, denominator = 1.0f - rf - er;
        if (!(denominator > 0.0f)) { RestoreScratch(savedCalc, savedU, savedV, savedH, savedL, savedTile); return false; }
        Sqrt(gather, gather, r8); Sqrt(v, v, r8); PipeBarrier<PIPE_V>();
        Muls(gather, gather, 1.001f, r8); Muls(v, v, 1.001f, r8); PipeBarrier<PIPE_V>();
        Muls(output, v, gd * af, r8); Muls(prod, gather, unit, r8); PipeBarrier<PIPE_V>(); Add(output, output, prod, r8); PipeBarrier<PIPE_V>();
        if(cubeResidual){Adds(output,output,cubeQuantizationBound,r8);PipeBarrier<PIPE_V>();}
        Duplicate(tileLow, 0.0f, owned * r8); PipeBarrier<PIPE_V>();
        VS();
        for (uint32_t row = 0; row < t.inverseTile; ++row) rowIndex.SetValue(row, row * r8 * sizeof(float));
        SV();
        for (uint32_t base = 0; base < t.cols; base += span) {
            uint32_t chunk = t.cols - base < span ? t.cols - base : span;
            SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
            if (span > 1) {
                DataCopyPad(block, residual[base * r8],
                            DataCopyExtParams{static_cast<uint16_t>(chunk), t.n * 4, 0, 0, 0},
                            DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - t.n % 8) % 8), 0});
                SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
            } else Load(block, residual[base * r8], t.n);
            for (uint32_t j = 0; j < chunk; ++j) {
                uint32_t k = base + j;
                auto source = block[j * r8];
                Gather(v, cache, rowIndex, k * sizeof(float), owned); PipeBarrier<PIPE_V>();
                Brcb(batchCalc[0], v, (owned + 7) / 8, {1, 8});
                PipeBarrier<PIPE_V>();
                for (uint32_t col = 0; col < r8; col += 64) {
                    uint32_t mask = r8 - col < 64 ? r8 - col : 64;
                    MulAddDst(tileLow[col], source[col], batchCalc[0], static_cast<uint64_t>(mask), static_cast<uint8_t>(owned), {1, 1, 0, static_cast<uint8_t>(r8 / 8), 0, 1});
                    PipeBarrier<PIPE_V>();
                }
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
        if (!(valid && correctedTotal > 0.0f && (!certify || correctedTotal * scaledCutoff * scaledCutoff < 81.0f))) {
            RestoreScratch(savedCalc, savedU, savedV, savedH, savedL, savedTile); return false;
        }
        for (uint32_t row = 0; row < owned; ++row) {
            Muls(output, tileHigh[row * r8], 1.0f / power, r8); PipeBarrier<PIPE_V>();
            Store(result[offset + (core + row * cores) * t.m], output, t.m);
        }
        RestoreScratch(savedCalc, savedU, savedV, savedH, savedL, savedTile);
        SyncAll(); return true;
    }
#else
    __aicore__ inline bool RefineResident(uint32_t b, uint32_t offset, float power,
                                          float total, float inverseTotal, GlobalTensor<float> meta,
                                          bool certify, float cutoff) {
        uint32_t core = GetBlockIdx(), cores = GetBlockNum();
        if constexpr (WAVE) if ((t.cols + cores - 1) / cores * r8 > lowElements) return false;
        uint32_t owned = (t.cols + cores - 1 - core) / cores;
        LocalTensor<float> savedCalc[8];
        for (uint32_t i = 0; i < 8; ++i) savedCalc[i] = batchCalc[i];
        LocalTensor<float> savedU = batchU, savedV = batchV, savedH = batchH, savedL = batchL;
        uint32_t savedTile = t.inverseTile;
        uint32_t repacked = InstallRefinePacking((t.cols + cores - 1) / cores);
        if (repacked) t.inverseTile = repacked;
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
        // The inverse rows are read once per column; batching the reads turns t.cols
        // single-row transfers (each with its own pipeline flush) into t.cols/span of them.
        uint32_t active = (t.cols + cores - 1) / cores;
        uint32_t cacheSpan = (active * r8 + 7) / 8 * 8;
        uint32_t span = 8 * s > cacheSpan ? (8 * s - cacheSpan) / r8 : 0;
        if (span > 8) span = 8;
        auto block = span > 1 ? cache[cacheSpan] : u;
        if (span < 1) span = 1;
        // The factor column is gathered and expanded on the vector pipe. Reading those
        // factors with the scalar unit cost one UB access per owned row per column plus
        // the pipeline flush that made the access legal, which profiling showed to be a
        // third of this kernel's time.
        auto rowIndex = prod.ReinterpretCast<uint32_t>();
        VS();
        for (uint32_t row = 0; row < t.inverseTile; ++row) rowIndex.SetValue(row, row * r8 * sizeof(float));
        SV();
        bool batchedSplit = r8 % 64 == 0 && t.inverseTile > 1;
        uint32_t sourceSpan = batchedSplit && span > t.inverseTile ? t.inverseTile : span;
        auto sourceHigh = batchU, sourceLow = batchV;
        uint32_t prepRows = (owned + 7) / 8 * 8;
        bool cachedFactors = batchedSplit && t.inverseTile >= owned
            && sourceSpan * prepRows <= r8 && sourceSpan * prepRows * 8 <= t.inverseTile * r8;
        auto fullFactors = batchH, highFactors = batchL, lowFactors = batchCalc[1];
        auto prepIndices = gather.ReinterpretCast<uint32_t>();
        if (cachedFactors) {
            VS();
            for (uint32_t entry = 0; entry < sourceSpan * prepRows; ++entry) {
                uint32_t row = entry % prepRows, column = entry / prepRows;
                prepIndices.SetValue(entry, ((row < owned ? row : 0) * r8 + column) * sizeof(float));
            }
            SV();
        }
        for (uint32_t base = 0; base < t.cols; base += sourceSpan) {
            uint32_t chunk = t.cols - base < sourceSpan ? t.cols - base : sourceSpan;
            SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
            if (span > 1) {
                DataCopyPad(block, result[offset + base * t.m],
                            DataCopyExtParams{static_cast<uint16_t>(chunk), t.m * 4, 0, 0, 0},
                            DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - t.m % 8) % 8), 0});
                SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
            } else Load(block, result[offset + base * t.m], t.m);
            if (cachedFactors) {
                uint32_t entries = chunk * prepRows;
                Gather(v, cache, prepIndices, base * sizeof(float), entries); PipeBarrier<PIPE_V>();
                Muls(prod, v, 4097.0f, entries); PipeBarrier<PIPE_V>();
                Sub(tmp, prod, v, entries); PipeBarrier<PIPE_V>();
                Sub(prod, prod, tmp, entries); PipeBarrier<PIPE_V>();
                Sub(tmp, v, prod, entries); PipeBarrier<PIPE_V>();
                Brcb(fullFactors, v, entries / 8, {1, 8});
                Brcb(highFactors, prod, entries / 8, {1, 8});
                Brcb(lowFactors, tmp, entries / 8, {1, 8}); PipeBarrier<PIPE_V>();
            }
            if (batchedSplit) {
                uint32_t elements = chunk * r8;
                Muls(sourceHigh, block, 4097.0f, elements); PipeBarrier<PIPE_V>();
                Sub(sourceLow, sourceHigh, block, elements); PipeBarrier<PIPE_V>();
                Sub(sourceHigh, sourceHigh, sourceLow, elements); PipeBarrier<PIPE_V>();
                Sub(sourceLow, block, sourceHigh, elements); PipeBarrier<PIPE_V>();
            }
            for (uint32_t j = 0; j < chunk; ++j) {
                uint32_t k = base + j;
                auto source = block[j * r8];
                if (batchedSplit) { batchU = sourceHigh[j * r8]; batchV = sourceLow[j * r8]; }
                else {
                Muls(batchU, source, 4097.0f, r8); PipeBarrier<PIPE_V>();
                Sub(batchV, batchU, source, r8); PipeBarrier<PIPE_V>();
                Sub(batchU, batchU, batchV, r8); PipeBarrier<PIPE_V>();
                Sub(batchV, source, batchU, r8); PipeBarrier<PIPE_V>();
                }
                if (cachedFactors) {
                    batchH = fullFactors[j * prepRows * 8];
                    batchL = highFactors[j * prepRows * 8];
                    batchCalc[1] = lowFactors[j * prepRows * 8];
                    x = tileHigh; y = tileLow;
                    RefineBroadcastEliminateFixed(owned, source, (k & 7) == 7 || k + 1 == t.cols, true);
                } else {
                for (uint32_t first = 0; first < owned; first += t.inverseTile) {
                    uint32_t rows = owned - first < t.inverseTile ? owned - first : t.inverseTile;
                    Gather(v, cache, rowIndex, (first * r8 + k) * sizeof(float), rows); PipeBarrier<PIPE_V>();
                    x = tileHigh[first * r8]; y = tileLow[first * r8];
                    RefineBroadcastEliminate(rows, source, (k & 7) == 7 || k + 1 == t.cols);
                }
                }
            }
        }
        batchU = sourceHigh; batchV = sourceLow;
        batchH = fullFactors; batchL = highFactors; batchCalc[1] = lowFactors;
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
        float gd = (r8 % 64 == 0 ? 128.0f : 64.0f) * t.cols * unit * unit;
        float gamma = (2.0f * t.cols * unit) / (1.0f - 2.0f * t.cols * unit);
        float er = gd * af * yf + 4.0f * unit * rf, denominator = 1.0f - rf - er;
        if (!(denominator > 0.0f)) { RestoreScratch(savedCalc, savedU, savedV, savedH, savedL, savedTile); return false; }
        Sqrt(gather, gather, r8); Sqrt(v, v, r8); PipeBarrier<PIPE_V>();
        Muls(gather, gather, 1.001f, r8); Muls(v, v, 1.001f, r8); PipeBarrier<PIPE_V>();
        Muls(output, v, gd * af, r8); Muls(prod, gather, unit, r8); PipeBarrier<PIPE_V>(); Add(output, output, prod, r8); PipeBarrier<PIPE_V>();
        Duplicate(tileLow, 0.0f, owned * r8); PipeBarrier<PIPE_V>();
        VS();
        for (uint32_t row = 0; row < t.inverseTile; ++row) rowIndex.SetValue(row, row * r8 * sizeof(float));
        SV();
        for (uint32_t base = 0; base < t.cols; base += span) {
            uint32_t chunk = t.cols - base < span ? t.cols - base : span;
            SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
            if (span > 1) {
                DataCopyPad(block, residual[base * r8],
                            DataCopyExtParams{static_cast<uint16_t>(chunk), t.n * 4, 0, 0, 0},
                            DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - t.n % 8) % 8), 0});
                SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
            } else Load(block, residual[base * r8], t.n);
            for (uint32_t j = 0; j < chunk; ++j) {
                uint32_t k = base + j;
                auto source = block[j * r8];
                Gather(v, cache, rowIndex, k * sizeof(float), owned); PipeBarrier<PIPE_V>();
                Brcb(batchCalc[0], v, (owned + 7) / 8, {1, 8});
                PipeBarrier<PIPE_V>();
                for (uint32_t col = 0; col < r8; col += 64) {
                    uint32_t mask = r8 - col < 64 ? r8 - col : 64;
                    MulAddDst(tileLow[col], source[col], batchCalc[0], static_cast<uint64_t>(mask), static_cast<uint8_t>(owned), {1, 1, 0, static_cast<uint8_t>(r8 / 8), 0, 1});
                    PipeBarrier<PIPE_V>();
                }
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
        if (!(valid && correctedTotal > 0.0f && (!certify || correctedTotal * scaledCutoff * scaledCutoff < 81.0f))) {
            RestoreScratch(savedCalc, savedU, savedV, savedH, savedL, savedTile); return false;
        }
        for (uint32_t row = 0; row < owned; ++row) {
            Muls(output, tileHigh[row * r8], 1.0f / power, r8); PipeBarrier<PIPE_V>();
            Store(result[offset + (core + row * cores) * t.m], output, t.m);
        }
        RestoreScratch(savedCalc, savedU, savedV, savedH, savedL, savedTile);
        SyncAll(); return true;
    }
#endif
