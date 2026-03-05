# zenwebp Development Log

Historical investigation notes, resolved bugs, and completed experiments.
Moved from CLAUDE.md on 2026-02-06 to keep the main file lean.

## Resolved Bugs

### I4 Mode Context Bug (2026-02-02, FIXED)

**Bug:** Edge blocks in I4 mode selection were using hardcoded DC (0) context instead of
cross-macroblock context from `top_b_pred` and `left_b_pred`. This caused suboptimal mode
selection for edge blocks because the mode cost tables are context-dependent.

**Root cause:** The `top_b_pred` and `left_b_pred` arrays were only being updated during
header writing (after all mode selection was complete), not during the encoding pass.

**Fix (2 parts):**
1. `mode_selection.rs`: Use `self.top_b_pred[mbx * 4 + sbx]` and `self.left_b_pred[sby]`
   for edge block context instead of hardcoded 0
2. `mod.rs`: Update `top_b_pred` and `left_b_pred` immediately after `choose_macroblock_info()`
   returns, so the next macroblock sees correct context

**Results:**
- Mode distribution shifted: -257 DC modes, +93 VE, +73 HE, +922 TM
- CID22 corpus (248 images): 0.999x → 0.990x (zenwebp now 1% smaller than libwebp)

### Multi-pass Probability Signaling Bug (2026-02-02, FIXED)

**Bug:** Multi-pass encoding (methods 5-6) produced corrupt output that decoded incorrectly.
99.9% of Y plane pixels differed between method 4 and method 5 decoded output.

**Root cause:** Probability update signaling compared against wrong baseline.
- In multi-pass, `token_probs` was set to pass 0's updated values at start of pass 1
- `compute_updated_probabilities` compared new probabilities against pass 0 values
- Finding no savings (pass 1 ≈ pass 0), it set `updated_probs = None`
- Header signaled 0 updates (comparing against pass 0, not defaults)
- Decoder started with `COEFF_PROBS` (defaults) but encoder used pass 0 probs
- Mismatch caused garbage decoding

**Fix:** Both `encode_updated_token_probabilities` and `compute_updated_probabilities`
now always compare against `COEFF_PROBS` (decoder defaults).

### TokenType Fix (2026-01-24, FIXED)

The TokenType enum had I16DC and I16AC values swapped compared to libwebp.
Fix reduced file sizes by ~2%.

### Trellis sign bit double-counting (2026-02-02, FIXED)

`level_cost_fast` and `level_cost_with_table` were adding 256 (sign bit cost)
to non-zero levels, but VP8_LEVEL_FIXED_COSTS already includes this. Fix improved
m4 by ~0.4%.

### SNS c_base Bug (2026-02-03, FIXED)

`compute_segment_quant()` computed c_base from base_quant instead of quality directly,
double-applying segment alpha transformation. Fix: CID22 Q90 1.0126x → 1.0060x.

### webpx Preset Bug (2026-02-01, RESOLVED)

webpx 0.1.2 `to_libwebp()` overwrote preset-specific filter values with defaults.
Fixed in webpx 0.1.3 with `Option<u8>` fields.

### Method Realignment (2026-02-04, RESOLVED)

Apparent "regression" from 0.9888x to 1.0099x at method 4 was NOT a bug. Commit `47873f5`
intentionally moved trellis from m4 to m5 to align with libwebp's RD optimization levels.

## Completed Investigations

### Multi-pass Re-quantization Experiment (2026-02-02, ABANDONED)

**Tested approaches:**
1. **Token re-recording:** ZERO benefit - identical tokens → identical stats → identical output.
2. **Re-quantization from stored raw DCT:** Files LARGER (+0.82%), quality WORSE.

**Root cause:** Raw DCT coefficients depend on predictions which depend on reconstructed
pixels from THIS pass's quantized coefficients. Reusing pass 0's DCT in pass 1 is wrong
because predictions change.

**How libwebp actually does multi-pass:** Full re-encoding each pass. `VP8IteratorImport()`
imports SOURCE pixels, `VP8Decimate()` computes prediction from boundary buffers,
`VP8IteratorSaveBoundary()` saves RECONSTRUCTED pixels.

### Multi-pass Implementation (2026-02-02, COMPLETED)

Multi-pass now matches libwebp's VP8EncTokenLoop. Result: NO compression benefit
without quality search. Both encoders produce ~0.1-0.3% LARGER files with more passes.
Multi-pass is only useful for target_size convergence (quality search).

### SNS Quality-Size Tradeoff Investigation (2026-02-01, FIXED)

Our SNS produced steeper quality degradation than libwebp's. Root cause was the c_base
bug (see above). After fix, Q90 production is 1.006x (near parity).

### VP8BitReader Success (2026-01-22)

Replaced ArithmeticDecoder with libwebp's VP8GetBitAlt algorithm. 16% speedup.
Key: `range - 1` representation, `leading_zeros()` → LZCNT, simpler split calc.

### Loop Filter Optimization (2026-01-23, RESOLVED)

Normal filter now SIMD for both V and H edges (luma + chroma). Remaining ~6M extra
instructions vs libwebp is from per-pixel threshold checks, not filter math.

### libwebp-rs Mechanical Translation Comparison (2026-01-22)

Both Rust decoders (ours and libwebp-rs) have similar ~2.5x slowdown vs C despite
different bottleneck distributions. libwebp-rs's inner edge `filter_loop24` (scalar,
24% of time) mirrors our loop filter bottleneck.

### TokenType Fix and File Size Investigation (2026-01-24)

Remaining 4.5% gap investigation verified these MATCH libwebp: lambda calculations,
VP8_LEVEL_FIXED_COSTS, VP8_FREQ_SHARPENING, VP8_WEIGHT_TRELLIS, RD score formula,
trellis distortion. The 4.5% gap in diagnostic mode disappears in production settings.

SIMD parity verified — outputs identical with and without `simd` feature.

### I4 Mode Path Investigation (2026-02-01)

File size gap is in I4 mode path, not I16. Method 0 (I16 only): we're 9-19% smaller.
Methods 2-6 (with I4): we're 9-14% larger. Fixed by sharpen/zthresh in quantize_coeff
(2026-02-02), token buffer (commit 9625d4e), and I4 penalty tuning.

### I4 Over-Selection Investigation (2026-02-03)

zenwebp selects I4 for ~5% more MBs than libwebp (73.4% vs 68.3%). Our I16 is MORE
efficient, our I4 is LESS efficient. BMODE_COST sweep showed no monotonic improvement.
Root cause is subtle coefficient cost estimation difference. Current status: CID22 1.0099x.

Verified components that MATCH: lambdas, BMODE_COST, VP8_FIXED_COSTS_I4,
VP8_LEVEL_FIXED_COSTS, LevelCosts, RD score formula, SSE calculation, context tracking.

### I4 Encoding Efficiency Decomposition Plan (2026-02-02)

Six-stage pipeline comparison plan (prediction → quantization → mode selection →
token encoding → probability updates → arithmetic encoding). Files involved:
`prediction.rs`, `quantize.rs`, `mode_selection.rs`, `residuals.rs`, `cost.rs`, `arithmetic.rs`.

Step 2 (simple quantization): VERIFIED match. Step 3 (probability tables): mid-stream
update helps, level_costs recalc hurts. Step 1 (trellis/I4 penalty): fixed with
`i4_penalty = 3000 * lambda_mode`.

### I4 Coefficient Level Analysis (2026-02-02)

Same-mode I4 blocks: 57.9% exact match, total |level| sum 2.7% higher, non-zero count
1.3% higher. Our trellis slightly less aggressive at zeroing. Inherent to trellis tuning;
corpus results show overall parity.

### Encoder Callgrind Analysis (2026-01-23)

Fair comparisons: Our m2 vs libwebp m2 = 0.940x (6% smaller). Our m4 vs libwebp m6
(both trellis) = 1.045x (4.5% larger). We're faster than libwebp with trellis (65ms vs 75ms)
but produce larger files.

Cache behavior: Our D1 miss rate 0.1% vs libwebp 0.3% (better).

### Detailed Callgrind/Cachegrind Decoder Analysis (2026-01-23)

Per-decode: 72.2M instructions (improved from 78.7M). Memory reads 1.8x, writes 1.8x,
but D1 read miss 0.33% vs libwebp 0.70% (better).

Root causes of gap: loop filter overhead (2.5x), coefficient reading (+17%),
function call overhead (Rust abstractions vs C inline macros).

## Completed Design Decisions

### API Convergence (2026-02-06, COMPLETE)

All tasks complete. Three-layer pattern: `LossyConfig`/`LosslessConfig` → `EncodeRequest<'a>` → output.

Key decisions made:
- Split `EncoderConfig` into `LossyConfig` / `LosslessConfig`
- `EncodeRequest::encode()` (one-shot), `encode_into()`, `encode_to()`
- `with_` prefix for builder setters, bare-name for getters
- `PixelLayout` enum (was `ColorType`), supports Rgba8/Bgra8
- `ImageInfo::from_bytes()` probe with `PROBE_BYTES` constant
- Two-phase decoder: `build()` → `info()` → `decode()`
- `Limits` on both encode and decode sides
- `ImageMetadata` struct factored out of request
- `&dyn Stop` cancellation, `#[non_exhaustive]` errors, `At<>` wrapping

### Why Streaming Encoding is N/A for WebP (2026-02-06)

WebP encoding algorithms require the entire image before encoding can start.
A streaming/push-based API would use MORE memory, not less.

- **VP8L (Lossless):** LZ77 backward refs, palette detection, predictor transforms, and
  Huffman coding all need the complete image. Cannot encode incrementally.
- **VP8 (Lossy):** Segmentation, SNS, filter strength, and RD optimization all analyze
  the full image. Row-by-row encoding would significantly hurt quality.
- **Animation:** Each frame must be fully encoded. `AnimationEncoder::add_frame()` already
  provides the right frame-by-frame API.

The one-shot `EncodeRequest::encode()` borrows the user's buffer (no copy) and is optimal.
