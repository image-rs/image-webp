//! Backward reference finding using hash chains.
//!
//! Converts image pixels into a stream of literals and LZ77 backward references.
//! Implements multiple LZ77 strategies matching libwebp:
//! - LZ77 Standard: greedy with look-ahead optimization
//! - LZ77 RLE: run-length encoding (distance=1 and distance=xsize)
//! - Strategy selection: tries both, picks best based on histogram entropy
//! - Color cache: estimates optimal cache bits, applies to refs
//! - 2D locality: converts raw distances to plane codes

use alloc::vec::Vec;

use super::color_cache::ColorCache;
use super::cost_model::trace_backwards_optimize;
use super::entropy::estimate_histogram_bits;
use super::hash_chain::HashChain;
use super::histogram::Histogram;
use super::types::{
    BackwardRefs, MAX_LENGTH, MIN_LENGTH, NUM_LENGTH_CODES, NUM_LITERAL_CODES, PixOrCopy,
    argb_alpha, argb_blue, argb_green, argb_red,
};

/// Distance code lookup table for 2D neighborhood.
/// Maps (xoffset, yoffset) pairs to distance codes 1-120.
#[rustfmt::skip]
const DISTANCE_MAP: [(i8, i8); 120] = [
    (0, 1),  (1, 0),  (1, 1),  (-1, 1), (0, 2),  (2, 0),  (1, 2),  (-1, 2),
    (2, 1),  (-2, 1), (2, 2),  (-2, 2), (0, 3),  (3, 0),  (1, 3),  (-1, 3),
    (3, 1),  (-3, 1), (2, 3),  (-2, 3), (3, 2),  (-3, 2), (0, 4),  (4, 0),
    (1, 4),  (-1, 4), (4, 1),  (-4, 1), (3, 3),  (-3, 3), (2, 4),  (-2, 4),
    (4, 2),  (-4, 2), (0, 5),  (3, 4),  (-3, 4), (4, 3),  (-4, 3), (5, 0),
    (1, 5),  (-1, 5), (5, 1),  (-5, 1), (2, 5),  (-2, 5), (5, 2),  (-5, 2),
    (4, 4),  (-4, 4), (3, 5),  (-3, 5), (5, 3),  (-5, 3), (0, 6),  (6, 0),
    (1, 6),  (-1, 6), (6, 1),  (-6, 1), (2, 6),  (-2, 6), (6, 2),  (-6, 2),
    (4, 5),  (-4, 5), (5, 4),  (-5, 4), (3, 6),  (-3, 6), (6, 3),  (-6, 3),
    (0, 7),  (7, 0),  (1, 7),  (-1, 7), (5, 5),  (-5, 5), (7, 1),  (-7, 1),
    (4, 6),  (-4, 6), (6, 4),  (-6, 4), (2, 7),  (-2, 7), (7, 2),  (-7, 2),
    (3, 7),  (-3, 7), (7, 3),  (-7, 3), (5, 6),  (-5, 6), (6, 5),  (-6, 5),
    (8, 0),  (4, 7),  (-4, 7), (7, 4),  (-7, 4), (8, 1),  (8, 2),  (6, 6),
    (-6, 6), (8, 3),  (5, 7),  (-5, 7), (7, 5),  (-7, 5), (8, 4),  (6, 7),
    (-6, 7), (7, 6),  (-7, 6), (8, 5),  (7, 7),  (-7, 7), (8, 6),  (8, 7)
];

/// Reverse lookup table: given (yoffset * 16 + 8 - xoffset), get distance code.
#[rustfmt::skip]
const PLANE_TO_CODE_LUT: [u8; 128] = [
    96,  73,  55,  39,  23, 13, 5,  1,  255, 255, 255, 255, 255, 255, 255, 255,
    101, 78,  58,  42,  26, 16, 8,  2,  0,   3,   9,   17,  27,  43,  59,  79,
    102, 86,  62,  46,  32, 20, 10, 6,  4,   7,   11,  21,  33,  47,  63,  87,
    105, 90,  70,  52,  37, 28, 18, 14, 12,  15,  19,  29,  38,  53,  71,  91,
    110, 99,  82,  66,  48, 35, 30, 24, 22,  25,  31,  36,  49,  67,  83,  100,
    115, 108, 94,  76,  64, 50, 44, 40, 34,  41,  45,  51,  65,  77,  95,  109,
    118, 113, 103, 92,  80, 68, 60, 56, 54,  57,  61,  69,  81,  93,  104, 114,
    119, 116, 111, 106, 97, 88, 84, 74, 72,  75,  85,  89,  98,  107, 112, 117
];

/// Convert linear distance to distance code.
pub fn distance_to_plane_code(xsize: usize, dist: usize) -> u32 {
    let yoffset = dist / xsize;
    let xoffset = dist - yoffset * xsize;

    if xoffset <= 8 && yoffset < 8 {
        u32::from(PLANE_TO_CODE_LUT[yoffset * 16 + 8 - xoffset]) + 1
    } else if xoffset + 8 > xsize && yoffset < 7 {
        u32::from(PLANE_TO_CODE_LUT[(yoffset + 1) * 16 + 8 + (xsize - xoffset)]) + 1
    } else {
        (dist + 120) as u32
    }
}

/// Convert distance code back to linear distance.
pub fn plane_code_to_distance(xsize: usize, code: u32) -> usize {
    if code > 120 {
        (code - 120) as usize
    } else {
        let (xoff, yoff) = DISTANCE_MAP[(code - 1) as usize];
        let dist = xoff as i32 + yoff as i32 * xsize as i32;
        if dist < 1 { 1 } else { dist as usize }
    }
}

/// Apply 2D locality transform to backward references.
/// Converts raw distances to plane codes for better compression.
/// Must be called after all LZ77 optimization is complete.
pub fn apply_2d_locality(refs: &mut BackwardRefs, xsize: usize) {
    for token in refs.tokens.iter_mut() {
        if let PixOrCopy::Copy { dist, .. } = token {
            *dist = distance_to_plane_code(xsize, *dist as usize);
        }
    }
}

/// Add a single literal, checking the color cache first.
#[inline]
fn add_single_literal(argb_val: u32, cache: &mut Option<&mut ColorCache>, refs: &mut BackwardRefs) {
    if let Some(c) = cache {
        if let Some(idx) = c.lookup(argb_val) {
            refs.push(PixOrCopy::cache_idx(idx));
        } else {
            refs.push(PixOrCopy::literal(argb_val));
            c.insert(argb_val);
        }
    } else {
        refs.push(PixOrCopy::literal(argb_val));
    }
}

/// LZ77 Standard with look-ahead optimization.
///
/// Matches libwebp's BackwardReferencesLz77:
/// - Uses hash chain for best matches
/// - Look-ahead: checks positions i+1..i+len for better "reach" (j + len_j)
/// - Stores RAW distances (not plane codes)
pub(super) fn backward_references_lz77(
    argb: &[u32],
    _xsize: usize,
    _ysize: usize,
    cache_bits: u8,
    hash_chain: &HashChain,
) -> BackwardRefs {
    let pix_count = argb.len();
    let mut refs = BackwardRefs::with_capacity(pix_count);
    let use_color_cache = cache_bits > 0;
    let mut cache = if use_color_cache {
        Some(ColorCache::new(cache_bits))
    } else {
        None
    };

    let mut i = 0;
    let mut i_last_check: i32 = -1;

    while i < pix_count {
        let (offset, len) = hash_chain.find_copy(i);
        let mut chosen_len;

        if len >= MIN_LENGTH {
            let len_ini = len;
            let mut max_reach = 0i64;
            let j_max = if i + len_ini >= pix_count {
                pix_count - 1
            } else {
                i + len_ini
            };

            // Only start from what we have not checked already
            i_last_check = i_last_check.max(i as i32);

            // Look-ahead: find best combination of current + next match
            // Check positions i+1 through i+len to see if deferring gives better reach
            chosen_len = len;
            for j in ((i_last_check + 1) as usize)..=j_max {
                let len_j = hash_chain.length(j);
                let reach = j as i64
                    + if len_j >= MIN_LENGTH {
                        len_j as i64
                    } else {
                        1 // Single literal
                    };
                if reach > max_reach {
                    chosen_len = j - i;
                    max_reach = reach;
                    if max_reach >= pix_count as i64 {
                        break;
                    }
                }
            }
        } else {
            chosen_len = 1;
        }

        if chosen_len <= 1 {
            // Emit literal
            add_single_literal(argb[i], &mut cache.as_mut(), &mut refs);
            i += 1;
        } else {
            // Emit backward reference with RAW distance
            refs.push(PixOrCopy::copy(chosen_len as u16, offset as u32));
            // Update color cache with copied pixels
            if let Some(ref mut c) = cache {
                for k in 0..chosen_len {
                    c.insert(argb[i + k]);
                }
            }
            i += chosen_len;
        }
    }

    refs
}

/// RLE backward references (distance=1 and distance=xsize only).
///
/// Matches libwebp's BackwardReferencesRle.
pub(super) fn backward_references_rle(
    argb: &[u32],
    xsize: usize,
    _ysize: usize,
    cache_bits: u8,
) -> BackwardRefs {
    let pix_count = argb.len();
    let mut refs = BackwardRefs::with_capacity(pix_count);
    let use_color_cache = cache_bits > 0;
    let mut cache = if use_color_cache {
        Some(ColorCache::new(cache_bits))
    } else {
        None
    };

    // Add first pixel as literal
    add_single_literal(argb[0], &mut cache.as_mut(), &mut refs);

    let mut i = 1;
    while i < pix_count {
        let max_len = (pix_count - i).min(MAX_LENGTH);

        // Check RLE match (distance=1, same as previous pixel)
        let rle_len = {
            let mut len = 0;
            while len < max_len && argb[i + len] == argb[i - 1 + len] {
                len += 1;
            }
            // Quick rejection first
            if argb[i] != argb[i - 1] { 0 } else { len }
        };

        // Check previous row match (distance=xsize)
        let prev_row_len = if i < xsize {
            0
        } else {
            let mut len = 0;
            let max = max_len;
            while len < max && argb[i + len] == argb[i - xsize + len] {
                len += 1;
            }
            // Quick rejection
            if argb[i] != argb[i - xsize] { 0 } else { len }
        };

        if rle_len >= prev_row_len && rle_len >= MIN_LENGTH {
            // Use RLE (distance=1)
            refs.push(PixOrCopy::copy(rle_len as u16, 1));
            // RLE doesn't change cache state (same pixel repeated)
            i += rle_len;
        } else if prev_row_len >= MIN_LENGTH {
            // Use previous row (distance=xsize)
            refs.push(PixOrCopy::copy(prev_row_len as u16, xsize as u32));
            if let Some(ref mut c) = cache {
                for k in 0..prev_row_len {
                    c.insert(argb[i + k]);
                }
            }
            i += prev_row_len;
        } else {
            // Literal
            add_single_literal(argb[i], &mut cache.as_mut(), &mut refs);
            i += 1;
        }
    }

    refs
}

/// Maximum window offsets for LZ77 Box (matching libwebp WINDOW_OFFSETS_SIZE_MAX).
/// Restricts Box to the 32 closest spatial offsets for speed.
const WINDOW_OFFSETS_SIZE_MAX: usize = 32;

/// LZ77 Box strategy for palette images.
///
/// Uses a fixed spatial window of offsets around each pixel, with run-length
/// counting for fast matching. Optimized for images with few colors where
/// consecutive identical pixels are common.
/// Matches libwebp's BackwardReferencesLz77Box.
///
/// Returns a HashChain populated with the best offset/length for each position,
/// then feeds into the standard BackwardReferencesLz77 to produce final refs.
pub(super) fn backward_references_lz77_box(
    argb: &[u32],
    xsize: usize,
    ysize: usize,
    cache_bits: u8,
    hash_chain_best: &HashChain,
) -> BackwardRefs {
    use alloc::vec;

    let pix_count = xsize * ysize;
    if pix_count == 0 {
        return BackwardRefs::new();
    }

    // counts[i] = how many times pixel at i is repeated starting from i
    let mut counts = vec![0u16; pix_count];
    counts[pix_count - 1] = 1;
    for i in (0..pix_count - 1).rev() {
        if argb[i] == argb[i + 1] {
            counts[i] = counts[i + 1].saturating_add(1).min(MAX_LENGTH as u16);
        } else {
            counts[i] = 1;
        }
    }

    // Build window offsets from VP8LDistanceToPlaneCode spiral order
    let mut window_offsets = [0i32; WINDOW_OFFSETS_SIZE_MAX];
    let mut window_offsets_size = 0;
    for y in 0..=6i32 {
        for x in -6i32..=6 {
            let offset = y * xsize as i32 + x;
            if offset <= 0 {
                continue;
            }
            let plane_code = distance_to_plane_code(xsize, offset as usize);
            let pc = plane_code as usize;
            if pc == 0 || pc > WINDOW_OFFSETS_SIZE_MAX {
                continue;
            }
            window_offsets[pc - 1] = offset;
        }
    }
    // Remove zero entries (narrow images may not reach all plane codes)
    let mut compacted = [0i32; WINDOW_OFFSETS_SIZE_MAX];
    for &wo in &window_offsets {
        if wo != 0 {
            compacted[window_offsets_size] = wo;
            window_offsets_size += 1;
        }
    }
    let window_offsets = &compacted[..window_offsets_size];

    // Find offsets that are new (not reachable from P-1 with existing offsets)
    let mut window_offsets_new = Vec::with_capacity(window_offsets_size);
    for &wo in window_offsets {
        let is_reachable = window_offsets.iter().any(|&wj| wo == wj + 1);
        if !is_reachable {
            window_offsets_new.push(wo);
        }
    }

    // Build a hash chain with Box-strategy matches
    let mut box_chain = HashChain::empty(pix_count);
    let mut best_offset_prev: i32 = -1;
    let mut best_length_prev: i32 = -1;

    for i in 1..pix_count {
        let mut best_length = hash_chain_best.find_copy(i).1 as i32;
        let mut best_offset: i32;
        let mut do_compute = true;

        if best_length >= MAX_LENGTH as i32 {
            // Check if best match from hash_chain_best is in our window
            let bo = hash_chain_best.offset(i) as i32;
            for &wo in window_offsets {
                if bo == wo {
                    do_compute = false;
                    break;
                }
            }
            best_offset = bo;
        } else {
            best_offset = 0;
        }

        if do_compute {
            let use_prev = best_length_prev > 1 && best_length_prev < MAX_LENGTH as i32;
            let offsets_to_try = if use_prev {
                &window_offsets_new[..]
            } else {
                window_offsets
            };

            best_length = if use_prev { best_length_prev - 1 } else { 0 };
            best_offset = if use_prev { best_offset_prev } else { 0 };

            for &wo in offsets_to_try {
                let j_offset = i as i32 - wo;
                if j_offset < 0 || argb[j_offset as usize] != argb[i] {
                    continue;
                }

                // Match using run-length counts for fast comparison
                let mut curr_length = 0i32;
                let mut j = i;
                let mut jo = j_offset as usize;
                loop {
                    let cjo = counts[jo] as i32;
                    let cj = counts[j] as i32;
                    if cjo != cj {
                        curr_length += cjo.min(cj);
                        break;
                    }
                    curr_length += cjo;
                    jo += cjo as usize;
                    j += cjo as usize;
                    if curr_length > MAX_LENGTH as i32 || j >= pix_count || argb[jo] != argb[j] {
                        break;
                    }
                }

                if best_length < curr_length {
                    best_offset = wo;
                    if curr_length >= MAX_LENGTH as i32 {
                        best_length = MAX_LENGTH as i32;
                        break;
                    } else {
                        best_length = curr_length;
                    }
                }
            }
        }

        if best_length <= MIN_LENGTH as i32 {
            box_chain.set(i, 0, 0);
            best_offset_prev = 0;
            best_length_prev = 0;
        } else {
            box_chain.set(
                i,
                best_offset as usize,
                best_length.min(MAX_LENGTH as i32) as usize,
            );
            best_offset_prev = best_offset;
            best_length_prev = best_length;
        }
    }

    // Use the box chain with standard LZ77 to produce final refs
    backward_references_lz77(argb, xsize, ysize, cache_bits, &box_chain)
}

/// Maximum color cache bits.
const MAX_COLOR_CACHE_BITS: u8 = 10;

/// Color cache hash multiplier (must match ColorCache::hash exactly).
const COLOR_CACHE_MULT: u32 = 0x1e35a7bd;

/// Calculate best cache size by simulating all sizes simultaneously.
///
/// Matches libwebp's CalculateBestCacheSize:
/// - Tests all cache sizes from 0 to cache_bits_max simultaneously
/// - Uses nested key derivation (key for size N is key for size N+1 >> 1)
/// - Picks the size with minimum estimated entropy
fn calculate_best_cache_size(
    argb: &[u32],
    quality: u8,
    refs: &BackwardRefs,
    cache_bits_max: u8,
) -> u8 {
    if quality <= 25 || cache_bits_max == 0 {
        return 0;
    }

    let cache_bits_max = cache_bits_max.min(MAX_COLOR_CACHE_BITS);

    // Build histograms for all cache sizes simultaneously
    let mut histos: Vec<Histogram> = (0..=cache_bits_max).map(Histogram::new).collect();

    // Build color caches for sizes 1..=cache_bits_max
    let mut caches: Vec<ColorCache> = (0..=cache_bits_max)
        .map(|bits| {
            if bits > 0 {
                ColorCache::new(bits)
            } else {
                ColorCache::new(1) // Dummy, not used
            }
        })
        .collect();

    let mut argb_idx = 0usize;

    for token in refs.iter() {
        match *token {
            PixOrCopy::Literal(_) | PixOrCopy::CacheIdx(_) => {
                // Get the actual pixel value (from token if literal, from argb if cache idx)
                let pix = if let PixOrCopy::Literal(p) = *token {
                    p
                } else {
                    argb[argb_idx]
                };

                let a = argb_alpha(pix) as usize;
                let r = argb_red(pix) as usize;
                let g = argb_green(pix) as usize;
                let b = argb_blue(pix) as usize;

                // For cache_bits = 0, always literal
                histos[0].literal[g] += 1;
                histos[0].red[r] += 1;
                histos[0].blue[b] += 1;
                histos[0].alpha[a] += 1;

                // For cache_bits > 0, check cache hit
                let key_max = (COLOR_CACHE_MULT.wrapping_mul(pix)) >> (32 - cache_bits_max);
                let mut key = key_max;

                for i in (1..=cache_bits_max).rev() {
                    let idx = i as usize;
                    if caches[idx].lookup(pix).is_some() {
                        // Cache hit - count as cache index
                        let cache_code = NUM_LITERAL_CODES + NUM_LENGTH_CODES + key as usize;
                        if cache_code < histos[idx].literal.len() {
                            histos[idx].literal[cache_code] += 1;
                        }
                    } else {
                        // Cache miss - count as literal, insert into cache
                        caches[idx].insert(pix);
                        histos[idx].literal[g] += 1;
                        histos[idx].red[r] += 1;
                        histos[idx].blue[b] += 1;
                        histos[idx].alpha[a] += 1;
                    }
                    key >>= 1;
                }

                argb_idx += 1;
            }
            PixOrCopy::Copy { len, .. } => {
                let len = len as usize;
                // Length codes go in literal histogram for all sizes
                let (len_code, _) = super::histogram::length_to_code(len as u16);
                for h in histos.iter_mut() {
                    h.literal[NUM_LITERAL_CODES + len_code as usize] += 1;
                }

                // Update caches with copied pixels
                let mut prev_argb = argb[argb_idx] ^ 0xFFFFFFFF;
                for k in 0..len {
                    let pix = argb[argb_idx + k];
                    if pix != prev_argb {
                        // Only update when color changes (optimization)
                        for i in (1..=cache_bits_max).rev() {
                            caches[i as usize].insert(pix);
                        }
                        prev_argb = pix;
                    }
                }
                argb_idx += len;
            }
        }
    }

    // Pick cache size with minimum entropy
    let mut best_bits = 0u8;
    let mut best_entropy = u64::MAX;

    for i in 0..=cache_bits_max {
        let entropy = estimate_histogram_bits(&histos[i as usize]);
        if entropy < best_entropy {
            best_entropy = entropy;
            best_bits = i;
        }
    }

    best_bits
}

/// Strip color cache references, converting CacheIdx back to Literal.
/// Uses the original argb array to recover pixel values.
pub fn strip_cache_from_refs(argb: &[u32], refs: &mut BackwardRefs) {
    let mut pixel_index = 0usize;
    for token in refs.tokens.iter_mut() {
        match token {
            PixOrCopy::Literal(_) => {
                pixel_index += 1;
            }
            PixOrCopy::CacheIdx(_) => {
                // Convert back to literal using original pixel data
                *token = PixOrCopy::literal(argb[pixel_index]);
                pixel_index += 1;
            }
            PixOrCopy::Copy { len, .. } => {
                pixel_index += *len as usize;
            }
        }
    }
}

/// Apply color cache to existing backward references.
///
/// Converts literal pixels that hit the cache into cache index references.
/// Matches libwebp's BackwardRefsWithLocalCache.
pub(super) fn apply_cache_to_refs(argb: &[u32], cache_bits: u8, refs: &mut BackwardRefs) {
    let mut cache = ColorCache::new(cache_bits);
    let mut pixel_index = 0usize;

    for token in refs.tokens.iter_mut() {
        match token {
            PixOrCopy::Literal(argb_val) => {
                if let Some(idx) = cache.lookup(*argb_val) {
                    *token = PixOrCopy::cache_idx(idx);
                } else {
                    cache.insert(*argb_val);
                }
                pixel_index += 1;
            }
            PixOrCopy::CacheIdx(_) => {
                // Should not happen - refs were built without cache
                pixel_index += 1;
            }
            PixOrCopy::Copy { len, .. } => {
                let len = *len as usize;
                for k in 0..len {
                    cache.insert(argb[pixel_index + k]);
                }
                pixel_index += len;
            }
        }
    }
}

/// Get the best backward references for the image.
///
/// Tries multiple LZ77 strategies and picks the best one.
/// For quality >= 25, applies cost-based optimal parsing (TraceBackwards).
/// Optionally applies color cache for further compression.
///
/// Matches libwebp's GetBackwardReferences flow:
/// 1. Try LZ77 Standard and RLE strategies
/// 2. Pick best based on histogram entropy
/// 3. For quality >= 25: apply TraceBackwards DP optimization
/// 4. Estimate optimal color cache size
/// 5. Apply cache to refs
/// 6. Apply 2D locality transform
pub fn get_backward_references(
    argb: &[u32],
    width: usize,
    height: usize,
    quality: u8,
    cache_bits_max: u8,
) -> (BackwardRefs, u8) {
    get_backward_references_inner(argb, width, height, quality, cache_bits_max, 0)
}

/// Get backward references with optional LZ77 Box for palette images.
pub fn get_backward_references_with_palette(
    argb: &[u32],
    width: usize,
    height: usize,
    quality: u8,
    cache_bits_max: u8,
    palette_size: usize,
) -> (BackwardRefs, u8) {
    get_backward_references_inner(argb, width, height, quality, cache_bits_max, palette_size)
}

fn get_backward_references_inner(
    argb: &[u32],
    width: usize,
    height: usize,
    quality: u8,
    cache_bits_max: u8,
    palette_size: usize,
) -> (BackwardRefs, u8) {
    let size = width * height;

    if size == 0 {
        return (BackwardRefs::new(), 0);
    }

    // Build hash chain
    let hash_chain = HashChain::new(argb, quality, width);

    // Try LZ77 Standard (no cache initially - find best LZ77 first)
    let refs_lz77 = backward_references_lz77(argb, width, height, 0, &hash_chain);

    // Try RLE
    let refs_rle = backward_references_rle(argb, width, height, 0);

    // Pick best strategy based on histogram entropy
    let histo_lz77 = Histogram::from_refs_with_plane_codes(&refs_lz77, 0, width);
    let histo_rle = Histogram::from_refs_with_plane_codes(&refs_rle, 0, width);

    let cost_lz77 = estimate_histogram_bits(&histo_lz77);
    let cost_rle = estimate_histogram_bits(&histo_rle);

    let mut best_refs = if cost_rle < cost_lz77 {
        refs_rle
    } else {
        refs_lz77
    };
    let mut best_cost = cost_lz77.min(cost_rle);

    // Try LZ77 Box for palette images (<=16 colors) — matching libwebp n_lz77s=2
    if palette_size > 0 && palette_size <= 16 {
        let refs_box = backward_references_lz77_box(argb, width, height, 0, &hash_chain);
        let histo_box = Histogram::from_refs_with_plane_codes(&refs_box, 0, width);
        let cost_box = estimate_histogram_bits(&histo_box);
        if cost_box < best_cost {
            best_refs = refs_box;
            best_cost = cost_box;
        }
    }
    let _ = best_cost; // suppress unused warning

    // Determine initial color cache size from greedy refs
    let mut cache_bits = if cache_bits_max > 0 && quality > 25 {
        calculate_best_cache_size(argb, quality, &best_refs, cache_bits_max)
    } else {
        0
    };

    // Apply cache to refs before TraceBackwards (matching libwebp's flow).
    // The CostModel needs cache entries to accurately estimate cache costs.
    if cache_bits > 0 {
        apply_cache_to_refs(argb, cache_bits, &mut best_refs);
    }

    // For quality >= 25, apply cost-based optimal parsing (TraceBackwards).
    // This uses DP to find the globally optimal literal/copy sequence
    // based on a cost model derived from the cache-applied greedy result.
    if quality >= 25 {
        let optimized =
            trace_backwards_optimize(argb, width, height, cache_bits, &hash_chain, &best_refs);

        // Use optimized result if it improves entropy
        let histo_orig = Histogram::from_refs_with_plane_codes(&best_refs, cache_bits, width);
        let histo_opt = Histogram::from_refs_with_plane_codes(&optimized, cache_bits, width);
        let cost_orig = estimate_histogram_bits(&histo_orig);
        let cost_opt = estimate_histogram_bits(&histo_opt);

        if cost_opt <= cost_orig {
            best_refs = optimized;
            // Recalculate cache from optimized refs, re-apply if different
            if cache_bits_max > 0 {
                let new_cache_bits =
                    calculate_best_cache_size(argb, quality, &best_refs, cache_bits_max);
                if new_cache_bits != cache_bits {
                    // Strip old cache entries and re-apply with new bits
                    strip_cache_from_refs(argb, &mut best_refs);
                    cache_bits = new_cache_bits;
                    if cache_bits > 0 {
                        apply_cache_to_refs(argb, cache_bits, &mut best_refs);
                    }
                }
            }
        } else {
            // Keep greedy result (already has cache applied)
        }
    }

    // If no TraceBackwards was done (quality < 25), apply cache now
    if quality < 25 && cache_bits > 0 {
        apply_cache_to_refs(argb, cache_bits, &mut best_refs);
    }

    // Apply 2D locality transform (convert raw distances to plane codes)
    apply_2d_locality(&mut best_refs, width);

    (best_refs, cache_bits)
}

/// Simple backward reference finder for very low quality.
pub fn compute_backward_refs_simple(argb: &[u32], _width: usize, _height: usize) -> BackwardRefs {
    let size = argb.len();
    let mut refs = BackwardRefs::with_capacity(size);

    if size == 0 {
        return refs;
    }

    let mut pos = 0;
    while pos < size {
        let argb_val = argb[pos];

        // Check for run of identical pixels (distance = 1)
        let mut run_len = 1;
        while pos + run_len < size && argb[pos + run_len] == argb_val && run_len < MAX_LENGTH {
            run_len += 1;
        }

        if run_len >= MIN_LENGTH {
            // Emit first pixel as literal, then backward reference
            refs.push(PixOrCopy::literal(argb_val));
            if run_len > 1 {
                // Distance code 1 = distance 1 (previous pixel)
                refs.push(PixOrCopy::copy((run_len - 1) as u16, 1));
            }
            pos += run_len;
        } else {
            // Just emit literal
            refs.push(PixOrCopy::literal(argb_val));
            pos += 1;
        }
    }

    refs
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_distance_code_roundtrip() {
        let xsize = 100;
        for dist in 1..=200 {
            let code = distance_to_plane_code(xsize, dist);
            let back = plane_code_to_distance(xsize, code);
            assert_eq!(back, dist, "dist={}, code={}", dist, code);
        }
    }

    #[test]
    fn test_2d_neighborhood_codes() {
        let xsize = 100;
        assert_eq!(distance_to_plane_code(xsize, 1), 2);
        assert_eq!(distance_to_plane_code(xsize, xsize), 1);
    }

    #[test]
    fn test_backward_refs_simple() {
        let pixels = vec![0xFF000000u32; 100];
        let refs = compute_backward_refs_simple(&pixels, 10, 10);

        assert_eq!(refs.len(), 2);
        assert!(refs.tokens[0].is_literal());
        assert!(refs.tokens[1].is_copy());
    }

    #[test]
    fn test_get_backward_references() {
        // Simple repeating pattern
        let mut pixels = Vec::new();
        for _ in 0..100 {
            pixels.push(0xFF112233u32);
            pixels.push(0xFF445566u32);
        }
        let (refs, _cache_bits) = get_backward_references(&pixels, 10, 20, 75, 10);
        assert!(!refs.is_empty());
    }

    #[test]
    fn test_lz77_vs_rle() {
        // Solid color image - RLE should win
        let pixels = vec![0xFF000000u32; 1000];
        let hash_chain = HashChain::new(&pixels, 75, 100);
        let refs_lz77 = backward_references_lz77(&pixels, 100, 10, 0, &hash_chain);
        let refs_rle = backward_references_rle(&pixels, 100, 10, 0);

        // Both should produce valid, compact output
        assert!(refs_lz77.len() < 100); // Should compress well
        assert!(refs_rle.len() < 100);
    }
}
