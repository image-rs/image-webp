//! Cost model and optimal parsing for VP8L backward references.
//!
//! Implements a Zopfli-like cost-based optimizer (TraceBackwards) that
//! improves greedy LZ77 by recomputing optimal literal/copy decisions
//! using dynamic programming with per-symbol bit costs.
//!
//! Uses an interval-based cost manager matching libwebp's
//! `BackwardReferencesHashChainDistanceOnly` for efficient O(n) processing
//! instead of the naive O(n * max_match_len) approach.

use alloc::vec;
use alloc::vec::Vec;

use super::backward_refs::distance_to_plane_code;
use super::color_cache::ColorCache;
use super::histogram::{distance_code_to_prefix, length_to_code, Histogram};
use super::types::{
    argb_alpha, argb_blue, argb_green, argb_red, BackwardRefs, PixOrCopy, MAX_LENGTH,
    NUM_LENGTH_CODES, NUM_LITERAL_CODES,
};

/// Fixed-point precision for entropy (matches libwebp LOG_2_PRECISION_BITS).
const LOG_2_PRECISION_BITS: u32 = 23;

/// Max active intervals before serializing (matches libwebp).
const COST_CACHE_INTERVAL_SIZE_MAX: usize = 500;

/// Small interval threshold — serialize directly instead of tracking.
const SKIP_DISTANCE: usize = 10;

/// Compute log2(v) in fixed-point (matching libwebp's VP8LFastLog2).
/// Returns 0 for v == 0 or v == 1.
fn fast_log2(v: u32) -> u32 {
    if v <= 1 {
        return 0;
    }
    let v_f64 = v as f64;
    (libm::log2(v_f64) * (1u64 << LOG_2_PRECISION_BITS) as f64) as u32
}

/// Rounding division matching libwebp's DivRound.
#[inline]
fn div_round(a: i64, b: i64) -> i64 {
    if (a < 0) == (b < 0) {
        (a + b / 2) / b
    } else {
        (a - b / 2) / b
    }
}

/// Convert histogram population counts to per-symbol bit cost estimates.
///
/// Each output[i] = VP8LFastLog2(sum) - VP8LFastLog2(counts[i]).
/// Matches libwebp's ConvertPopulationCountTableToBitEstimates.
fn counts_to_bit_estimates(counts: &[u32]) -> Vec<u32> {
    let total: u32 = counts.iter().sum();
    let n = counts.len();

    let nonzeros = counts.iter().filter(|&&c| c > 0).count();
    if nonzeros <= 1 {
        return vec![0u32; n];
    }

    let logsum = fast_log2(total);
    let mut output = vec![0u32; n];
    for (i, &count) in counts.iter().enumerate() {
        output[i] = logsum.saturating_sub(fast_log2(count));
    }
    output
}

/// Per-symbol bit cost model built from histogram statistics.
pub struct CostModel {
    /// Green/literal/length costs (256 + 24 + cache).
    literal: Vec<u32>,
    /// Red channel costs (256).
    red: Vec<u32>,
    /// Blue channel costs (256).
    blue: Vec<u32>,
    /// Alpha channel costs (256).
    alpha: Vec<u32>,
    /// Distance costs (40).
    distance: Vec<u32>,
}

impl CostModel {
    /// Build cost model from histogram of initial backward refs.
    pub fn build(xsize: usize, cache_bits: u8, refs: &BackwardRefs) -> Self {
        // Build histogram with plane-code-aware distances
        let histo = Histogram::from_refs_with_plane_codes(refs, cache_bits, xsize);

        Self {
            literal: counts_to_bit_estimates(&histo.literal),
            red: counts_to_bit_estimates(&histo.red),
            blue: counts_to_bit_estimates(&histo.blue),
            alpha: counts_to_bit_estimates(&histo.alpha),
            distance: counts_to_bit_estimates(&histo.distance),
        }
    }

    /// Cost of encoding a literal ARGB pixel.
    #[inline]
    fn literal_cost(&self, argb: u32) -> i64 {
        let a = argb_alpha(argb) as usize;
        let r = argb_red(argb) as usize;
        let g = argb_green(argb) as usize;
        let b = argb_blue(argb) as usize;
        self.alpha[a] as i64 + self.red[r] as i64 + self.literal[g] as i64 + self.blue[b] as i64
    }

    /// Cost of encoding a color cache index.
    #[inline]
    fn cache_cost(&self, idx: u16) -> i64 {
        let literal_idx = NUM_LITERAL_CODES + NUM_LENGTH_CODES + idx as usize;
        if literal_idx < self.literal.len() {
            self.literal[literal_idx] as i64
        } else {
            i64::MAX / 2
        }
    }

    /// Cost of encoding a copy length.
    #[inline]
    fn length_cost(&self, length: u32) -> i64 {
        let (code, _) = length_to_code(length as u16);
        let extra_bits = if code < 4 {
            0
        } else {
            (code / 2).saturating_sub(1) as u32
        };
        self.literal[NUM_LITERAL_CODES + code as usize] as i64
            + ((extra_bits as i64) << LOG_2_PRECISION_BITS)
    }

    /// Cost of encoding a distance code.
    #[inline]
    fn distance_cost(&self, dist_code: u32) -> i64 {
        let (code, _) = distance_code_to_prefix(dist_code);
        let extra_bits = if code < 4 {
            0
        } else {
            (code / 2).saturating_sub(1) as u32
        };
        self.distance[code as usize] as i64 + ((extra_bits as i64) << LOG_2_PRECISION_BITS)
    }
}

// ---------------------------------------------------------------------------
// CostManager and interval handling
// ---------------------------------------------------------------------------

/// A cost interval tracking the minimum cost contribution from a single
/// source position over a range of target pixels.
///
/// Uses index-based doubly-linked list instead of raw pointers.
#[derive(Clone)]
struct CostInterval {
    cost: i64,
    start: usize,
    end: usize, // exclusive
    index: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Cached length cost interval — groups consecutive lengths with same cost.
struct CostCacheInterval {
    cost: i64,
    start: usize,
    end: usize, // exclusive
}

/// Interval-based cost manager matching libwebp's CostManager.
///
/// Maintains a sorted linked list of non-overlapping cost intervals
/// to efficiently track optimal paths without iterating over all copy
/// lengths for every pixel.
struct CostManager {
    /// Pool of interval nodes (index-based linked list).
    intervals: Vec<CostInterval>,
    /// Head of the active interval list (index into `intervals`).
    head: Option<usize>,
    /// Free slot indices for reuse.
    free_slots: Vec<usize>,
    /// Number of active intervals.
    count: usize,
    /// Pre-grouped length cost intervals.
    cache_intervals: Vec<CostCacheInterval>,
    /// Per-length costs: cost_cache[k] = GetLengthCost(k).
    cost_cache: Vec<i64>,
    /// Minimum cost to reach each pixel.
    costs: Vec<i64>,
    /// Step size at each pixel (1 = literal, >=2 = copy length).
    dist_array: Vec<u16>,
}

impl CostManager {
    /// Initialize the cost manager for a given pixel count and cost model.
    fn new(pix_count: usize, cost_model: &CostModel) -> Self {
        let cost_cache_size = pix_count.min(MAX_LENGTH);

        // Fill cost_cache with GetLengthCost(k) for k in 0..cost_cache_size.
        // Matches libwebp where VP8LPrefixEncodeBits(k) does --k internally.
        // k=0 → code 0, extra_bits 0 (copy length 1).
        // k=N → same as our length_to_code(N) for N >= 1.
        let mut cost_cache = vec![0i64; cost_cache_size];
        // k=0: code=0, extra_bits=0 → just literal[256]
        if cost_cache_size > 0 {
            cost_cache[0] = cost_model.literal[NUM_LITERAL_CODES] as i64;
        }
        for (k, slot) in cost_cache.iter_mut().enumerate().skip(1) {
            *slot = cost_model.length_cost(k as u32);
        }

        // Build cache_intervals: group consecutive lengths with same cost.
        let mut cache_intervals = Vec::with_capacity(32);
        if cost_cache_size > 0 {
            let mut cur_start = 0;
            let mut cur_cost = cost_cache[0];
            for (i, &c) in cost_cache.iter().enumerate().skip(1) {
                if c != cur_cost {
                    cache_intervals.push(CostCacheInterval {
                        cost: cur_cost,
                        start: cur_start,
                        end: i,
                    });
                    cur_start = i;
                    cur_cost = cost_cache[i];
                }
            }
            cache_intervals.push(CostCacheInterval {
                cost: cur_cost,
                start: cur_start,
                end: cost_cache_size,
            });
        }

        let costs = vec![i64::MAX; pix_count];
        let dist_array = vec![0u16; pix_count];

        Self {
            intervals: Vec::with_capacity(64),
            head: None,
            free_slots: Vec::new(),
            count: 0,
            cache_intervals,
            cost_cache,
            costs,
            dist_array,
        }
    }

    /// Allocate or reuse an interval slot, returning its index.
    fn alloc_interval(&mut self, cost: i64, start: usize, end: usize, index: usize) -> usize {
        if let Some(slot) = self.free_slots.pop() {
            let interval = &mut self.intervals[slot];
            interval.cost = cost;
            interval.start = start;
            interval.end = end;
            interval.index = index;
            interval.prev = None;
            interval.next = None;
            slot
        } else {
            let slot = self.intervals.len();
            self.intervals.push(CostInterval {
                cost,
                start,
                end,
                index,
                prev: None,
                next: None,
            });
            slot
        }
    }

    /// Free an interval slot back to the pool.
    fn free_interval(&mut self, slot: usize) {
        self.free_slots.push(slot);
    }

    /// Update cost at pixel `i` if `cost` is better, recording `position`
    /// as the source.
    #[inline]
    fn update_cost(&mut self, i: usize, position: usize, cost: i64) {
        let k = i - position;
        debug_assert!(k < MAX_LENGTH);
        if self.costs[i] > cost {
            self.costs[i] = cost;
            self.dist_array[i] = (k + 1) as u16;
        }
    }

    /// Update cost for all pixels in [start, end) from a single source.
    fn update_cost_per_interval(&mut self, start: usize, end: usize, position: usize, cost: i64) {
        for i in start..end {
            self.update_cost(i, position, cost);
        }
    }

    /// Connect `prev_slot` and `next_slot` in the linked list.
    fn connect(&mut self, prev_slot: Option<usize>, next_slot: Option<usize>) {
        if let Some(p) = prev_slot {
            self.intervals[p].next = next_slot;
        } else {
            self.head = next_slot;
        }
        if let Some(n) = next_slot {
            self.intervals[n].prev = prev_slot;
        }
    }

    /// Remove an interval from the active list and recycle it.
    fn pop_interval(&mut self, slot: usize) {
        let prev = self.intervals[slot].prev;
        let next = self.intervals[slot].next;
        self.connect(prev, next);
        self.free_interval(slot);
        self.count -= 1;
    }

    /// Find the correct position and insert `new_slot` into the sorted list,
    /// using `hint` as a starting point for the search.
    fn position_orphan(&mut self, new_slot: usize, hint: Option<usize>) {
        let new_start = self.intervals[new_slot].start;
        let mut prev = hint;

        // If hint is None, start from head
        if prev.is_none() {
            prev = self.head;
        }

        // Walk backward if new_start < prev.start
        while let Some(p) = prev {
            if new_start >= self.intervals[p].start {
                break;
            }
            prev = self.intervals[p].prev;
        }

        // Walk forward past elements with start < new_start
        while let Some(p) = prev {
            if let Some(nxt) = self.intervals[p].next {
                if self.intervals[nxt].start < new_start {
                    prev = Some(nxt);
                    continue;
                }
            }
            break;
        }

        // Insert after prev
        if let Some(p) = prev {
            let after = self.intervals[p].next;
            self.connect(Some(new_slot), after);
            self.connect(Some(p), Some(new_slot));
        } else {
            let old_head = self.head;
            self.connect(Some(new_slot), old_head);
            self.connect(None, Some(new_slot));
        }
    }

    /// Insert an interval [start, end) into the sorted list. If we've hit
    /// the max interval count, serialize directly to costs instead.
    fn insert_interval(
        &mut self,
        hint: Option<usize>,
        cost: i64,
        position: usize,
        start: usize,
        end: usize,
    ) {
        if start >= end {
            return;
        }
        if self.count >= COST_CACHE_INTERVAL_SIZE_MAX {
            self.update_cost_per_interval(start, end, position, cost);
            return;
        }

        let new_slot = self.alloc_interval(cost, start, end, position);
        self.position_orphan(new_slot, hint);
        self.count += 1;
    }

    /// Push a new interval contribution from `position` covering `len` pixels.
    /// Handles splitting, merging, and removing existing intervals as needed.
    fn push_interval(&mut self, distance_cost: i64, position: usize, len: usize) {
        // Small intervals: serialize directly (cheaper than list management).
        if len < SKIP_DISTANCE {
            for j in position..position + len {
                let k = j - position;
                debug_assert!(k < self.cost_cache.len());
                let cost_tmp = distance_cost + self.cost_cache[k];
                if self.costs[j] > cost_tmp {
                    self.costs[j] = cost_tmp;
                    self.dist_array[j] = (k + 1) as u16;
                }
            }
            return;
        }

        let mut interval = self.head;

        for ci_idx in 0..self.cache_intervals.len() {
            let ci_start = self.cache_intervals[ci_idx].start;
            if ci_start >= len {
                break;
            }
            let ci_end = self.cache_intervals[ci_idx].end.min(len);
            let ci_cost = self.cache_intervals[ci_idx].cost;

            // The new interval covers [start, end) with this cost
            let mut start = position + ci_start;
            let end = position + ci_end;
            let cost = distance_cost + ci_cost;

            while let Some(slot) = interval {
                let int_start = self.intervals[slot].start;
                if int_start >= end {
                    break;
                }
                let int_end = self.intervals[slot].end;
                let int_cost = self.intervals[slot].cost;
                let int_index = self.intervals[slot].index;
                let next_slot = self.intervals[slot].next;

                // Skip if no overlap
                if start >= int_end {
                    interval = Some(next_slot.unwrap_or(slot));
                    if next_slot.is_none() {
                        break;
                    }
                    interval = next_slot;
                    continue;
                }

                if cost >= int_cost {
                    // Existing interval is better — insert our part before it,
                    // then skip past.
                    let start_new = int_end;
                    self.insert_interval(Some(slot), cost, position, start, int_start);
                    start = start_new;
                    if start >= end {
                        break;
                    }
                    interval = next_slot;
                    continue;
                }

                // New interval is better (cost < int_cost)
                if start <= int_start {
                    if int_end <= end {
                        // Existing fully contained — remove it.
                        self.pop_interval(slot);
                    } else {
                        // Existing extends past end — shrink its start.
                        self.intervals[slot].start = end;
                        break;
                    }
                } else {
                    // start > int_start
                    if end < int_end {
                        // Existing fully contains new — split existing.
                        let end_original = int_end;
                        self.intervals[slot].end = start;
                        self.insert_interval(Some(slot), int_cost, int_index, end, end_original);
                        // After split, the new part was inserted after slot.
                        // Move interval to after the newly inserted piece.
                        interval = self.intervals[slot].next;
                        break;
                    } else {
                        // Existing starts before new — shrink its end.
                        self.intervals[slot].end = start;
                    }
                }

                interval = next_slot;
            }

            // Insert remaining [start, end)
            self.insert_interval(interval, cost, position, start, end);
        }
    }

    /// Update cost at pixel `i` from all active intervals overlapping it.
    /// If `do_clean`, remove intervals that end before `i`.
    fn update_cost_at_index(&mut self, i: usize, do_clean: bool) {
        let mut current = self.head;
        while let Some(slot) = current {
            let int_start = self.intervals[slot].start;
            if int_start > i {
                break;
            }
            let int_end = self.intervals[slot].end;
            let next = self.intervals[slot].next;

            if int_end <= i {
                if do_clean {
                    self.pop_interval(slot);
                }
            } else {
                let int_cost = self.intervals[slot].cost;
                let int_index = self.intervals[slot].index;
                self.update_cost(i, int_index, int_cost);
            }
            current = next;
        }
    }
}

/// Compute cost-based optimal backward references using TraceBackwards.
///
/// Improves the greedy LZ77 result by using dynamic programming to find
/// the globally optimal sequence of literals and copies.
///
/// Uses an interval-based cost manager (matching libwebp) that avoids
/// the O(n * max_match_len) inner loop by tracking active cost intervals
/// in a sorted linked list.
///
/// # Arguments
/// - `argb`: transformed pixel data
/// - `xsize`, `ysize`: image dimensions
/// - `cache_bits`: color cache bits (0 = disabled)
/// - `hash_chain`: pre-built hash chain for match finding
/// - `initial_refs`: greedy LZ77 refs to derive cost model from
///
/// Returns improved backward refs (with raw distances, no 2D locality).
pub fn trace_backwards_optimize(
    argb: &[u32],
    xsize: usize,
    _ysize: usize,
    cache_bits: u8,
    hash_chain: &super::hash_chain::HashChain,
    initial_refs: &BackwardRefs,
) -> BackwardRefs {
    let pix_count = argb.len();
    let use_color_cache = cache_bits > 0;

    // Phase 1: Build cost model from initial greedy refs
    let cost_model = CostModel::build(xsize, cache_bits, initial_refs);

    // Phase 2: Forward DP pass using interval-based cost manager
    let mut manager = CostManager::new(pix_count, &cost_model);

    // Color cache for literal cost estimation (greedy approximation).
    let mut cache = if use_color_cache {
        Some(ColorCache::new(cache_bits))
    } else {
        None
    };

    // Add first pixel as literal
    if pix_count > 0 {
        add_single_literal_cost(
            argb,
            &cost_model,
            &mut cache,
            0,
            0, // prev_cost = 0
            &mut manager.costs,
            &mut manager.dist_array,
        );
    }

    // State for same-offset optimization (matching libwebp)
    let mut offset_prev: i64 = -1;
    let mut len_prev: i64 = -1;
    let mut offset_cost: i64 = -1;
    let mut first_offset_is_constant: i32 = -1;
    let mut reach: usize = 0;

    for i in 1..pix_count {
        let prev_cost = manager.costs[i - 1];

        // Try adding the pixel as a literal
        add_single_literal_cost(
            argb,
            &cost_model,
            &mut cache,
            i,
            prev_cost,
            &mut manager.costs,
            &mut manager.dist_array,
        );

        // Try copy starting at position i
        let (offset, len) = hash_chain.find_copy(i);

        if len >= 2 && offset > 0 {
            if offset as i64 != offset_prev {
                let plane_code = distance_to_plane_code(xsize, offset);
                offset_cost = cost_model.distance_cost(plane_code);
                first_offset_is_constant = 1;
                manager.push_interval(prev_cost + offset_cost, i, len);
            } else {
                // Same offset as previous pixel — use the optimization from libwebp
                // that avoids redundant PushInterval calls for constant-offset regions.
                debug_assert!(offset_cost >= 0);
                debug_assert!(len_prev >= 0);
                debug_assert!(first_offset_is_constant == 0 || first_offset_is_constant == 1);

                if first_offset_is_constant == 1 {
                    reach = i - 1 + len_prev as usize - 1;
                    first_offset_is_constant = 0;
                }

                if i + len - 1 > reach {
                    // Find last consecutive pixel with same offset within [i, reach+1]
                    let mut j = i;
                    while j <= reach {
                        let (offset_j, _) = hash_chain.find_copy(j + 1);
                        if offset_j as i64 != offset_prev {
                            break;
                        }
                        j += 1;
                    }
                    // Get the match at j (the boundary position)
                    let (_, len_j) = hash_chain.find_copy(j);

                    // Update costs at j-1 and j from active intervals
                    if j > 0 {
                        manager.update_cost_at_index(j - 1, false);
                    }
                    manager.update_cost_at_index(j, false);

                    let cost_at_j_minus_1 = if j > 0 { manager.costs[j - 1] } else { 0 };
                    manager.push_interval(cost_at_j_minus_1 + offset_cost, j, len_j);
                    reach = j + len_j - 1;
                }
            }
        }

        manager.update_cost_at_index(i, true);
        offset_prev = offset as i64;
        len_prev = len as i64;
    }

    // Phase 3: Backward trace — extract optimal path
    let mut path = Vec::new();
    let mut cur = pix_count as i64 - 1;
    while cur >= 0 {
        let step = manager.dist_array[cur as usize] as i64;
        if step == 0 {
            // Safety: shouldn't happen if DP is correct, but handle gracefully
            path.push(1u16);
            cur -= 1;
        } else {
            path.push(step as u16);
            cur -= step;
        }
    }
    path.reverse();

    // Phase 4: Follow chosen path to build final backward refs
    let mut refs = BackwardRefs::with_capacity(path.len());
    let mut emit_cache = if use_color_cache {
        Some(ColorCache::new(cache_bits))
    } else {
        None
    };

    let mut i = 0;
    for &step in &path {
        let step = step as usize;
        if step == 1 {
            // Literal or cache hit
            let argb_val = argb[i];
            if let Some(ref mut c) = emit_cache {
                if let Some(idx) = c.lookup(argb_val) {
                    refs.push(PixOrCopy::cache_idx(idx));
                } else {
                    refs.push(PixOrCopy::literal(argb_val));
                    c.insert(argb_val);
                }
            } else {
                refs.push(PixOrCopy::literal(argb_val));
            }
            i += 1;
        } else {
            // Copy with offset from hash chain
            let offset = hash_chain.offset(i);
            refs.push(PixOrCopy::copy(step as u16, offset as u32));
            if let Some(ref mut c) = emit_cache {
                for k in 0..step {
                    c.insert(argb[i + k]);
                }
            }
            i += step;
        }
    }

    refs
}

/// Try adding a literal (or cache hit) at position idx with cost tracking.
/// Matches libwebp's AddSingleLiteralWithCostModel.
#[inline]
fn add_single_literal_cost(
    argb: &[u32],
    cost_model: &CostModel,
    cache: &mut Option<ColorCache>,
    idx: usize,
    prev_cost: i64,
    costs: &mut [i64],
    dist_array: &mut [u16],
) {
    let color = argb[idx];
    let cost_val;

    if let Some(c) = cache {
        if let Some(cache_idx) = c.lookup(color) {
            // Cache hit: use cache cost, scaled by 68%
            cost_val = prev_cost + div_round(cost_model.cache_cost(cache_idx) * 68, 100);
        } else {
            // Cache miss: insert and use literal cost, scaled by 82%
            c.insert(color);
            cost_val = prev_cost + div_round(cost_model.literal_cost(color) * 82, 100);
        }
    } else {
        cost_val = prev_cost + div_round(cost_model.literal_cost(color) * 82, 100);
    }

    if cost_val < costs[idx] {
        costs[idx] = cost_val;
        dist_array[idx] = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_log2() {
        assert_eq!(fast_log2(0), 0);
        assert_eq!(fast_log2(1), 0);
        // log2(2) = 1.0
        let result = fast_log2(2) as f64 / (1u64 << LOG_2_PRECISION_BITS) as f64;
        assert!((result - 1.0).abs() < 0.01, "log2(2)={result}");
        // log2(8) = 3.0
        let result = fast_log2(8) as f64 / (1u64 << LOG_2_PRECISION_BITS) as f64;
        assert!((result - 3.0).abs() < 0.01, "log2(8)={result}");
        // log2(256) = 8.0
        let result = fast_log2(256) as f64 / (1u64 << LOG_2_PRECISION_BITS) as f64;
        assert!((result - 8.0).abs() < 0.01, "log2(256)={result}");
    }

    #[test]
    fn test_div_round() {
        assert_eq!(div_round(100, 100), 1);
        assert_eq!(div_round(150, 100), 2); // 1.5 rounds to 2
        assert_eq!(div_round(149, 100), 1); // 1.49 rounds to 1
        assert_eq!(div_round(-150, 100), -2);
    }

    #[test]
    fn test_counts_to_bit_estimates() {
        // Single symbol: all costs zero (trivial code)
        let counts = vec![0, 100, 0, 0];
        let estimates = counts_to_bit_estimates(&counts);
        assert!(estimates.iter().all(|&e| e == 0));

        // Uniform distribution: all non-zero symbols have equal cost
        let counts = vec![100, 100, 100, 100];
        let estimates = counts_to_bit_estimates(&counts);
        assert_eq!(estimates[0], estimates[1]);
        assert_eq!(estimates[1], estimates[2]);
    }

    #[test]
    fn test_cost_manager_basic() {
        // Verify CostManager can be created without panicking
        let histo_size = NUM_LITERAL_CODES + NUM_LENGTH_CODES;
        let model = CostModel {
            literal: vec![0u32; histo_size],
            red: vec![0u32; 256],
            blue: vec![0u32; 256],
            alpha: vec![0u32; 256],
            distance: vec![0u32; 40],
        };
        let manager = CostManager::new(100, &model);
        assert_eq!(manager.costs.len(), 100);
        assert_eq!(manager.dist_array.len(), 100);
        assert!(!manager.cache_intervals.is_empty());
    }

    #[test]
    fn test_push_interval_small() {
        // Small intervals (<10) should be serialized directly
        let histo_size = NUM_LITERAL_CODES + NUM_LENGTH_CODES;
        let model = CostModel {
            literal: vec![1u32 << LOG_2_PRECISION_BITS; histo_size],
            red: vec![0u32; 256],
            blue: vec![0u32; 256],
            alpha: vec![0u32; 256],
            distance: vec![0u32; 40],
        };
        let mut manager = CostManager::new(100, &model);
        manager.push_interval(0, 10, 5);

        // Costs at positions 10..15 should be updated
        for j in 10..15 {
            assert!(
                manager.costs[j] < i64::MAX,
                "position {j} should be updated"
            );
        }
        // Positions outside should be unchanged
        assert_eq!(manager.costs[9], i64::MAX);
        assert_eq!(manager.costs[15], i64::MAX);
    }
}
