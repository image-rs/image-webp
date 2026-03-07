//! Meta-Huffman encoding for spatially-varying codes.
//!
//! Builds per-tile histograms from backward references, clusters them
//! using entropy-based greedy merging (matching libwebp), and produces
//! a histogram index map for spatially-varying Huffman codes.

use alloc::vec;
use alloc::vec::Vec;

use super::entropy::{HistogramCosts, compute_histogram_cost, get_combined_histogram_cost};
use super::histogram::Histogram;
use super::types::{BackwardRefs, PixOrCopy, subsample_size};

/// Maximum Huffman group count for greedy combining.
const MAX_HISTO_GREEDY: usize = 100;

/// Number of entropy bins for initial binning (4×4×4 = 64 for 3D binning).
const NUM_PARTITIONS: usize = 4;
const BIN_SIZE: usize = NUM_PARTITIONS * NUM_PARTITIONS * NUM_PARTITIONS; // 64

/// Meta-Huffman configuration.
pub struct MetaHuffmanInfo {
    /// Histogram bits (block size = 2^bits pixels).
    pub histo_bits: u8,
    /// Number of final histogram groups.
    pub num_histograms: usize,
    /// Mapping from tile index to histogram group index.
    pub histogram_symbols: Vec<u16>,
    /// Final clustered histograms (one per group).
    pub histograms: Vec<Histogram>,
    /// Cached costs for each final histogram.
    pub costs: Vec<HistogramCosts>,
    /// Image width (for sub-image dimension computation).
    pub image_width: usize,
    /// Image height (for sub-image dimension computation).
    pub image_height: usize,
}

/// Build per-tile histograms from backward references.
///
/// Each tile covers a (2^histo_bits × 2^histo_bits) block of pixels.
/// Tokens are assigned to tiles based on the pixel position they encode.
/// Build per-tile histograms from backward references.
///
/// Each tile covers a (2^histo_bits × 2^histo_bits) block of pixels.
/// Tokens are assigned to tiles based on the pixel position they encode.
/// Assumes distances in refs are already plane codes (from apply_2d_locality).
fn build_tile_histograms(
    refs: &BackwardRefs,
    width: usize,
    _height: usize,
    histo_bits: u8,
    cache_bits: u8,
) -> Vec<Histogram> {
    let histo_xsize = subsample_size(width as u32, histo_bits) as usize;
    let histo_ysize = subsample_size(_height as u32, histo_bits) as usize;
    let num_tiles = histo_xsize * histo_ysize;

    let mut histos: Vec<Histogram> = (0..num_tiles).map(|_| Histogram::new(cache_bits)).collect();

    let mut x = 0usize;
    let mut y = 0usize;

    for token in refs.iter() {
        let tile_idx = (y >> histo_bits) * histo_xsize + (x >> histo_bits);
        debug_assert!(tile_idx < num_tiles);

        match *token {
            PixOrCopy::Literal(argb) => {
                histos[tile_idx].add_literal(argb);
                x += 1;
                if x >= width {
                    x = 0;
                    y += 1;
                }
            }
            PixOrCopy::CacheIdx(idx) => {
                histos[tile_idx].add_cache_idx(idx);
                x += 1;
                if x >= width {
                    x = 0;
                    y += 1;
                }
            }
            PixOrCopy::Copy { len, dist } => {
                // Distances are already plane codes
                histos[tile_idx].add_copy(len, dist);

                for _ in 0..len {
                    x += 1;
                    if x >= width {
                        x = 0;
                        y += 1;
                    }
                }
            }
        }
    }

    histos
}

/// Get bin ID for entropy-based binning (matching libwebp's GetBinIdForEntropy).
fn get_bin_id_for_entropy(min: u64, max: u64, val: u64) -> usize {
    if min == max {
        return 0;
    }
    // Map val to [0, NUM_PARTITIONS-1] linearly
    let range = max - min;
    let offset = val.saturating_sub(min);
    let bin = ((NUM_PARTITIONS as u64 - 1) * offset) / range;
    bin.min(NUM_PARTITIONS as u64 - 1) as usize
}

/// Cluster histograms using entropy binning + greedy combining.
/// Matches libwebp's HistogramCombineEntropyBin + HistogramCombineGreedy.
fn cluster_histograms(
    tile_histos: &[Histogram],
    quality: u8,
    cache_bits: u8,
) -> (Vec<Histogram>, Vec<u16>) {
    let n = tile_histos.len();

    if n <= 1 {
        if n == 1 {
            return (vec![tile_histos[0].clone()], vec![0]);
        }
        return (Vec::new(), Vec::new());
    }

    // Compute initial costs for all histograms
    let mut costs: Vec<HistogramCosts> = tile_histos.iter().map(compute_histogram_cost).collect();

    // Track which histograms are active and their mapping
    let mut active: Vec<bool> = vec![true; n];
    let mut mapping: Vec<usize> = (0..n).collect(); // tile -> cluster index
    let mut histos: Vec<Histogram> = tile_histos.to_vec();

    // Phase 1: Entropy binning - assign each histogram to an entropy bin
    // Find min/max costs for literal, red, blue types
    let mut lit_min = u64::MAX;
    let mut lit_max = 0u64;
    let mut red_min = u64::MAX;
    let mut red_max = 0u64;
    let mut blue_min = u64::MAX;
    let mut blue_max = 0u64;

    for (i, cost) in costs.iter().enumerate() {
        if !active[i] {
            continue;
        }
        lit_min = lit_min.min(cost.per_type[0]);
        lit_max = lit_max.max(cost.per_type[0]);
        red_min = red_min.min(cost.per_type[1]);
        red_max = red_max.max(cost.per_type[1]);
        blue_min = blue_min.min(cost.per_type[2]);
        blue_max = blue_max.max(cost.per_type[2]);
    }

    // Assign bin IDs (3D: literal × red × blue)
    let mut bin_ids: Vec<usize> = Vec::with_capacity(n);
    for cost in &costs {
        let lit_bin = get_bin_id_for_entropy(lit_min, lit_max, cost.per_type[0]);
        let red_bin = get_bin_id_for_entropy(red_min, red_max, cost.per_type[1]);
        let blue_bin = get_bin_id_for_entropy(blue_min, blue_max, cost.per_type[2]);
        bin_ids
            .push(lit_bin * NUM_PARTITIONS * NUM_PARTITIONS + red_bin * NUM_PARTITIONS + blue_bin);
    }

    // Phase 2: Merge within entropy bins
    let num_active = active.iter().filter(|&&a| a).count();
    let num_bins = BIN_SIZE.min(num_active);
    let do_entropy_combine = num_active > num_bins * 2 && quality < 100;

    if do_entropy_combine {
        // Track first histogram per bin
        let mut bin_first: Vec<Option<usize>> = vec![None; BIN_SIZE];

        // Combine cost factor (matching libwebp's GetCombineCostFactor)
        let mut combine_cost_factor = 16i64;
        if quality < 90 {
            if num_active > 256 {
                combine_cost_factor /= 2;
            }
            if num_active > 512 {
                combine_cost_factor /= 2;
            }
            if num_active > 1024 {
                combine_cost_factor /= 2;
            }
            if quality <= 50 {
                combine_cost_factor /= 2;
            }
        }

        for i in 0..n {
            if !active[i] {
                continue;
            }

            let bin_id = bin_ids[i];
            if let Some(first) = bin_first[bin_id] {
                // Try to merge with first histogram in bin
                let bit_cost = costs[first].total;
                let threshold = bit_cost + costs[i].total;
                let cost_thresh_val = threshold
                    .saturating_sub(
                        div_round_i64(bit_cost as i64 * combine_cost_factor, 100) as u64
                    );

                if get_combined_histogram_cost(
                    &histos[first],
                    &costs[first],
                    &histos[i],
                    &costs[i],
                    cost_thresh_val,
                )
                .is_some()
                {
                    // Merge i into first
                    let i_histo = histos[i].clone();
                    histos[first].add(&i_histo);
                    costs[first] = compute_histogram_cost(&histos[first]);
                    active[i] = false;

                    // Remap all tiles pointing to i → first
                    for m in mapping.iter_mut() {
                        if *m == i {
                            *m = first;
                        }
                    }
                }
            } else {
                bin_first[bin_id] = Some(i);
            }
        }
    }

    // Phase 2b: Stochastic combining (matching libwebp's HistogramCombineStochastic).
    // Uses random pair sampling with Lehmer RNG, finding BEST pair per iteration.
    let target_size = {
        let q3 = (quality as u64) * (quality as u64) * (quality as u64);
        let t = 1 + div_round_u64(q3 * (MAX_HISTO_GREEDY as u64 - 1), 100u64 * 100 * 100);
        t.min(MAX_HISTO_GREEDY as u64) as usize
    };

    let mut num_active = active.iter().filter(|&&a| a).count();

    if num_active > target_size {
        // Lehmer RNG matching libwebp (multiplier=48271, modulus=2^31-1)
        let mut seed: u32 = 1;
        let outer_iters = num_active;
        let num_tries_no_success = outer_iters / 2;
        let mut tries_with_no_success = 0usize;

        for _iter in 0..outer_iters {
            if num_active < 2 || num_active <= target_size {
                break;
            }
            tries_with_no_success += 1;
            if tries_with_no_success >= num_tries_no_success {
                break;
            }

            // Build active index list for this iteration
            let active_indices: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
            let size = active_indices.len();
            if size < 2 {
                break;
            }

            // Try size/2 random pairs, find the BEST merge
            let num_tries = size / 2;
            let mut best_savings = 0i64; // Must be strictly negative to merge
            let mut best_i = 0usize;
            let mut best_j = 0usize;

            let rand_range = (size as u64) * (size as u64 - 1);

            for _ in 0..num_tries {
                // Lehmer RNG: seed = (seed * 48271) % (2^31 - 1)
                seed = ((seed as u64 * 48271u64) % 2147483647u64) as u32;
                let tmp = (seed as u64) % rand_range;
                let idx1 = (tmp / (size as u64 - 1)) as usize;
                let mut idx2 = (tmp % (size as u64 - 1)) as usize;
                if idx2 >= idx1 {
                    idx2 += 1;
                }

                let i = active_indices[idx1];
                let j = active_indices[idx2];

                // Compute cost difference: combined - (a + b)
                // Use best_savings as threshold for early bail-out
                let sum_cost = costs[i].total as i64 + costs[j].total as i64;
                let threshold = if best_savings < 0 {
                    (sum_cost + best_savings) as u64
                } else {
                    costs[i].total + costs[j].total
                };

                if let Some(combined_cost) = get_combined_histogram_cost(
                    &histos[i], &costs[i], &histos[j], &costs[j], threshold,
                ) {
                    let savings = combined_cost as i64 - sum_cost;
                    if savings < best_savings {
                        best_savings = savings;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            // Merge the best pair if beneficial
            if best_savings < 0 {
                let j_histo = histos[best_j].clone();
                histos[best_i].add(&j_histo);
                costs[best_i] = compute_histogram_cost(&histos[best_i]);
                active[best_j] = false;

                for m in mapping.iter_mut() {
                    if *m == best_j {
                        *m = best_i;
                    }
                }

                num_active -= 1;
                tries_with_no_success = 0; // Reset on success
            }
        }
    }

    // Phase 3: Greedy combining — runs when set is small enough for O(n²)
    // Merges any beneficial pair, even below target_size (matching libwebp).
    num_active = active.iter().filter(|&&a| a).count();

    while num_active > 1 && num_active <= MAX_HISTO_GREEDY {
        let mut best_i = 0usize;
        let mut best_j = 0usize;
        let mut best_savings = i64::MIN;

        let active_indices: Vec<usize> = (0..n).filter(|&i| active[i]).collect();

        for (ai, &i) in active_indices.iter().enumerate() {
            for &j in active_indices[(ai + 1)..].iter() {
                let threshold = costs[i].total + costs[j].total;
                if let Some(combined_cost) = get_combined_histogram_cost(
                    &histos[i], &costs[i], &histos[j], &costs[j], threshold,
                ) {
                    let savings =
                        costs[i].total as i64 + costs[j].total as i64 - combined_cost as i64;
                    if savings > best_savings {
                        best_savings = savings;
                        best_i = i;
                        best_j = j;
                    }
                }
            }
        }

        if best_savings <= 0 {
            break;
        }

        let j_histo = histos[best_j].clone();
        histos[best_i].add(&j_histo);
        costs[best_i] = compute_histogram_cost(&histos[best_i]);
        active[best_j] = false;

        for m in mapping.iter_mut() {
            if *m == best_j {
                *m = best_i;
            }
        }

        num_active -= 1;
    }

    // Phase 4: Remap - find best cluster for each original tile histogram
    let active_indices: Vec<usize> = (0..n).filter(|&i| active[i]).collect();

    // Cache tile histogram costs to avoid recomputation
    let tile_costs: Vec<HistogramCosts> = tile_histos.iter().map(compute_histogram_cost).collect();

    for tile_idx in 0..n {
        let mut best_cluster = mapping[tile_idx];
        let mut best_cost = i64::MAX;

        for &cluster_idx in &active_indices {
            if let Some(combined_cost) = get_combined_histogram_cost(
                &histos[cluster_idx],
                &costs[cluster_idx],
                &tile_histos[tile_idx],
                &tile_costs[tile_idx],
                u64::MAX,
            ) {
                let cost = combined_cost as i64 - costs[cluster_idx].total as i64;
                if cost < best_cost {
                    best_cost = cost;
                    best_cluster = cluster_idx;
                }
            }
        }

        mapping[tile_idx] = best_cluster;
    }

    // Phase 5: Rebuild final histograms from remapped tiles
    let mut final_histos: Vec<Histogram> = active_indices
        .iter()
        .map(|_| Histogram::new(cache_bits))
        .collect();

    // Create index mapping from cluster_idx → final histogram index
    let mut cluster_to_final: Vec<Option<u16>> = vec![None; n];
    for (final_idx, &cluster_idx) in active_indices.iter().enumerate() {
        cluster_to_final[cluster_idx] = Some(final_idx as u16);
    }

    // Build final symbols and accumulate histograms
    let mut symbols: Vec<u16> = Vec::with_capacity(n);
    for tile_idx in 0..n {
        let cluster = mapping[tile_idx];
        let final_idx = cluster_to_final[cluster].unwrap_or(0);
        symbols.push(final_idx);
        final_histos[final_idx as usize].add(&tile_histos[tile_idx]);
    }

    // Recompute costs for final histograms
    let final_costs: Vec<HistogramCosts> =
        final_histos.iter().map(compute_histogram_cost).collect();

    // Return results
    let _ = final_costs; // Will be recomputed by caller if needed
    (final_histos, symbols)
}

#[inline]
fn div_round_i64(num: i64, den: i64) -> i64 {
    if num >= 0 {
        (num + den / 2) / den
    } else {
        (num - den / 2) / den
    }
}

#[inline]
fn div_round_u64(num: u64, den: u64) -> u64 {
    (num + den / 2) / den
}

/// Build meta-Huffman info from backward references.
/// Returns info for spatially-varying Huffman codes.
pub fn build_meta_huffman(
    refs: &BackwardRefs,
    width: usize,
    height: usize,
    histo_bits: u8,
    cache_bits: u8,
    quality: u8,
) -> MetaHuffmanInfo {
    let histo_bits = histo_bits.clamp(2, 8);

    // Build per-tile histograms from backward reference tokens
    let tile_histos = build_tile_histograms(refs, width, height, histo_bits, cache_bits);

    // Cluster histograms
    let (histograms, symbols) = cluster_histograms(&tile_histos, quality, cache_bits);

    // Compute final costs
    let costs = histograms.iter().map(compute_histogram_cost).collect();

    MetaHuffmanInfo {
        histo_bits,
        num_histograms: histograms.len(),
        histogram_symbols: symbols,
        histograms,
        costs,
        image_width: width,
        image_height: height,
    }
}

/// Build a single-histogram MetaHuffmanInfo (no spatial variation).
/// Assumes distances in refs are already plane codes (from apply_2d_locality).
pub fn build_single_histogram(refs: &BackwardRefs, cache_bits: u8) -> MetaHuffmanInfo {
    let histogram = Histogram::from_refs(refs, cache_bits);
    let costs = vec![compute_histogram_cost(&histogram)];

    MetaHuffmanInfo {
        histo_bits: 0,
        num_histograms: 1,
        histogram_symbols: Vec::new(),
        histograms: vec![histogram],
        costs,
        image_width: 0,
        image_height: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::make_argb;
    use super::*;

    #[test]
    fn test_single_histogram() {
        let mut refs = BackwardRefs::new();
        refs.push(PixOrCopy::literal(make_argb(255, 128, 64, 32)));
        refs.push(PixOrCopy::literal(make_argb(255, 128, 64, 32)));
        let info = build_single_histogram(&refs, 0);
        assert_eq!(info.num_histograms, 1);
        assert!(info.histogram_symbols.is_empty());
    }

    #[test]
    fn test_build_tile_histograms() {
        // 4x4 image with bits=2 → tile size 4x4, so 1 tile
        let mut refs = BackwardRefs::new();
        for _ in 0..16 {
            refs.push(PixOrCopy::literal(make_argb(255, 128, 64, 32)));
        }
        let histos = build_tile_histograms(&refs, 4, 4, 2, 0);
        assert_eq!(histos.len(), 1); // 4/4 = 1 tile for bits=2
    }

    #[test]
    fn test_cluster_identical() {
        // Two identical histograms should merge into one
        let mut h1 = Histogram::new(0);
        let mut h2 = Histogram::new(0);
        for _ in 0..100 {
            h1.add_literal(make_argb(255, 128, 64, 32));
            h2.add_literal(make_argb(255, 128, 64, 32));
        }
        let (groups, symbols) = cluster_histograms(&[h1, h2], 75, 0);
        assert_eq!(groups.len(), 1);
        assert_eq!(symbols.len(), 2);
        assert_eq!(symbols[0], symbols[1]);
    }
}
