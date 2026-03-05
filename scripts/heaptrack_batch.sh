#!/usr/bin/env bash
# Run heaptrack on various encode/decode configurations
set -euo pipefail

OUTPUT_DIR="${ZENWEBP_OUTPUT_DIR:-/mnt/v/output/zenwebp}/heaptrack_data"
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${ZENWEBP_HEAPTRACK_BIN:-$SCRIPT_DIR/target/release/examples/heaptrack_test}"

echo "width,height,pixels,mode,method,quality,peak_heap_mb" > "$OUTPUT_DIR/heaptrack_results.csv"

# Test configurations
SIZES=(
    "512 512"
    "1024 1024"
    "1920 1080"
    "2560 1440"
    "3840 2160"
)

# Encode tests
for size in "${SIZES[@]}"; do
    read -r width height <<< "$size"
    pixels=$((width * height))

    for method in 0 2 4 6; do
        for quality in 50 75 90; do
            echo "Testing encode ${width}x${height} method=$method quality=$quality..."

            HEAP_FILE="$OUTPUT_DIR/encode_${width}x${height}_m${method}_q${quality}.heaptrack"

            heaptrack --output "$HEAP_FILE" "$BINARY" encode "$width" "$height" "$method" "$quality" 2>&1 | grep -i "peak"

            # Extract peak memory from the .gz file
            if [ -f "${HEAP_FILE}.gz" ]; then
                peak_mb=$(heaptrack_print "${HEAP_FILE}.gz" 2>/dev/null | \
                    grep -i "peak heap memory" | \
                    grep -oP '\d+\.\d+' | \
                    head -1)

                if [ -n "$peak_mb" ]; then
                    echo "$width,$height,$pixels,encode,$method,$quality,$peak_mb" >> "$OUTPUT_DIR/heaptrack_results.csv"
                fi

                # Clean up to save space
                rm -f "${HEAP_FILE}.gz"
            fi
        done
    done
done

# Decode tests
for size in "${SIZES[@]}"; do
    read -r width height <<< "$size"
    pixels=$((width * height))

    echo "Testing decode ${width}x${height}..."

    HEAP_FILE="$OUTPUT_DIR/decode_${width}x${height}.heaptrack"

    heaptrack --output "$HEAP_FILE" "$BINARY" decode "$width" "$height" 2>&1 | grep -i "peak"

    if [ -f "${HEAP_FILE}.gz" ]; then
        peak_mb=$(heaptrack_print "${HEAP_FILE}.gz" 2>/dev/null | \
            grep -i "peak heap memory" | \
            grep -oP '\d+\.\d+' | \
            head -1)

        if [ -n "$peak_mb" ]; then
            echo "$width,$height,$pixels,decode,0,0,$peak_mb" >> "$OUTPUT_DIR/heaptrack_results.csv"
        fi

        rm -f "${HEAP_FILE}.gz"
    fi
done

echo "Results saved to: $OUTPUT_DIR/heaptrack_results.csv"
cat "$OUTPUT_DIR/heaptrack_results.csv"
