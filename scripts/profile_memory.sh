#!/usr/bin/env bash
# Profile memory usage with heaptrack and extract peak memory data
#
# Usage: ./scripts/profile_memory.sh
#
# Outputs: memory_profile_data.csv

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Memory Profiling with Heaptrack ===${NC}\n"

# Check if heaptrack is installed
if ! command -v heaptrack &> /dev/null; then
    echo -e "${RED}Error: heaptrack not found${NC}"
    echo "Install with: sudo apt-get install heaptrack"
    exit 1
fi

# Build the profiler in release mode
echo -e "${YELLOW}Building profiler...${NC}"
cargo build --release --example profile_memory

OUTPUT_DIR="${ZENWEBP_OUTPUT_DIR:-/mnt/v/output/zenwebp}/memory_profile"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Output directory: $OUTPUT_DIR${NC}\n"

# Profile encoding
echo -e "${YELLOW}Profiling encoding...${NC}"
heaptrack --output "$OUTPUT_DIR/encode.heaptrack" \
    target/release/examples/profile_memory encode > "$OUTPUT_DIR/encode_raw.csv"

# Profile decoding
echo -e "${YELLOW}Profiling decoding...${NC}"
heaptrack --output "$OUTPUT_DIR/decode.heaptrack" \
    target/release/examples/profile_memory decode > "$OUTPUT_DIR/decode_raw.csv"

# Parse heaptrack output to extract peak memory
echo -e "${YELLOW}Extracting peak memory data...${NC}"

# Function to extract peak memory from heaptrack output
extract_peak_memory() {
    local heaptrack_file=$1
    # heaptrack prints "peak heap memory consumption: X.XX MB"
    # or outputs detailed data in the .gz file

    # For now, use heaptrack_print to analyze
    if [ -f "${heaptrack_file}.gz" ]; then
        heaptrack_print "${heaptrack_file}.gz" | \
            grep -i "peak heap" | \
            grep -oP '\d+\.\d+\s+[KMG]B' | \
            head -1
    else
        echo "N/A"
    fi
}

encode_peak=$(extract_peak_memory "$OUTPUT_DIR/encode.heaptrack")
decode_peak=$(extract_peak_memory "$OUTPUT_DIR/decode.heaptrack")

echo -e "${GREEN}Peak memory usage:${NC}"
echo "  Encode: $encode_peak"
echo "  Decode: $decode_peak"

# Combine CSV data with peak memory info
echo -e "${YELLOW}Generating combined dataset...${NC}"

# We'll need to run heaptrack separately for each configuration to get per-run peak memory
# For now, just preserve the CSV output
cp "$OUTPUT_DIR/encode_raw.csv" "$OUTPUT_DIR/encode_profile.csv"
cp "$OUTPUT_DIR/decode_raw.csv" "$OUTPUT_DIR/decode_profile.csv"

echo -e "${GREEN}Profile data saved to:${NC}"
echo "  $OUTPUT_DIR/encode_profile.csv"
echo "  $OUTPUT_DIR/decode_profile.csv"
echo "  $OUTPUT_DIR/encode.heaptrack.gz"
echo "  $OUTPUT_DIR/decode.heaptrack.gz"

echo -e "\n${GREEN}To analyze, run:${NC}"
echo "  heaptrack_gui $OUTPUT_DIR/encode.heaptrack.gz"
echo "  heaptrack_gui $OUTPUT_DIR/decode.heaptrack.gz"
echo ""
echo -e "${GREEN}Or build estimation formulas with:${NC}"
echo "  cargo run --release --example analyze_profiles"
