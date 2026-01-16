#!/bin/bash
# Linear Merge Script for Linux/Mac
# Prerequisites: pip install mergekit

echo "========================================"
echo "Linear Merge: Qwen3-0.6B Experts"
echo "========================================"

OUTPUT_DIR="./merged_linear"
CONFIG_FILE="merge_config.yaml"

# Check if mergekit is installed
if ! command -v mergekit-yaml &> /dev/null; then
    echo "mergekit not found. Installing..."
    pip install mergekit
fi

echo ""
echo "Starting merge..."
echo "Output directory: $OUTPUT_DIR"
echo ""

mergekit-yaml "$CONFIG_FILE" "$OUTPUT_DIR" --cuda --low-cpu-memory

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Merge completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
    echo "========================================"
else
    echo ""
    echo "Merge failed with error code $?"
fi
