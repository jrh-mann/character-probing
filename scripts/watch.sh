#!/bin/bash
# Live status watcher. Run: bash scripts/watch.sh
while true; do
    clear
    echo "=== $(date -u +%H:%M:%S) UTC ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo ""

    # Current model from overnight log
    current=$(grep "Starting" /workspace/characterprobing/results/logs/overnight.log 2>/dev/null | tail -1 | grep -oP 'Qwen2\.5-\S+')
    if [ -n "$current" ]; then
        f="/workspace/characterprobing/results/logs/${current}.log"
        echo "RUNNING: $current"
        # Use tail -c to get last chunk, tr to strip carriage returns, grep for progress
        tail -c 5000 "$f" 2>/dev/null | tr '\r' '\n' | grep -oP '(Train e\d+|Eval|Solving).*' | tail -1
    fi
    echo ""

    # Completed
    echo "--- Completed ---"
    for f in /workspace/characterprobing/results/*_per_layer_results.csv; do
        [ -f "$f" ] && basename "$f" _per_layer_results.csv
    done 2>/dev/null
    echo ""

    # Log
    grep "==>" /workspace/characterprobing/results/logs/overnight.log 2>/dev/null | tail -3

    sleep 5
done
