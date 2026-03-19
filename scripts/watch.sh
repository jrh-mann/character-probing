#!/bin/bash
# Live status watcher. Run: bash scripts/watch.sh
TASK_DIR="/tmp/claude-0/-root-characterprobing/4766c109-8228-44e8-a595-e367dc94a112/tasks"
RESULTS="/workspace/characterprobing/results"

while true; do
    clear
    echo "=== $(date -u +%H:%M:%S) UTC ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo ""

    # Active task output (most recent .output file)
    latest=$(ls -t "$TASK_DIR"/*.output 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "RUNNING:"
        tail -c 3000 "$latest" 2>/dev/null | tr '\r' '\n' | grep -oP '(Train e\d+|Eval|Solving|=====).*' | tail -3
    fi
    echo ""

    # Completed results
    echo "--- Completed ---"
    for f in "$RESULTS"/*_per_layer_results.csv; do
        [ -f "$f" ] && basename "$f" _per_layer_results.csv
    done 2>/dev/null
    echo ""

    # Reeval results
    reeval=$(ls "$RESULTS"/*_reeval_results.csv 2>/dev/null)
    if [ -n "$reeval" ]; then
        echo "--- Reeval ---"
        for f in $reeval; do
            basename "$f" _reeval_results.csv
        done
    fi
    echo ""

    # Attention probe results
    attn=$(ls "$RESULTS"/*_attn_results.csv 2>/dev/null)
    if [ -n "$attn" ]; then
        echo "--- Attention Probes ---"
        for f in $attn; do
            basename "$f" _attn_results.csv
        done
    fi

    sleep 5
done
