#!/bin/bash
# Live monitoring: GPU stats, completed models, and graph regeneration.
# Run in background: bash scripts/monitor.sh &
# Check status: cat results/status.txt

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$REPO_DIR/results"
STATUS="$RESULTS/status.txt"

mkdir -p "$RESULTS"

while true; do
    {
        echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC ==="
        echo ""

        # GPU stats (if available)
        if command -v nvidia-smi &>/dev/null; then
            echo "── GPU ──"
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"
            echo ""
        fi

        # Disk usage
        echo "── Disk ──"
        df -h "$REPO_DIR" 2>/dev/null | tail -1 | awk '{print "  Used: " $3 " / " $2 " (" $5 " full)"}'
        echo ""

        # Completed probe results
        echo "── Completed Probes ──"
        completed=0
        for f in "$RESULTS"/*_per_layer_results.csv; do
            if [ -f "$f" ]; then
                basename "$f" _per_layer_results.csv
                completed=$((completed + 1))
            fi
        done 2>/dev/null
        echo "  ($completed completed)"
        echo ""

        # Completed attention probes
        echo "── Completed Attention Probes ──"
        attn_done=0
        for f in "$RESULTS"/*_attn_results.csv; do
            if [ -f "$f" ]; then
                basename "$f" _attn_results.csv
                attn_done=$((attn_done + 1))
            fi
        done 2>/dev/null
        echo "  ($attn_done completed)"
        echo ""

        # Currently running (check for active log being written)
        echo "── Currently Running ──"
        latest_log=$(ls -t "$RESULTS/logs"/*.log 2>/dev/null | head -1)
        if [ -n "$latest_log" ]; then
            log_name=$(basename "$latest_log" .log)
            # Check if log was modified in the last 2 minutes
            if find "$latest_log" -mmin -2 2>/dev/null | grep -q .; then
                echo "  Active: $log_name"
                # Show last progress line
                tail -c 2000 "$latest_log" 2>/dev/null | tr '\r' '\n' | grep -oP '(Train e\d+|Eval|Solving|TRAINING|EVALUATION).*' | tail -1
            else
                echo "  (idle)"
            fi
        else
            echo "  (no logs yet)"
        fi

    } > "$STATUS" 2>/dev/null

    # Regenerate live training graphs (suppress errors — may fail if no data yet)
    python "$REPO_DIR/scripts/plot_live.py" > /dev/null 2>&1 || true

    sleep 60
done
