#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="$REPO_DIR/hf_cache"

# ── Header ────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           character-probing — automated setup               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Repo:     $REPO_DIR"
echo "  HF cache: $HF_HOME"
echo "  Started:  $(date)"
echo ""

# ── System info ──────────────────────────────────────────────────────
echo "── System ──"
echo "  Python:   $(python3 --version 2>&1)"
echo "  CPUs:     $(nproc 2>/dev/null || echo unknown)"
echo "  RAM:      $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo unknown)"
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
else
    echo "  GPU:      not detected"
fi
echo "  Disk:     $(df -h "$REPO_DIR" 2>/dev/null | awk 'NR==2{print $4 " free / " $2 " total"}')"
echo ""

# ── 1. Install PyTorch + dependencies ────────────────────────────────
echo "━━━ [1/6] Installing dependencies ━━━"
# Install PyTorch first (picks up CUDA automatically)
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  PyTorch already installed with CUDA support ✓"
else
    echo "  Installing PyTorch..."
    pip install -q torch --index-url https://download.pytorch.org/whl/cu124 2>/dev/null \
        || pip install -q torch  # fallback to default
    python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
fi
echo "  Installing other dependencies..."
pip install -q -r "$REPO_DIR/requirements.txt"
# Ensure system tools are available
if ! command -v unzip &>/dev/null; then
    echo "  Installing unzip..."
    apt-get update -qq && apt-get install -y -qq unzip > /dev/null 2>&1
fi
echo "  Done ✓"

# HuggingFace login (needed for gated models like Llama)
if [ -n "${HF_TOKEN:-}" ]; then
    echo "  Logging into HuggingFace with HF_TOKEN..."
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
elif ! huggingface-cli whoami &>/dev/null; then
    echo ""
    echo "  ⚠ Not logged into HuggingFace. Llama models will fail."
    echo "    Set HF_TOKEN env var or run: huggingface-cli login"
    echo ""
fi

# ── 2. Download Blog Authorship Corpus ───────────────────────────────
echo ""
echo "━━━ [2/6] Downloading data ━━━"
RAW_DIR="$REPO_DIR/data/raw/blogs"
if [ -d "$RAW_DIR" ] && [ -n "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
    echo "  Raw data already present ($(ls "$RAW_DIR"/*.xml 2>/dev/null | wc -l) XML files) ✓"
else
    echo "  Downloading Blog Authorship Corpus..."
    mkdir -p "$REPO_DIR/data/raw"
    cd "$REPO_DIR/data/raw"
    rm -f blogs.zip
    if ! wget -q --show-progress "https://u.cs.biu.ac.il/~koppel/blogs/blogs.zip" -O blogs.zip; then
        rm -f blogs.zip
        echo "  ✗ Download failed."
        echo "    Manually download from https://u.cs.biu.ac.il/~koppel/blogs/blogs.zip"
        echo "    and extract to $RAW_DIR"
        exit 1
    fi
    unzip -qo blogs.zip -d blogs_tmp/
    rm blogs.zip
    mkdir -p blogs
    find blogs_tmp -name '*.xml' -exec mv {} blogs/ \;
    rm -rf blogs_tmp
    cd "$REPO_DIR"
    n_files=$(ls "$RAW_DIR"/*.xml 2>/dev/null | wc -l)
    if [ "$n_files" -eq 0 ]; then
        echo "  ✗ No XML files found after extraction."
        exit 1
    fi
    echo "  Downloaded and extracted $n_files XML files ✓"
fi

# ── 3. Preprocess data ───────────────────────────────────────────────
echo ""
echo "━━━ [3/6] Preprocessing data ━━━"
PARQUET="$REPO_DIR/data/processed/blog_corpus.parquet"
if [ -f "$PARQUET" ]; then
    echo "  Preprocessed parquet already exists ✓"
else
    echo "  Parsing XML, cleaning, filtering, balancing, splitting..."
    python3 "$REPO_DIR/scripts/01_preprocess_data.py"
    echo "  Done ✓"
fi

# ── 4. Check existing progress ───────────────────────────────────────
echo ""
echo "━━━ [4/6] Checking existing progress ━━━"
completed=$(find "$REPO_DIR/results" -maxdepth 1 -name '*_per_layer_results.csv' 2>/dev/null | wc -l)
if [ "$completed" -gt 0 ]; then
    echo "  Found $completed completed models — will resume from where we left off."
    echo "  (To start fresh, run: rm -rf results/* probes/* figures/*)"
else
    echo "  No previous results found — starting fresh."
fi
mkdir -p "$REPO_DIR/results/logs" "$REPO_DIR/figures"

# ── 5. Start monitoring + run experiments ────────────────────────────
echo ""
echo "━━━ [5/6] Running experiments ━━━"
echo ""
echo "  15 models × (7 EMA + 4 MLP + 1 shuffled + ridge + mass-mean + attention)"
echo "  Estimated runtime: 8-10 hours on A100-80GB"
echo ""
echo "  Monitor progress:"
echo "    cat results/status.txt        # quick status"
echo "    tail -f results/logs/overnight.log  # live output"
echo "    ls figures/                    # live-updating graphs"
echo ""

# Start background monitor (regenerates status.txt + figures every 60s)
bash "$REPO_DIR/scripts/monitor.sh" &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

# Run all experiments — output goes to terminal AND log file
python3 "$REPO_DIR/scripts/overnight.py" 2>&1 | tee "$REPO_DIR/results/logs/overnight.log"

# ── 6. Final figures ─────────────────────────────────────────────────
echo ""
echo "━━━ [6/6] Generating final figures ━━━"
python3 "$REPO_DIR/scripts/05_make_figures.py" 2>&1 && echo "  Publication figures ✓" || echo "  ⚠ Figure generation failed (non-fatal)"
python3 "$REPO_DIR/scripts/plot_position_accuracy.py" 2>&1 && echo "  Position accuracy plots ✓" || echo "  ⚠ Position plots failed (non-fatal)"
python3 "$REPO_DIR/scripts/plot_live.py" 2>&1 && echo "  Scaling curves ✓" || echo "  ⚠ Scaling curves failed (non-fatal)"

# ── Done ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                      ALL DONE                               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Results:  results/*_per_layer_results.csv                  ║"
echo "║  Figures:  figures/*.png                                    ║"
echo "║  Logs:     results/logs/*.log                               ║"
echo "║  Probes:   probes/*/                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Finished: $(date)"

# Print quick summary of results
if ls "$REPO_DIR/results/"*_per_layer_results.csv &>/dev/null; then
    echo ""
    echo "── Quick Results ──"
    completed=$(find "$REPO_DIR/results" -maxdepth 1 -name '*_per_layer_results.csv' 2>/dev/null | wc -l)
    echo "  $completed models completed"
    echo "  Run 'python3 scripts/plot_live.py' for detailed analysis"
fi
