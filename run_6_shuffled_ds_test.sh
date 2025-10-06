#!/bin/bash

# Run 19 shuffled datasets test with best parameter configurations
# Usage: ./run_6_shuffled_ds_test.sh [NUM_WORKERS] [EXPERIMENT_NAME]

set -e

# Parse arguments  
NUM_WORKERS=${1:-4}
EXPERIMENT_NAME=${2:-"shuffled"}
DISABLE_CACHE=${3:-"false"}  # Third argument to disable cache

# Configuration
#PYTHON_SCRIPT="new_test_scripts/run_6_shuffled_ds_tests.py"
PYTHON_SCRIPT="run_6_shuffled_ds_tests.py"
RESULTS_DIR="res/cbresults_${EXPERIMENT_NAME}"
# PLOTS_DIR removed - only CSV generation is performed

echo "🎯 19 Shuffled Datasets Test with Best Configurations"
echo "====================================================="
echo "Workers: ${NUM_WORKERS}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Disable Cache: ${DISABLE_CACHE}"
echo ""

# Check if shuffled datasets exist
SHUFFLED_DIR="datasets_shuffled"
if [ ! -d "$SHUFFLED_DIR" ]; then
    echo "❌ Error: Shuffled datasets directory not found: $SHUFFLED_DIR"
    echo "   Please run the dataset shuffling script first."
    exit 1
fi

# Check if required datasets exist
REQUIRED_DATASETS=("ds_1006_2" "ds_1012_2" "ds_1015_2" "ds_1062_2.0" "ds_1073_2.0" "ds_1084_3" "ds_339_3" "ds_346_2" "ds_457_4" "ds_476_2" "ds_729_2" "ds_785_2" "ds_835_2" "ds_848_2" "ds_874_2" "ds_905_2" "ds_928_2" "ds_964_2")
MISSING_DATASETS=()

echo "🔍 Checking for required shuffled datasets..."

for dataset in "${REQUIRED_DATASETS[@]}"; do
    # Check for shuffled versions (shuf0 through shuf9)
    found=false
    for i in {0..9}; do
        if [ -f "${SHUFFLED_DIR}/${dataset}_shuf${i}.vw.gz" ]; then
            found=true
            break
        fi
    done
    
    if [ "$found" = true ]; then
        echo "  ✅ $dataset: Found shuffled versions"
    else
        echo "  ❌ $dataset: Missing shuffled versions"
        MISSING_DATASETS+=("$dataset")
    fi
done

if [ ${#MISSING_DATASETS[@]} -gt 0 ]; then
    echo ""
    echo "❌ Error: Missing shuffled datasets:"
    printf '  %s\n' "${MISSING_DATASETS[@]}"
    echo "   Please run the dataset shuffling script first."
    exit 1
fi

echo ""
echo "📊 Dataset summary:"
for dataset in "${REQUIRED_DATASETS[@]}"; do
    # Count shuffled versions
    count=$(ls "${SHUFFLED_DIR}/${dataset}_shuf"*.vw.gz 2>/dev/null | wc -l)
    echo "  $dataset: $count shuffled versions"
done

# Estimate job count and time
# 19 datasets × 6 algorithms × 10 shuffles = 1140 jobs
total_jobs=1140
echo ""
echo "🔢 Total jobs: $total_jobs (19 datasets × 6 algorithms × 10 shuffles)"
echo "⏱️  Estimated time: ~$((total_jobs * 5 / NUM_WORKERS / 60)) minutes"

echo ""
echo "🚀 Starting experiment automatically..."

# Clean up existing results directory completely
echo "🧹 Cleaning and preparing results directory..."

# Check if results directory exists and has content
if [ -d "$RESULTS_DIR" ]; then
    # Count existing files for user information
    file_count=$(find "$RESULTS_DIR" -type f | wc -l)
    if [ $file_count -gt 0 ]; then
        echo "  📁 Found existing results directory with $file_count files"
        echo "  🗑️  Completely removing existing results directory to ensure clean start..."
        rm -rf "$RESULTS_DIR"
        echo "  ✅ Existing results directory removed"
    else
        echo "  📁 Found empty results directory, cleaning it..."
        rm -rf "$RESULTS_DIR"
    fi
else
    echo "  📁 Results directory doesn't exist, creating fresh..."
fi

# Create fresh results directory structure
echo "  📂 Creating fresh results directory structure..."
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/cache"
mkdir -p "$RESULTS_DIR/csv_outputs"
mkdir -p "$RESULTS_DIR/vw_outputs"

echo "✅ Results directory cleaned and prepared for fresh experiment"

# Kill any existing processes
echo "🔪 Killing any existing processes..."
pkill -f "run_6_shuffled_ds_tests" 2>/dev/null || true
sleep 2

# Start workers
echo "🚀 Starting $NUM_WORKERS workers..."
for ((i=0; i<NUM_WORKERS; i++)); do
    echo "  Starting worker $i..."
    # Set environment variable for cache control
    if [ "$DISABLE_CACHE" = "true" ]; then
        echo "    Cache disabled for worker $i"
        PYTHONUNBUFFERED=1 DISABLE_CACHE=true nohup python3 -u "${PYTHON_SCRIPT}" $i ${NUM_WORKERS} \
            --results_dir="${RESULTS_DIR}" \
            --dataset_dir="${SHUFFLED_DIR}" \
            > "${RESULTS_DIR}/worker_${i}.log" 2>&1 &
    else
        PYTHONUNBUFFERED=1 nohup python3 -u "${PYTHON_SCRIPT}" $i ${NUM_WORKERS} \
            --results_dir="${RESULTS_DIR}" \
            --dataset_dir="${SHUFFLED_DIR}" \
            > "${RESULTS_DIR}/worker_${i}.log" 2>&1 &
    fi
done

echo ""
echo "✅ Experiment started!"
echo ""
echo "📊 Monitor progress:"
echo "  # Check worker logs for errors:"
echo "  grep -r 'ERROR\|Exception\|Failed' ${RESULTS_DIR}/worker_*.log"
echo "  # Monitor real-time progress:"
echo "  tail -f ${RESULTS_DIR}/worker_*.log"
echo "  # Check worker status:"
echo "  pgrep -f 'run_6_shuffled_ds_tests' | wc -l"
echo ""
echo "🔍 Check workers:"
echo "  pgrep -f 'run_6_shuffled_ds_tests' | wc -l"
echo ""
echo "📁 Results will be in: $RESULTS_DIR"
echo "📊 CSV files will be in: $RESULTS_DIR/csv_outputs"
if [ "$DISABLE_CACHE" = "true" ]; then
    echo "⚠️  CACHE DISABLED: All experiments will run fresh (slower but guaranteed unique)"
fi
echo ""
echo "🛑 To stop the experiment:"
echo "  pkill -f 'run_6_shuffled_ds_tests'"
echo ""
echo "📋 Worker log files:"
echo "  ${RESULTS_DIR}/worker_*.log"
echo ""
echo "Usage: $0 [NUM_WORKERS] [EXPERIMENT_NAME] [DISABLE_CACHE]"
echo "Example: $0 4 test_clean_run true  # Run with 4 workers, cache disabled" 