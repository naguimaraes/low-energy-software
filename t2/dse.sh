#!/usr/bin/env bash
# Design Space Exploration (DSE) script for AlexNet variants.
#
# The script trains and evaluates three architectures defined in classes.py
# (ConvolutionalAlexNet, FFTAlexNet, TorchAlexNet) with different kernel sizes
# while sampling GPU metrics via nvidia-smi or CPU metrics via perf. 
# A consolidated summary is written to results/dse/summary.csv and raw logs 
# are stored under results/dse/raw/.
#
# USAGE MODES:
#
# 1. Kernel Size Exploration (default):
#    - KERNEL_SIZE_SWEEP=1: Test multiple first kernel sizes
#    - KERNEL_SIZE_SWEEP=0: Use fixed FIRST_KERNEL value
#    - FIRST_KERNEL_SIZES="3 5 7 9 11 13": Specify kernel sizes to test
#    - FIRST_KERNEL=11: Default kernel size when sweep disabled
#
# 2. Hardware Monitoring Selection:
#    - MONITOR_MODE="gpu": Use nvidia-smi for GPU monitoring (default)
#    - MONITOR_MODE="cpu": Use perf for CPU energy monitoring
#
# EXAMPLE: Explore specific kernel sizes
#    KERNEL_SIZE_SWEEP=1 FIRST_KERNEL_SIZES="5 7 11" bash dse.sh
#
# EXAMPLE: Single kernel size with CPU monitoring
#    KERNEL_SIZE_SWEEP=0 FIRST_KERNEL=7 MONITOR_MODE=cpu bash dse.sh
#
# EXAMPLE: Full kernel size sweep with GPU monitoring
#    KERNEL_SIZE_SWEEP=1 FIRST_KERNEL_SIZES="3 5 7 9 11 13 15" bash dse.sh

set -euo pipefail

PERF_PATH="/usr/lib/linux-tools/6.8.0-87-generic/perf"  # Adjust as needed
PYTHON_BIN=${PYTHON_BIN:-python3}
RESULTS_DIR=${RESULTS_DIR:-results/dse}
RAW_DIR="$RESULTS_DIR/raw"
SUMMARY_FILE="$RESULTS_DIR/summary.csv"
MEASURE_INTERVAL=${MEASURE_INTERVAL:-0.2}
PRE_SAMPLING_SECONDS=${PRE_SAMPLING_SECONDS:-5}
POST_SAMPLING_SECONDS=${POST_SAMPLING_SECONDS:-5}

TRAIN_EPOCHS=${TRAIN_EPOCHS:-45}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-2}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-}

INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-32}
INFER_ITERATIONS=${INFER_ITERATIONS:-25}
INFER_NUM_WORKERS=${INFER_NUM_WORKERS:-1}
INFER_EXTRA_ARGS=${INFER_EXTRA_ARGS:-}

# Kernel size sweep for DSE
# Set to 1 to enable kernel size exploration, 0 to use fixed FIRST_KERNEL
KERNEL_SIZE_SWEEP=${KERNEL_SIZE_SWEEP:-0}
# First kernel sizes to test (space-separated list)
# These determine the starting kernel size for the progression
FIRST_KERNEL_SIZES=${FIRST_KERNEL_SIZES:-"3 7"}

# Default first kernel size (used when KERNEL_SIZE_SWEEP=0)
FIRST_KERNEL=${FIRST_KERNEL:-3}

# Evaluation-only mode
# Set to 1 to skip training and only run evaluation on pre-trained models
EVAL_ONLY=${EVAL_ONLY:-1}
# Note: EVAL_ONLY mode is not compatible with kernel size exploration
PRETRAINED_MODELS=${PRETRAINED_MODELS:-"results/convolutional_3_45.pth results/fft_3_45.pth results/torch_3_45.pth"}

# Training mode selection:
# - If MATCH_BASELINE_ACCURACY=1: Train AlexNet for TRAIN_EPOCHS, then train FFT models until they match AlexNet's Top-1 accuracy
# - If MATCH_BASELINE_ACCURACY=0: Train all 3 models for the same number of epochs (TRAIN_EPOCHS)
MATCH_BASELINE_ACCURACY=${MATCH_BASELINE_ACCURACY:-0}

# For FFT models: train until they match AlexNet's Top-1 accuracy (only used when MATCH_BASELINE_ACCURACY=1)
TARGET_TOP1=""

DATA_ROOT=${DATA_ROOT:-dataset}
PROFILE_LAYERS=${PROFILE_LAYERS:-0}

MONITOR_MODE=${MONITOR_MODE:-cpu}
# CPU monitoring settings
# Hardware monitoring mode: "gpu" or "cpu"
PERF_REPETITIONS=${PERF_REPETITIONS:-1}

# Store original perf_event_paranoid value for restoration
ORIGINAL_PERF_PARANOID=""

if [[ "$MONITOR_MODE" == "gpu" ]]; then
    command -v nvidia-smi >/dev/null || {
        echo "Error: nvidia-smi not found in PATH." >&2
        exit 1
    }
elif [[ "$MONITOR_MODE" == "cpu" ]]; then
    command -v $PERF_PATH >/dev/null || {
        echo "Error: perf not found in PATH." >&2
        echo "Install with: sudo apt-get install linux-tools-common linux-tools-generic" >&2
        exit 1
    }
    
    # Store original perf_event_paranoid value
    ORIGINAL_PERF_PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "4")
    
    # Check if we need to adjust perf_event_paranoid
    if [[ "$ORIGINAL_PERF_PARANOID" -gt 2 ]]; then
        echo ">>> Current perf_event_paranoid = $ORIGINAL_PERF_PARANOID (too restrictive)"
        echo ">>> Attempting to set perf_event_paranoid = -1 for CPU monitoring..."
        
        if sudo sysctl -w kernel.perf_event_paranoid=-1 >/dev/null 2>&1; then
            echo ">>> Successfully set perf_event_paranoid = -1"
            echo ">>> Will restore to $ORIGINAL_PERF_PARANOID on exit"
        else
            echo "Error: Failed to set perf_event_paranoid. This script needs sudo privileges in CPU mode." >&2
            echo "" >&2
            echo "Please run with sudo:" >&2
            echo "  sudo bash dse.sh" >&2
            echo "" >&2
            echo "Or manually set permissions before running:" >&2
            echo "  sudo sysctl -w kernel.perf_event_paranoid=-1" >&2
            echo "" >&2
            exit 1
        fi
    else
        echo ">>> perf_event_paranoid = $ORIGINAL_PERF_PARANOID (OK for CPU monitoring)"
    fi
    
    # Verify that power/energy-pkg/ is available
    if ! $PERF_PATH list 2>/dev/null | grep "power/energy-pkg/"; then
        echo "Error: power/energy-pkg/ counter not available." >&2
        echo "" >&2
        echo "Possible causes:" >&2
        echo "  - CPU doesn't support RAPL (Running Average Power Limit)" >&2
        echo "  - RAPL module not loaded" >&2
        echo "" >&2
        echo "Try loading the module:" >&2
        echo "  sudo modprobe intel_rapl_msr" >&2
        echo "" >&2
        exit 1
    fi
else
    echo "Error: MONITOR_MODE must be 'gpu' or 'cpu'." >&2
    exit 1
fi

command -v python3 >/dev/null || {
    echo "Error: python3 not found in PATH." >&2
    exit 1
}

# Prepare kernel size list for exploration
if [[ "$KERNEL_SIZE_SWEEP" == "1" ]]; then
    read -r -a KERNEL_SIZE_LIST <<< "$FIRST_KERNEL_SIZES"
else
    KERNEL_SIZE_LIST=("$FIRST_KERNEL")
fi
if [[ ${#KERNEL_SIZE_LIST[@]} -eq 0 ]]; then
    echo "Error: KERNEL_SIZE_LIST is empty." >&2
    exit 1
fi

TRAIN_EXTRA_ARGS_ARRAY=()
if [[ -n "$TRAIN_EXTRA_ARGS" ]]; then
    read -r -a TRAIN_EXTRA_ARGS_ARRAY <<< "$TRAIN_EXTRA_ARGS"
fi

INFER_EXTRA_ARGS_ARRAY=()
if [[ -n "$INFER_EXTRA_ARGS" ]]; then
    read -r -a INFER_EXTRA_ARGS_ARRAY <<< "$INFER_EXTRA_ARGS"
fi

PROFILE_ARGS=()
if [[ "$PROFILE_LAYERS" == "1" ]]; then
    PROFILE_ARGS+=(--profile-layers)
fi

ARCHES=("convolutional" "fft" "torch")
ALEXNET_MODEL_PATH=""

mkdir -p "$RAW_DIR"
rm -f "$SUMMARY_FILE"

if [[ "$MONITOR_MODE" == "cpu" ]]; then
    printf "phase,arch,label,num_workers,command,total_runtime_s,runtime_stddev_s,energy_J,energy_stddev_J,metrics_log,command_log,status\n" > "$SUMMARY_FILE"
else
    printf "phase,arch,label,num_workers,command,total_runtime_s,avg_power_W,max_power_W,avg_util_pct,max_util_pct,avg_mem_used_MiB,max_mem_used_MiB,avg_temp_C,max_temp_C,metrics_log,command_log,status\n" > "$SUMMARY_FILE"
fi

declare -a MONITOR_PIDS=()
declare -a CMD_PIDS=()

cleanup() {
    for pid in "${MONITOR_PIDS[@]:-}"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${CMD_PIDS[@]:-}"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    
    # Restore original perf_event_paranoid if in CPU mode
    if [[ "$MONITOR_MODE" == "cpu" ]] && [[ -n "$ORIGINAL_PERF_PARANOID" ]] && [[ "$ORIGINAL_PERF_PARANOID" != "-1" ]]; then
        current_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "-1")
        if [[ "$current_paranoid" == "-1" ]]; then
            echo ""
            echo ">>> Restoring perf_event_paranoid to $ORIGINAL_PERF_PARANOID..."
            if sudo sysctl -w kernel.perf_event_paranoid="$ORIGINAL_PERF_PARANOID" >/dev/null 2>&1; then
                echo ">>> Successfully restored perf_event_paranoid = $ORIGINAL_PERF_PARANOID"
            else
                echo ">>> Warning: Failed to restore perf_event_paranoid (may require manual restoration)" >&2
            fi
        fi
    fi
}
trap cleanup EXIT INT TERM

monitor_gpu() {
    local target_pid="$1"
    local outfile="$2"
    local base_ms_opt="${3:-}"
    local base_ms=""
    if [[ -n "$base_ms_opt" ]]; then
        base_ms=$base_ms_opt
    fi
    while true; do
        if ! kill -0 "$target_pid" 2>/dev/null; then
            break
        fi

        # milliseconds since epoch (no decimal point)
        local ts_ms
        ts_ms=$(date +%s%3N)
        if [[ -z "$base_ms" ]]; then
            base_ms=$ts_ms
        fi
        local rel_ms=$((ts_ms - base_ms))

        local metrics
        metrics=$(nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null) || {
            sleep "$MEASURE_INTERVAL"
            continue
        }

        while IFS=',' read -r power util mem_used mem_total temp; do
            [[ -z "$power" ]] && continue
            printf "%d,%s,%s,%s,%s,%s\n" "$rel_ms" "$power" "$util" "$mem_used" "$mem_total" "$temp" >> "$outfile"
        done <<< "$metrics"

        sleep "$MEASURE_INTERVAL"
    done
}

sample_window() {
    local outfile="$1"
    local duration_sec="$2"
    local base_ms="$3"

    if [[ -z "$duration_sec" ]]; then
        return
    fi

    local duration_ms
    duration_ms=$(awk -v d="$duration_sec" 'BEGIN {
        if (d == "" ) {
            print 0;
            exit
        }
        if (d + 0 <= 0) {
            print 0;
            exit
        }
        printf "%d", d * 1000
    }') || duration_ms=0

    if [[ -z "$duration_ms" ]] || (( duration_ms <= 0 )); then
        return
    fi

    local start_ms
    start_ms=$(date +%s%3N)
    while true; do
        local ts_ms
        ts_ms=$(date +%s%3N)
        local rel_ms=$((ts_ms - base_ms))

        local metrics
        metrics=$(nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null) || {
            sleep "$MEASURE_INTERVAL"
            continue
        }

        while IFS=',' read -r power util mem_used mem_total temp; do
            [[ -z "$power" ]] && continue
            printf "%d,%s,%s,%s,%s,%s\n" "$rel_ms" "$power" "$util" "$mem_used" "$mem_total" "$temp" >> "$outfile"
        done <<< "$metrics"

        if (( ts_ms - start_ms >= duration_ms )); then
            break
        fi
        sleep "$MEASURE_INTERVAL"
    done
}

run_with_metrics_cpu() {
    local phase="$1"
    local arch="$2"
    local label="$3"
    local num_workers="$4"
    shift 4
    local -a cmd=("$@")

    local metrics_base="${label}_metrics"
    local log_base="${label}_${MONITOR_MODE}"
    local metrics_file="$RAW_DIR/${metrics_base}.csv"
    local log_file="$RAW_DIR/${log_base}.log"
    local perf_file="$RAW_DIR/${label}_perf.txt"

    local cmd_str
    printf -v cmd_str '%q ' "${cmd[@]}"
    cmd_str=${cmd_str%% }

    echo ">>> [$phase][$arch][$label] Starting: ${cmd_str}"
    echo ">>> Running ${PERF_REPETITIONS} repetitions with perf..."

    local status="OK"

    # Run command with perf stat
    $PERF_PATH stat -r "$PERF_REPETITIONS" -e power/energy-pkg/ "${cmd[@]}" > "$log_file" 2> "$perf_file"
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        status="FAIL($exit_code)"
        echo "!!! [$phase][$arch][$label] Command failed with exit code $exit_code. See $log_file" >&2
    fi

    # Parse perf output to extract energy and time statistics
    local energy_mean=0 energy_stddev=0 time_mean=0 time_stddev=0
    
    # Extract energy (Joules) - format: "X.XX Joules power/energy-pkg/ ( +- Y.YY% )"
    if grep -q "Joules power/energy-pkg/" "$perf_file"; then
        energy_mean=$(grep "Joules power/energy-pkg/" "$perf_file" | awk '{print $1}' | tr -d ',' || echo "0")
        # Extract stddev percentage and convert to absolute value
        local energy_stddev_pct=$(grep "Joules power/energy-pkg/" "$perf_file" | grep -oP '\+\-\s*\K[0-9.]+' || echo "0")
        energy_stddev=$(python3 -c "print(float('${energy_mean}') * float('${energy_stddev_pct}') / 100.0)" 2>/dev/null || echo "0")
    fi
    
    # Extract time (seconds) - format: "X.XXXXXX +- Y.YYYY seconds time elapsed ( +- Z.ZZ% )"
    if grep -q "seconds time elapsed" "$perf_file"; then
        # Extract mean (first number before +-)
        time_mean=$(grep "seconds time elapsed" "$perf_file" | awk '{print $1}' | tr -d ',' || echo "0")
        # Extract stddev absolute value (number after +- and before "seconds")
        time_stddev=$(grep "seconds time elapsed" "$perf_file" | grep -oP '\+\-\s*\K[0-9.]+' | head -1 || echo "0")
    fi

    # Write metrics to CSV
    printf "repetitions,energy_J,energy_stddev_J,time_s,time_stddev_s\n" > "$metrics_file"
    printf "%d,%s,%s,%s,%s\n" "$PERF_REPETITIONS" "$energy_mean" "$energy_stddev" "$time_mean" "$time_stddev" >> "$metrics_file"

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$phase" "$arch" "$label" "$num_workers" "$cmd_str" "$time_mean" "$time_stddev" \
        "$energy_mean" "$energy_stddev" "$metrics_file" "$log_file" "$status" >> "$SUMMARY_FILE"

    echo ">>> [$phase][$arch][$label] Done. Runtime ${time_mean}s (±${time_stddev}s), Energy ${energy_mean}J (±${energy_stddev}J). Logs: $log_file"

    return $exit_code
}

run_with_metrics_gpu() {
    local phase="$1"
    local arch="$2"
    local label="$3"
    local num_workers="$4"
    shift 4
    local -a cmd=("$@")

    local metrics_base="${label}_metrics"
    local log_base="${label}_${MONITOR_MODE}"
    local metrics_file="$RAW_DIR/${metrics_base}.csv"
    local log_file="$RAW_DIR/${log_base}.log"

    printf "rel_ms,power_W,util_pct,mem_used_MiB,mem_total_MiB,temp_C\n" > "$metrics_file"

    local cmd_str
    printf -v cmd_str '%q ' "${cmd[@]}"
    cmd_str=${cmd_str%% }

    echo ">>> [$phase][$arch][$label] Starting: ${cmd_str}"

    local start_time end_time runtime status="OK"

    # establish a common base millisecond timestamp for relative times
    local base_ms
    base_ms=$(date +%s%3N)

    # pre-sampling: capture metrics for PRE_SAMPLING_SECONDS before starting the command
    sample_window "$metrics_file" "$PRE_SAMPLING_SECONDS" "$base_ms"

    # now start the command and monitoring (using same base_ms so rel_ms is continuous)
    start_time=$(date +%s.%N)
    "${cmd[@]}" &>"$log_file" &
    local cmd_pid=$!
    CMD_PIDS+=("$cmd_pid")

    monitor_gpu "$cmd_pid" "$metrics_file" "$base_ms" &
    local monitor_pid=$!
    MONITOR_PIDS+=("$monitor_pid")

    wait "$cmd_pid"
    local exit_code=$?
    end_time=$(date +%s.%N)

    if kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
    fi
    wait "$monitor_pid" 2>/dev/null || true
    for idx in "${!MONITOR_PIDS[@]}"; do
        if [[ "${MONITOR_PIDS[$idx]}" == "$monitor_pid" ]]; then
            unset 'MONITOR_PIDS[idx]'
            break
        fi
    done
    for idx in "${!CMD_PIDS[@]}"; do
        if [[ "${CMD_PIDS[$idx]}" == "$cmd_pid" ]]; then
            unset 'CMD_PIDS[idx]'
            break
        fi
    done

    # post-sampling: capture metrics for POST_SAMPLING_SECONDS after the command
    sample_window "$metrics_file" "$POST_SAMPLING_SECONDS" "$base_ms"

    runtime=$(python3 - <<PY
start = float("$start_time")
end = float("$end_time")
print(f"{end - start:.4f}")
PY
)

    local stats="0,0,0,0,0,0,0,0"
    if [[ -s "$metrics_file" && $(wc -l < "$metrics_file") -gt 1 ]]; then
        stats=$(awk -F',' 'NR>1 {
            count++;
            power_sum += $2;
            util_sum += $3;
            mem_sum += $4;
            temp_sum += $6;
            if ($2 > power_max) power_max = $2;
            if ($3 > util_max) util_max = $3;
            if ($4 > mem_max) mem_max = $4;
            if ($6 > temp_max) temp_max = $6;
        }
        END {
            if (count > 0) {
                printf "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f", power_sum / count, power_max, util_sum / count, util_max, mem_sum / count, mem_max, temp_sum / count, temp_max;
            } else {
                printf "0,0,0,0,0,0,0,0";
            }
        }' "$metrics_file")
    fi

    if [[ $exit_code -ne 0 ]]; then
        status="FAIL($exit_code)"
        echo "!!! [$phase][$arch][$label] Command failed with exit code $exit_code. See $log_file" >&2
    fi

    local avg_power=0 max_power=0 avg_util=0 max_util=0 avg_mem=0 max_mem=0 avg_temp=0 max_temp=0
    IFS=',' read -r avg_power max_power avg_util max_util avg_mem max_mem avg_temp max_temp <<< "$stats"

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$phase" "$arch" "$label" "$num_workers" "$cmd_str" "$runtime" \
        "$avg_power" "$max_power" "$avg_util" "$max_util" "$avg_mem" "$max_mem" "$avg_temp" "$max_temp" \
        "$metrics_file" "$log_file" "$status" >> "$SUMMARY_FILE"

    echo ">>> [$phase][$arch][$label] Done. Runtime ${runtime}s. Logs: $log_file"

    return $exit_code
}

run_with_metrics() {
    if [[ "$MONITOR_MODE" == "cpu" ]]; then
        run_with_metrics_cpu "$@"
    else
        run_with_metrics_gpu "$@"
    fi
}

overall_status=0

# Helper function to run inference evaluation
run_inference_eval() {
    local arch="$1"
    local model_path="$2"
    local kernel_size="$3"
    
    echo "=== Evaluating ${arch} with first_kernel=${kernel_size} ==="
    
    eval_cmd=("$PYTHON_BIN" inference.py --model "$model_path" --arch "$arch" --iterations "$INFER_ITERATIONS" --batch-size "$INFER_BATCH_SIZE" --data-root "$DATA_ROOT" --num-workers "$INFER_NUM_WORKERS")
    eval_cmd+=("${INFER_EXTRA_ARGS_ARRAY[@]}")
    eval_cmd+=("${PROFILE_ARGS[@]}")

    inference_label="inference_${arch}_k${kernel_size}_bs${INFER_BATCH_SIZE}_w${INFER_NUM_WORKERS}_i${INFER_ITERATIONS}"
    if ! run_with_metrics "inference" "$arch" "$inference_label" "$INFER_NUM_WORKERS" "${eval_cmd[@]}"; then
        overall_status=1
    fi
}

if [[ "$EVAL_ONLY" == "1" ]]; then
    # EVALUATION-ONLY MODE: Skip training, evaluate pre-trained models
    echo "=== MODE: Evaluation Only ==="
    echo "=== Evaluating pre-trained models: ${PRETRAINED_MODELS} ==="
    
    # Parse and evaluate each pre-trained model
    for model_path in $PRETRAINED_MODELS; do
        if [[ ! -f "$model_path" ]]; then
            echo "Warning: Pre-trained model not found: $model_path. Skipping." >&2
            overall_status=1
            continue
        fi
        
        # Extract arch and kernel_size from filename (format: results/<arch>_<kernel>_<epochs>.pth)
        model_name=$(basename "$model_path" .pth)
        IFS='_' read -r arch kernel_size epochs <<< "$model_name"
        
        if [[ -z "$arch" || -z "$kernel_size" || -z "$epochs" ]]; then
            echo "Warning: Could not parse arch/kernel from $model_path. Expected format: <arch>_<kernel>_<epochs>.pth. Skipping." >&2
            overall_status=1
            continue
        fi
        
        # Evaluate the model
        run_inference_eval "$arch" "$model_path" "$kernel_size"
    done
    
elif [[ "$MATCH_BASELINE_ACCURACY" == "1" ]]; then
    echo "=== MODE: Match Baseline Accuracy ==="
    echo "Error: MATCH_BASELINE_ACCURACY mode not supported in kernel size sweep." >&2
    echo "Please use default mode (MATCH_BASELINE_ACCURACY=0) for kernel size exploration." >&2
    exit 1

else
    # MODE: Kernel Size Exploration
    echo "=== MODE: Kernel Size Exploration ==="
    if [[ "$KERNEL_SIZE_SWEEP" == "1" ]]; then
        echo "=== Exploring kernel sizes: ${KERNEL_SIZE_LIST[*]} ==="
    else
        echo "=== Using fixed kernel size: ${FIRST_KERNEL} ==="
    fi
    echo "=== Training all architectures for ${TRAIN_EPOCHS} epochs ==="
    
    for kernel_size in "${KERNEL_SIZE_LIST[@]}"; do
        echo ""
        echo "========================================"
        echo "=== Kernel Size: ${kernel_size} ==="
        echo "========================================"
        
        for arch in "${ARCHES[@]}"; do
            echo ""
            echo "=== Training ${arch} with first_kernel=${kernel_size} ==="
            train_cmd=("$PYTHON_BIN" train.py --models "$arch" --first-kernel "$kernel_size" --epochs "$TRAIN_EPOCHS" --batch-size "$TRAIN_BATCH_SIZE" --num-workers "$TRAIN_NUM_WORKERS")
            train_cmd+=("${TRAIN_EXTRA_ARGS_ARRAY[@]}")
            train_cmd+=("${PROFILE_ARGS[@]}")

            train_label="train_${arch}_k${kernel_size}_e${TRAIN_EPOCHS}_bs${TRAIN_BATCH_SIZE}_w${TRAIN_NUM_WORKERS}"
            if ! run_with_metrics "train" "$arch" "$train_label" "$TRAIN_NUM_WORKERS" "${train_cmd[@]}"; then
                overall_status=1
                echo "Warning: ${arch} training with kernel_size=${kernel_size} failed. Continuing with other configurations." >&2
                continue
            fi
            
            # Find the model file (new format: <topology>_<kernel_size>_<epochs>.pth)
            model_path="results/${arch}_${kernel_size}_${TRAIN_EPOCHS}.pth"
            if [[ ! -f "$model_path" ]]; then
                echo "Warning: expected checkpoint not found for $arch at $model_path. Skipping evaluation." >&2
                overall_status=1
                continue
            fi

            # Evaluate the model
            run_inference_eval "$arch" "$model_path" "$kernel_size"
        done
    done
fi

echo "DSE complete. Summary available at $SUMMARY_FILE"
exit $overall_status
