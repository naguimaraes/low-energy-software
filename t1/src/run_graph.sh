#!/bin/bash
# Must be run with sudo privileges

set -e
set +x

# Settings
CXX=g++
CXXFLAGS="-std=c++17"
TARGET=graph
SRC=graph.cpp
PERF_PATH=/usr/lib/linux-tools-6.8.0-85/perf #if normally installed, leave it as only 'perf'
LOGICAL_CPU=9
LOGICAL_CPU_TWIN=21 # manual setting of twin core; discover it running: lscpu -e
RESULTS_DIR="../results/graph"
DATASETS=(
    "../datasets/artist_edges.csv"
    "../datasets/crocodile_edges.csv"
)
STRUCTURES=(matrix list hash matrix_vec list_vec hash_umap)
OPERATIONS=(insertion components clustering)
REPEATS=5
METRICS_CACHE="L1-dcache-load-misses" # l1 data cache misses
METRICS_ENERGY="power_core/energy-core/" # the energy consumed only by the current core running the process

# Compile the program
printf "Compiling $SRC...\n"
if ! $CXX $CXXFLAGS -o $TARGET $SRC; then
    printf "Compilation failed.\n"
    exit 1
fi
printf "Compilation finished.\n\n"

# Prepare results directory
printf "Preparing results directory at '$RESULTS_DIR'...\n\n"
if [ -d "$RESULTS_DIR" ]; then
    rm -rf -- "$RESULTS_DIR"
fi
mkdir -p -- "$RESULTS_DIR"

# Ensure directory is owned by the user running the script (even if invoked via sudo)
owner_user="${SUDO_USER:-$(id -un)}"
owner_group="$(id -gn "$owner_user" 2>/dev/null || echo "$owner_user")"
sudo chown -R "$owner_user:$owner_group" "$RESULTS_DIR" 2>/dev/null || \
chown -R "$owner_user:$owner_group" "$RESULTS_DIR" 2>/dev/null || true

# Configure perf_event_paranoid for perf access
printf "Configuring perf permissions...\n"
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null
printf "perf_event_paranoid set to -1.\n\n"

# Disable twin core
printf "Disabling core $LOGICAL_CPU_TWIN...\n"
echo 0 | sudo tee /sys/devices/system/cpu/cpu${LOGICAL_CPU_TWIN}/online > /dev/null
printf "Core $LOGICAL_CPU_TWIN disabled.\n\n"

# Fix CPU frequency to maximum
printf "Fixing CPU frequency to maximum on core $LOGICAL_CPU...\n"
echo "performance" | sudo tee /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/scaling_governor > /dev/null
MAX_FREQ=$(cat /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/cpuinfo_max_freq)
sudo bash -c "echo $MAX_FREQ > /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/scaling_min_freq"
sudo bash -c "echo $MAX_FREQ > /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/scaling_max_freq"
printf "Fixed at maximum frequency found as: %sHz\n\n" "$MAX_FREQ"

printf "Starting experiments...\n\n"
printf "Dataset \t Structure \t Operation \t Metric \t Mean ± StdDev \n"

for dataset in "${DATASETS[@]}"; do
    dataset_name=$(basename "$dataset" | cut -d. -f1)
    for structure in "${STRUCTURES[@]}"; do
        for operation in "${OPERATIONS[@]}"; do

            # 2 runs for each combination, 1 for cache and 1 for energy + time
            # each run runs for $REPEATS times to get mean and standard deviation

            # Log dataset, structure, operation for cache metrics
            if [ "$structure" != "matrix" ]; then
                printf "%s \t %s \t\t %s \t %s \t" "$dataset_name" "$structure" "$operation" "cache"
            else
                printf "%s \t %s \t %s \t %s \t" "$dataset_name" "$structure" "$operation" "cache"
            fi

            # Flush the caches
            sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
            
            # Define output files
            out_cache="$RESULTS_DIR/${dataset_name}_${structure}_${operation}_cache.out"
            out_main="$RESULTS_DIR/${dataset_name}_${structure}_${operation}.out"
            
            # Run perf for cache metric (keep default text output with mean and +- %)
            taskset -c $LOGICAL_CPU $PERF_PATH stat -r $REPEATS -e $METRICS_CACHE ./$TARGET "$dataset" "$structure" "$operation" 2> perf_tmp.txt 1> main_tmp.txt
            
            # Store perf output and program stdout
            tail -n +2 perf_tmp.txt >> "$out_cache"
            cat main_tmp.txt >> "$out_main"
            
            # Extract mean and relative stddev (%) then compute absolute stddev = mean * (%/100)
            cache_stats=$(grep 'L1-dcache-load-misses' perf_tmp.txt | grep '+-' | tail -n1 | \
              awk '{gsub(",","",$1); mean=$1; pct=0; for(i=1;i<=NF;i++){ if($i=="+-"){p=$(i+1); gsub("%","",p); pct=p/100; break; }} \
                    printf "%.2e ± %.2e misses", mean, mean*pct }')

            # Print the table line with cache results
            printf "\t %s \n" "$cache_stats"

            # Delete temporary files
            rm -f perf_tmp.txt main_tmp.txt
            
            # Log dataset, structure, operation for energy metrics
            if [ "$structure" != "matrix" ]; then
                printf "%s \t %s \t\t %s \t %s \t" "$dataset_name" "$structure" "$operation" "energy"
            else
                printf "%s \t %s \t %s \t %s \t" "$dataset_name" "$structure" "$operation" "energy"
            fi
            
            # Flush the caches
            sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"

            # Define output files
            out_energy="$RESULTS_DIR/${dataset_name}_${structure}_${operation}_energy.out"
            out_time="$RESULTS_DIR/${dataset_name}_${structure}_${operation}_time.out"

            # Run perf for energy (and it also prints time)
            taskset -c $LOGICAL_CPU $PERF_PATH stat -r $REPEATS -e $METRICS_ENERGY ./$TARGET "$dataset" "$structure" "$operation" 2> perf_tmp.txt 1> main_tmp.txt
            
            # Save raw values for energy and time and program stdout
            grep 'Joules' perf_tmp.txt | awk '{print $1}' >> "$out_energy"
            grep 'seconds time elapsed' perf_tmp.txt | awk '{print $1}' >> "$out_time"
            cat main_tmp.txt >> "$out_main"
            
            # Energy: use relative stddev (%) from perf to compute absolute stddev
            energy_stats=$(grep 'Joules' perf_tmp.txt | grep '+-' | tail -n1 | \
              awk '{gsub(",","",$1); mean=$1; pct=0; for(i=1;i<=NF;i++){ if($i=="+-"){p=$(i+1); gsub("%","",p); pct=p/100; break; }} \
                    printf "%.4f ± %.4f J", mean, mean*pct }')
            printf " %s \n" "$energy_stats"
            
            # Log dataset, structure, operation for time metrics
            if [ "$structure" != "matrix" ]; then
                printf "%s \t %s \t\t %s \t %s \t" "$dataset_name" "$structure" "$operation" "time"
            else
                printf "%s \t %s \t %s \t %s \t" "$dataset_name" "$structure" "$operation" "time"
            fi

            # Time: use the absolute stddev right after "+-" provided by perf
            time_stats=$(grep 'seconds time elapsed' perf_tmp.txt | grep '+-' | tail -n1 | \
              awk '{mean=$1; std=0; for(i=1;i<=NF;i++){ if($i=="+-"){std=$(i+1); break; }} \
                    gsub(",","",mean); gsub(",","",std); printf "%.4f ± %.4f s", mean, std }')
            printf "\t %s \n" "$time_stats"
            
            # Delete temporary files
            rm -f perf_tmp.txt main_tmp.txt
        done
    done
done

# Re-enable twin core
printf "Re-enabling core $LOGICAL_CPU_TWIN...\n"
echo 1 | sudo tee /sys/devices/system/cpu/cpu${LOGICAL_CPU_TWIN}/online > /dev/null
printf "Core $LOGICAL_CPU_TWIN re-enabled.\n\n"

# Restore CPU frequency scaling governor to 'powersave'
printf "Restoring core $LOGICAL_CPU to its default settings...\n\n"
echo "powersave" | sudo tee /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/scaling_governor > /dev/null
MIN_FREQ=$(cat /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/cpuinfo_min_freq)
sudo bash -c "echo $MIN_FREQ > /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/scaling_min_freq"
sudo bash -c "echo $MAX_FREQ > /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/scaling_max_freq"
printf "CPU ${LOGICAL_CPU} restored successfully.\n\n"

# Restore perf_event_paranoid to original value
printf "Restoring perf_event_paranoid to 4...\n"
echo 4 | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null
printf "perf_event_paranoid restored.\n\n"

printf "Experiments finished.\n\n"

printf "Now generating the plots...\n"
python3 plot_graph.py
printf "Plots generated successfully.\n\n"

printf "Erasing the target executable...\n"
rm -f $TARGET
printf "Target executable erased successfully.\n\n"

printf "End of this script. All experiments completed successfully!\n"
