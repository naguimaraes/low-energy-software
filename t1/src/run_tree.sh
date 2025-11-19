#!/bin/bash
# Must be run with sudo privileges

set -e
set +x

# Settings
CXX=g++
CXXFLAGS="-std=c++17"
TARGET=tree
SRC=tree.cpp
PERF_PATH=/usr/lib/linux-tools-6.8.0-85/perf # if normally installed, leave it as only 'perf'
LOGICAL_CPU=9
LOGICAL_CPU_TWIN=21 # manual setting of twin core; discover it running: lscpu -e
RESULTS_DIR="../results/tree"

# Datasets
BIBLE_FILE="../datasets/bible.txt"
PLAYERS_FILE="../datasets/players.csv"

# Structures to test
STRUCTURES=(trie ifthenelse)

# Search terms per dataset
WORDS_BIBLE_EXISTING=(faith Magormissabib)
WORDS_BIBLE_NONEXISTING=(luigi carro)
WORDS_PLAYERS_EXISTING=("Lionel Andres Messi Cuccittini" "Zitong Chen")
WORDS_PLAYERS_NONEXISTING=(Zidano PelédosEUA)

# Repetitions for perf aggregation
REPEATS=5

# Metrics
METRICS_CACHE="L1-dcache-load-misses" # L1 data cache misses
METRICS_ENERGY="power_core/energy-core/" # energy consumed by the current core

# Compile the program
printf "Compiling %s...\n" "$SRC"
if ! $CXX $CXXFLAGS -o "$TARGET" "$SRC"; then
    printf "Compilation failed.\n"
    exit 1
fi
printf "Compilation finished.\n\n"

# Prepare results directory
printf "Preparing results directory at '%s'...\n\n" "$RESULTS_DIR"
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
printf "Disabling core %s...\n" "$LOGICAL_CPU_TWIN"
echo 0 | sudo tee /sys/devices/system/cpu/cpu${LOGICAL_CPU_TWIN}/online > /dev/null
printf "Core %s disabled.\n\n" "$LOGICAL_CPU_TWIN"

# Fix CPU frequency to maximum
printf "Fixing CPU frequency to maximum on core %s...\n" "$LOGICAL_CPU"
echo "performance" | sudo tee /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/scaling_governor > /dev/null
MAX_FREQ=$(cat /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/cpuinfo_max_freq)
MIN_FREQ=$(cat /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/cpuinfo_min_freq)
sudo bash -c "echo $MAX_FREQ > /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/scaling_min_freq"
sudo bash -c "echo $MAX_FREQ > /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/scaling_max_freq"
printf "Fixed at maximum frequency found as: %sHz\n\n" "$MAX_FREQ"

printf "Starting experiments (interleaving 1000 existing + 1000 nonexisting queries per run)...\n\n"
printf "Dataset\tStructure\tRun Type\tMetric\tMean ± StdDev\n"

# Helper: repeat-and-trim an array to N items
repeat_to_n() {
  local n=$1; shift
  local -n src=$1; shift || true
  local out=()
  local len=${#src[@]}
  if (( len == 0 )); then echo ""; return; fi
  for ((i=0; i<n; i++)); do
    out+=("${src[$((i%len))]}")
  done
  printf '%s\n' "${out[@]}"
}

# Helper: build interleaved query array of size 40 from two arrays (existing/nonexisting)
build_interleaved_queries() {
  local -n arrA=$1
  local -n arrB=$2
  local A=( $(repeat_to_n 1000 arrA) )
  local B=( $(repeat_to_n 1000 arrB) )
  local Q=()
  for ((i=0;i<1000;i++)); do
    Q+=("${A[$i]}")
    Q+=("${B[$i]}")
  done
  printf '%s\n' "${Q[@]}"
}

for structure in "${STRUCTURES[@]}"; do
  # Bible dataset (40 queries: mixed existing/nonexisting)
  dataset="bible"
  dataset_file="$BIBLE_FILE"
  mapfile -t QUERIES < <(build_interleaved_queries WORDS_BIBLE_EXISTING WORDS_BIBLE_NONEXISTING)
  stem="${dataset}_${structure}_mixed_mixed"

  # Cache
  printf "%s\t%s\t%s\t%s\t" "$dataset" "$structure" "2000-queries" "cache"
  sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
  out_cache="$RESULTS_DIR/${stem}_cache.out"
  out_main="$RESULTS_DIR/${stem}.out"
  taskset -c $LOGICAL_CPU $PERF_PATH stat -r $REPEATS -e $METRICS_CACHE ./$TARGET "$dataset_file" "$dataset" "$structure" "${QUERIES[@]}" 2> perf_tmp.txt 1> main_tmp.txt
  tail -n +2 perf_tmp.txt >> "$out_cache"
  cat main_tmp.txt >> "$out_main"
  cache_stats=$(grep 'L1-dcache-load-misses' perf_tmp.txt | grep '+-' | tail -n1 | \
    awk '{gsub(",","",$1); mean=$1; pct=0; for(i=1;i<=NF;i++){ if($i=="+-"){p=$(i+1); gsub("%","",p); pct=p/100; break; }} \
          printf "%.2e ± %.2e misses", mean, mean*pct }')
  printf "%s\n" "$cache_stats"
  rm -f perf_tmp.txt main_tmp.txt

  # Energy + Time
  printf "%s\t%s\t%s\t%s\t" "$dataset" "$structure" "2000-queries" "energy"
  sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
  out_energy="$RESULTS_DIR/${stem}_energy.out"
  out_time="$RESULTS_DIR/${stem}_time.out"
  taskset -c $LOGICAL_CPU $PERF_PATH stat -r $REPEATS -e $METRICS_ENERGY ./$TARGET "$dataset_file" "$dataset" "$structure" "${QUERIES[@]}" 2> perf_tmp.txt 1> main_tmp.txt
  grep 'Joules' perf_tmp.txt | awk '{print $1}' >> "$out_energy"
  grep 'seconds time elapsed' perf_tmp.txt | awk '{print $1}' >> "$out_time"
  cat main_tmp.txt >> "$out_main"
  energy_stats=$(grep 'Joules' perf_tmp.txt | grep '+-' | tail -n1 | \
    awk '{gsub(",","",$1); mean=$1; pct=0; for(i=1;i<=NF;i++){ if($i=="+-"){p=$(i+1); gsub("%","",p); pct=p/100; break; }} \
          printf "%.4f ± %.4f J", mean, mean*pct }')
  printf "%s\n" "$energy_stats"
  time_stats=$(grep 'seconds time elapsed' perf_tmp.txt | grep '+-' | tail -n1 | \
    awk '{mean=$1; std=0; for(i=1;i<=NF;i++){ if($i=="+-"){std=$(i+1); break; }} \
          gsub(",","",mean); gsub(",","",std); printf "time %.4f ± %.4f s", mean, std }')
  printf "%s\n" "$time_stats"
  rm -f perf_tmp.txt main_tmp.txt

  # Players dataset (40 queries: mixed existing/nonexisting)
  dataset="players"
  dataset_file="$PLAYERS_FILE"
  mapfile -t QUERIES < <(build_interleaved_queries WORDS_PLAYERS_EXISTING WORDS_PLAYERS_NONEXISTING)
  stem="${dataset}_${structure}_mixed_mixed"

  # Cache
  printf "%s\t%s\t%s\t%s\t" "$dataset" "$structure" "2000-queries" "cache"
  sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
  out_cache="$RESULTS_DIR/${stem}_cache.out"
  out_main="$RESULTS_DIR/${stem}.out"
  taskset -c $LOGICAL_CPU $PERF_PATH stat -r $REPEATS -e $METRICS_CACHE ./$TARGET "$dataset_file" "$dataset" "$structure" "${QUERIES[@]}" 2> perf_tmp.txt 1> main_tmp.txt
  tail -n +2 perf_tmp.txt >> "$out_cache"
  cat main_tmp.txt >> "$out_main"
  cache_stats=$(grep 'L1-dcache-load-misses' perf_tmp.txt | grep '+-' | tail -n1 | \
    awk '{gsub(",","",$1); mean=$1; pct=0; for(i=1;i<=NF;i++){ if($i=="+-"){p=$(i+1); gsub("%","",p); pct=p/100; break; }} \
          printf "%.2e ± %.2e misses", mean, mean*pct }')
  printf "%s\n" "$cache_stats"
  rm -f perf_tmp.txt main_tmp.txt

  # Energy + Time
  printf "%s\t%s\t%s\t%s\t" "$dataset" "$structure" "2000-queries" "energy"
  sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
  out_energy="$RESULTS_DIR/${stem}_energy.out"
  out_time="$RESULTS_DIR/${stem}_time.out"
  taskset -c $LOGICAL_CPU $PERF_PATH stat -r $REPEATS -e $METRICS_ENERGY ./$TARGET "$dataset_file" "$dataset" "$structure" "${QUERIES[@]}" 2> perf_tmp.txt 1> main_tmp.txt
  grep 'Joules' perf_tmp.txt | awk '{print $1}' >> "$out_energy"
  grep 'seconds time elapsed' perf_tmp.txt | awk '{print $1}' >> "$out_time"
  cat main_tmp.txt >> "$out_main"
  energy_stats=$(grep 'Joules' perf_tmp.txt | grep '+-' | tail -n1 | \
    awk '{gsub(",","",$1); mean=$1; pct=0; for(i=1;i<=NF;i++){ if($i=="+-"){p=$(i+1); gsub("%","",p); pct=p/100; break; }} \
          printf "%.4f ± %.4f J", mean, mean*pct }')
  printf "%s\n" "$energy_stats"
  time_stats=$(grep 'seconds time elapsed' perf_tmp.txt | grep '+-' | tail -n1 | \
    awk '{mean=$1; std=0; for(i=1;i<=NF;i++){ if($i=="+-"){std=$(i+1); break; }} \
          gsub(",","",mean); gsub(",","",std); printf "time %.4f ± %.4f s", mean, std }')
  printf "%s\n" "$time_stats"
  rm -f perf_tmp.txt main_tmp.txt
done

# Re-enable twin core
printf "Re-enabling core %s...\n" "$LOGICAL_CPU_TWIN"
echo 1 | sudo tee /sys/devices/system/cpu/cpu${LOGICAL_CPU_TWIN}/online > /dev/null
printf "Core %s re-enabled.\n\n" "$LOGICAL_CPU_TWIN"

# Restore CPU frequency scaling governor to 'powersave'
printf "Restoring core %s to its default settings...\n\n" "$LOGICAL_CPU"
echo "powersave" | sudo tee /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/scaling_governor > /dev/null
sudo bash -c "echo $MIN_FREQ > /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/scaling_min_freq"
sudo bash -c "echo $MAX_FREQ > /sys/devices/system/cpu/cpu${LOGICAL_CPU}/cpufreq/scaling_max_freq"
printf "CPU %s restored successfully.\n\n" "$LOGICAL_CPU"

# Restore perf_event_paranoid to original value
printf "Restoring perf_event_paranoid to 4...\n"
echo 4 | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null
printf "perf_event_paranoid restored.\n\n"

printf "Experiments finished.\n\n"

printf "Now generating the plots...\n"
python3 plot_tree.py
printf "Plots generated successfully.\n\n"

printf "Erasing the target executable...\n"
rm -f "$TARGET"
printf "Target executable erased successfully.\n\n"

printf "End of this script. All experiments completed successfully!\n"
