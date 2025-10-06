# OPOCMAB Experiments 

This directory contains the complete codebase and scripts needed to reproduce the OPOCMAB (Optimistic Policy Optimization for Contextual Multi-Armed Bandits) experimental results.

## Overview

This code includes:
- **VowpalWabbit** with OPOCMAB implementation
- **Dataset downloading and preprocessing scripts**
- **Experiment execution scripts** 
- **Analysis and plotting tools**

The experiments evaluate OPOCMAB against other contextual bandit algorithms (SquareCB, FastCB, RegCB, AdaCB, and Supervised learning) on 18 carefully selected OpenML datasets.

## Requirements

- **C++ compiler** with C++11 support (GCC 4.7+ or Clang 3.4+)
- **CMake** (≥3.10) for building VowpalWabbit  
- **Boost libraries** (≥1.58) - **Note**: We build Boost 1.83.0 locally to ensure C++11 compatibility and avoid system-specific paths
- **Python 3.6+** with packages: `openml`, `numpy`, `scipy`, `matplotlib`, `pandas`

**Why local Boost build?**
- Ensures C++11 compatibility (newer system Boost may require C++14+)
- Avoids username-specific paths in the build
- Provides exact control over Boost version and configuration

### Python Package Installation

Before running the dataset scripts, install the required Python packages:

```bash
pip3 install openml numpy scipy matplotlib pandas
```

**Note**: On macOS, you may see warnings about scripts installed to directories not on PATH. These warnings can be safely ignored for this project.

**Potential Issues:**
- If you see urllib3/OpenSSL warnings, these are informational and don't affect functionality
- Ensure you have sufficient disk space (~500MB for datasets)

## Quick Start

### 1. Build VowpalWabbit

VowpalWabbit requires Boost libraries (program_options and system) that are compatible with C++11. To ensure compatibility and avoid username-specific paths, we build Boost 1.83.0 locally:

#### Step 1.1: Build Boost 1.83.0 locally

```bash
# Create directory for external dependencies
mkdir -p external_libs
cd external_libs

# Download Boost 1.83.0 (C++11 compatible)
curl -L -O https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.bz2
tar -xjf boost_1_83_0.tar.bz2
cd boost_1_83_0

# Configure Boost for required libraries only
./bootstrap.sh --with-libraries=program_options,system

# Build Boost libraries
./b2 --with-program_options --with-system
cd ../..
```

#### Step 1.2: Build VowpalWabbit

```bash
cd vowpal_wabbit
mkdir -p build
cd build

# Configure CMake with local Boost installation
cmake -DBOOST_ROOT=$(pwd)/../external_libs/boost_1_83_0 \
      -DBoost_INCLUDE_DIR=$(pwd)/../external_libs/boost_1_83_0 \
      -DBoost_LIBRARY_DIR=$(pwd)/../external_libs/boost_1_83_0/stage/lib \
      -DBUILD_TESTS=OFF ..

# Build VowpalWabbit
make -j$(nproc)

# Test the build
./vowpalwabbit/vw --version
cd ../..
```

### 2. Download and Prepare Datasets

Download the 18 specific datasets used in the experiments:

```bash
# Downloads datasets: 339, 346, 457, 476, 729, 785, 835, 848, 874, 905, 928, 964, 1006, 1012, 1015, 1062, 1073, 1084
python oml_to_vw.py 0 9999
```

Create shuffled versions for robust evaluation:

```bash
python shuffle.py
```

### Verification

Before running experiments, verify your setup is complete:

```bash
# Check VowpalWabbit binary works
./vowpal_wabbit/build/vowpalwabbit/vw --version

# Verify datasets were downloaded (should show 18 files)
ls -1 datasets/*.vw.gz | wc -l

# Verify shuffled datasets were created (should show 180 files)  
ls -1 datasets_shuffled/*.vw.gz | wc -l

# Check Python packages are available
python3 -c "import openml, numpy, scipy, matplotlib, pandas; print('All packages available')"
```

Expected output:
- VW version information
- `18` (original datasets)
- `180` (shuffled datasets: 18 × 10 shuffles)
- `All packages available`

### 3. Run Experiments

**Important**: Edit `run_6_shuffled_ds_test.sh` and set the `VW_BINARY` variable to point to your built VowpalWabbit executable:

```bash
# In run_6_shuffled_ds_test.sh, update this line:
VW_BINARY="/path/to/your/vowpal_wabbit/build/vowpalwabbit/vw"
```

Run the complete experiment suite:

```bash
./run_6_shuffled_ds_test.sh
```

**Note**: This can take several hours to complete. Monitor progress with `top` or `htop`.

**System Requirements for Experiments:**
- **RAM**: Minimum 8GB recommended (experiments are memory-intensive)
- **CPU**: Multi-core processor recommended (experiments can use parallel workers)
- **Storage**: ~2GB free space for results and intermediate files
- **Runtime**: 4-24 hours depending on system specifications

**Performance Tips:**
- Use multiple workers: `./run_6_shuffled_ds_test.sh 4` (for 4 parallel workers)
- Monitor system resources during execution
- Ensure sufficient swap space if RAM is limited

### 4. Generate Analysis and Plots

After experiments complete, generate visualizations and statistical analysis:

```bash
# Generate performance plots and comparison tables
python process_csv_plots.py \
  "res/cbresults_shuffled/csv_outputs" \
  "res/cbresults_shuffled/plots"

# Run statistical significance analysis  
python statistical_significance_analysis.py \
  "res/cbresults_shuffled/csv_outputs" \
  "res/cbresults_shuffled/statistical_analysis"
```

## Detailed Instructions

### Dataset Information

The experiments use 18 OpenML datasets selected for their diverse characteristics:

| Dataset ID | Classes | Name | 
|------------|---------|------|
| 339 | 3 | Australian |
| 346 | 2 | Ionosphere |
| 457 | 4 | Dermatology |
| 476 | 2 | Heart Disease |
| 729 | 2 | Pima Indians Diabetes |
| 785 | 2 | Chronic Kidney Disease |
| 835 | 2 | Credit Approval |
| 848 | 2 | Parkinsons |
| 874 | 2 | Mushroom |
| 905 | 2 | Mammographic Mass |
| 928 | 2 | SPECTF Heart |
| 964 | 2 | SPECT Heart |
| 1006 | 2 | Cylinder Bands |
| 1012 | 2 | MONK's Problems 2 |
| 1015 | 2 | MONK's Problems 1 |
| 1062 | 2 | Monks Problems 3 |
| 1073 | 2 | Titanic |
| 1084 | 3 | Haberman's Survival |

### Algorithm Configurations

The experiments evaluate six contextual bandit algorithms:

- **SquareCB**: Contextual bandit with squared loss
- **FastCB**: SquareCB with `--fast` flag for faster convergence  
- **OPOCMAB**: Optimistic policy optimization with confidence bounds
- **RegCB**: Regularized contextual bandit
- **AdaCB**: Adaptive contextual bandit with elimination (uses `--squarecb --elim`)
- **Supervised**: Logistic regression baseline

Each dataset uses optimized hyperparameters specific to the algorithm and dataset characteristics.

### Directory Structure After Completion

```
final/
├── external_libs/              # Locally built dependencies
│   └── boost_1_83_0/          # Boost 1.83.0 C++11 compatible
│       ├── boost/             # Boost headers
│       └── stage/lib/         # Built Boost libraries
├── vowpal_wabbit/             # VW source code and binary
│   ├── build/                 # CMake build directory
│   │   └── vowpalwabbit/vw    # Built VW executable
│   └── ext_libs/              # VW's internal dependencies
├── oml_to_vw.py              # Dataset downloader (18 specific datasets)
├── shuffle.py                # Dataset shuffling utility
├── run_6_shuffled_ds_test.sh  # Main experiment runner
├── run_6_shuffled_ds_tests.py # Core experiment logic
├── process_csv_plots.py      # Plotting and visualization
├── statistical_significance_analysis.py  # Statistical analysis
├── datasets/                 # Downloaded original datasets (after step 2)
├── datasets_shuffled/        # Shuffled datasets (10 versions each)
└── res/cbresults_shuffled/   # Experiment results
    ├── csv_outputs/          # Raw CSV results
    ├── plots/               # Generated plots
    └── statistical_analysis/  # Statistical test results
```

### Troubleshooting

**Build Issues:**
- Ensure all VowpalWabbit dependencies are installed
- Check that you have a C++11 compatible compiler
- Verify Boost libraries are properly built in `external_libs/boost_1_83_0/stage/lib/`
- If CMake can't find Boost, ensure you're using absolute paths in the configuration

**Python Package Issues:**
- Install required packages: `pip3 install openml numpy scipy matplotlib pandas`
- Warnings about PATH or urllib3/OpenSSL are informational and can be ignored
- Ensure you have Python 3.6+ installed

**Dataset Download Issues:**
- Ensure internet connectivity for OpenML access
- Check that Python packages (openml, scipy, numpy) are installed
- Verify sufficient disk space (datasets ~500MB total)
- Original datasets should be in `datasets/` directory
- Shuffled datasets should be in `datasets_shuffled/` directory (180 files total)

**Experiment Execution Issues:**
- Confirm VW_BINARY path is correct in `run_6_shuffled_ds_test.sh`
- Example: `VW_BINARY="/full/path/to/vowpal_wabbit/build/vowpalwabbit/vw"`
- Ensure shuffled datasets exist in `datasets_shuffled/` directory
- Check available system memory (experiments are memory-intensive)

### Customization


```

**Script Parameters and Advanced Usage:**
```bash
# Full script usage
./run_6_shuffled_ds_test.sh [NUM_WORKERS] [EXPERIMENT_NAME]
```

The script automatically:
- Creates output directory: `res/cbresults_${EXPERIMENT_NAME}/` (default: `res/cbresults_shuffled/`)
- Uses default data directory: `datasets_shuffled/`
- Validates all required shuffled datasets exist
- Sets up proper directory structure (`cache/`, `csv_outputs/`, `vw_outputs/`)
- Provides real-time monitoring commands

### Monitoring and Verification

**Check Experiment Status:**
```bash
# Check if experiment is still running
ps aux | grep -E "(python|vw|vowpal)" | grep -v grep

# Monitor worker progress in real-time
tail -f res/cbresults_shuffled/worker_*.log

# Check for errors
grep -r 'ERROR\|Exception\|Failed' res/cbresults_shuffled/worker_*.log
```

### Plot Generation

After experiments complete, generate performance visualizations and comparison tables:

**Generate All Plots and Tables:**

```bash
# Full script usage
python process_csv_plots.py [INPUT_CSV_DIR] [OUTPUT_PLOTS_DIR]
```

**Script Usage:**
- **Input**: CSV results directory (e.g., `res/cbresults_shuffled/csv_outputs`)
- **Output**: Plot directory where all visualizations will be saved (e.g., `res/cbresults_shuffled/plots`)
- The script automatically creates the output directory if it doesn't exist
- Processes all CSV files and generates performance plots and comparison tables

**Generated Outputs:**
```bash
# Performance plots for each dataset (PDF format)
${DATASET}_median_performance.pdf     # Median performance curves
${DATASET}_average_performance.pdf    # Average performance with std deviation

# Comparison tables and analysis
comparison_table_mean_difference_from_supervised.csv     # Raw mean differences
comparison_table_median_difference_from_supervised.csv   # Raw median differences
comparison_table_difference_from_supervised.txt         # Comprehensive text summary
mean_differences_table.pdf                             # Visual table (mean)
median_differences_table.pdf                           # Visual table (median)

# Additional files
legend.pdf                                              # Algorithm legend for all plots
```

### Statistical Significance Analysis

After experiments complete, perform statistical significance testing for pairwise algorithm comparisons:

```bash
# Full script usage
python3 statistical_significance_analysis.py [INPUT_CSV_DIR] [OUTPUT_ANALYSIS_DIR]
```

**Script Usage:**
- **Input**: CSV results directory (e.g., `res/cbresults_shuffled/csv_outputs`)
- **Output**: Analysis directory where statistical results will be saved (e.g., `res/cbresults_shuffled/statistical_analysis`)
- The script automatically creates the output directory if it doesn't exist
- Performs pairwise statistical significance testing using two-tailed tests with normal CDF
- Uses significance level α = 0.05 (configurable with `--alpha` parameter)

**Generated Outputs:**
```bash
# Summary files
statistical_significance_summary.txt                          # Comprehensive text summary
statistical_significance_wins_losses_summary.csv             # Overall wins/losses across datasets

# Comprehensive analysis tables  
statistical_significance_comprehensive_p_values.csv/.pdf     # All p-values in matrix format
statistical_significance_comprehensive_wins.csv/.pdf         # All wins/losses in matrix format
statistical_significance_detailed.csv                        # Detailed pairwise results

# Visual tables (PDF format)
statistical_significance_wins_losses_table.pdf               # Overall wins/losses visualization
statistical_significance_p_values_table_${DATASET}.pdf       # P-values table per dataset

# Per-dataset analysis (CSV format)
statistical_significance_wins_losses_${DATASET}.csv          # Wins/losses per dataset
statistical_significance_p_values_${DATASET}.csv             # P-values per dataset
```

## Citation

If you use this code in your research, please cite the paper.

---

# Vowpal Wabbit

<img src="logo_assets/vowpal-wabbits-github-logo@3x.png" height="auto" width="100%" alt="Vowpal Wabbit">

[![Linux build status](https://img.shields.io/azure-devops/build/vowpalwabbit/3934113c-9e2b-4dbc-8972-72ab9b9b4342/23?label=Linux%20build&logo=Azure%20Devops)](https://dev.azure.com/vowpalwabbit/Vowpal%20Wabbit/_build/latest?definitionId=23&branchName=master)
[![Windows build status](https://img.shields.io/azure-devops/build/vowpalwabbit/3934113c-9e2b-4dbc-8972-72ab9b9b4342/14?label=Windows%20build&logo=Azure%20Devops)](https://dev.azure.com/vowpalwabbit/Vowpal%20Wabbit/_build/latest?definitionId=14&branchName=master)
[![MacOS build status](https://img.shields.io/azure-devops/build/vowpalwabbit/3934113c-9e2b-4dbc-8972-72ab9b9b4342/22?label=MacOS%20build&logo=Azure%20Devops)](https://dev.azure.com/vowpalwabbit/Vowpal%20Wabbit/_build/latest?definitionId=22&branchName=master)

[![codecov](https://codecov.io/gh/VowpalWabbit/vowpal_wabbit/branch/master/graph/badge.svg)](https://codecov.io/gh/VowpalWabbit/vowpal_wabbit)
[![Total Alerts](https://img.shields.io/lgtm/alerts/g/JohnLangford/vowpal_wabbit.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/JohnLangford/vowpal_wabbit/alerts/)
[![Gitter chat](https://badges.gitter.im/VowpalWabbit.svg)](https://gitter.im/VowpalWabbit)

This is the *Vowpal Wabbit* fast online learning code.

## Why Vowpal Wabbit?
Vowpal Wabbit is a machine learning system which pushes the frontier of machine learning with techniques such as online, hashing, allreduce, reductions, learning2search, active, and interactive learning. There is a specific focus on reinforcement learning with several contextual bandit algorithms implemented and the online nature lending to the problem well. Vowpal Wabbit is a destination for implementing and maturing state of the art algorithms with performance in mind.

- **Input Format.** The input format for the learning algorithm is substantially more flexible than might be expected. Examples can have features consisting of free form text, which is interpreted in a bag-of-words way. There can even be multiple sets of free form text in different namespaces.
- **Speed.** The learning algorithm is fast -- similar to the few other online algorithm implementations out there. There are several optimization algorithms available with the baseline being sparse gradient descent (GD) on a loss function.
- **Scalability.** This is not the same as fast. Instead, the important characteristic here is that the memory footprint of the program is bounded independent of data. This means the training set is not loaded into main memory before learning starts. In addition, the size of the set of features is bounded independent of the amount of training data using the hashing trick.
- **Feature Interaction.** Subsets of features can be internally paired so that the algorithm is linear in the cross-product of the subsets. This is useful for ranking problems. The alternative of explicitly expanding the features before feeding them into the learning algorithm can be both computation and space intensive, depending on how it's handled.

[Visit the wiki to learn more.](https://github.com/VowpalWabbit/vowpal_wabbit/wiki)

## Getting Started
For the most up to date instructions for getting started on Windows, MacOS or Linux [please see the wiki](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Getting-started). This includes:

- [Installing with a package manager](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Getting-started)
- [Dependencies](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Dependencies)
- [Building](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Building)
- [Tutorial](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Tutorial)

## License

This project is licensed under the same terms as VowpalWabbit.

## Contact

For questions about the experiments or OPOCMAB implementation, please refer the authors.
