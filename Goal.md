# Performance Analysis Goals

## Overview
Generate FGC speedup analysis plots comparing FGC performance against state-of-the-art methods (FAISS, Annoy, ScaNN, HNSWLIB).

## Target Visualizations

### 1. FGC Speedup Analysis - 1M Points, K=40, Max Dimensions = 20
**Purpose**: Analyze how FGC speedup varies with dimensionality at a fixed dataset size and k-value.

**Parameters**:
- Dataset size: 1,000,000 points
- K-value: 40
- Dimensions: 1 to 20
- Algorithms: FGC vs FAISS, Annoy, ScaNN, HNSWLIB

**Expected Output**: 
- Single plot showing speedup curves for each algorithm
- X-axis: Number of dimensions (1-20)
- Y-axis: Speedup factor (FGC time / competitor time)
- Reference line at y=1 (no speedup)

### 2. D=3 Speedup Analysis - Varying Dataset Sizes
**Purpose**: Analyze how FGC speedup scales with dataset size at fixed dimension and varying k-values.

**Parameters**:
- Dimension: 3
- K-values: 10, 40, 100
- Dataset sizes: Increments of 500K (500K, 1M, 1.5M, 2M, 2.5M, 3M, 3.5M, 4M, 4.5M, 5M)
- Algorithms: FGC vs FAISS, Annoy, ScaNN, HNSWLIB

**Expected Output**:
- Subplot layout: 3 columns (one for each k-value)
- X-axis: Dataset size (500K to 5M)
- Y-axis: Speedup factor
- Reference line at y=1

### 3. D=5 Speedup Analysis - Varying Dataset Sizes
**Purpose**: Analyze how FGC speedup scales with dataset size at higher dimension and varying k-values.

**Parameters**:
- Dimension: 5
- K-values: 10, 40, 100
- Dataset sizes: Increments of 500K (500K, 1M, 1.5M, 2M, 2.5M, 3M, 3.5M, 4M, 4.5M, 5M)
- Algorithms: FGC vs FAISS, Annoy, ScaNN, HNSWLIB

**Expected Output**:
- Subplot layout: 3 columns (one for each k-value)
- X-axis: Dataset size (500K to 5M)
- Y-axis: Speedup factor
- Reference line at y=1

## Implementation Notes

### Data Requirements
- Ensure all algorithms have data for the specified parameter combinations
- Handle missing data points gracefully
- Apply data quality filters (remove outliers, failed runs)

### Plot Styling
- Professional academic publication style
- Consistent color scheme across all plots
- Clear legends and axis labels
- High DPI output for publications

### File Naming Convention
1. `fgc_speedup_1M_k40_dimensions.png`
2. `fgc_speedup_d3_varying_sizes.png`
3. `fgc_speedup_d5_varying_sizes.png`

## Success Criteria
- All three plots generated successfully
- Clear visualization of FGC performance advantages
- Professional quality suitable for academic publication
- Consistent styling and formatting across all plots

