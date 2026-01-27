# Journal-Quality Visualization for RAG System Analysis

## Overview

This Python script creates publication-ready graphs for comparing baseline vs reranker performance across different LLM configurations (Jina, OpenAI, Cohere). The visualizations are designed for journal articles with high-quality output (300 DPI).

## Key Improvements Over Original Code

### 1. **Modular Architecture**
- Separated concerns into distinct functions for each visualization type
- Easy to maintain and extend
- Clear function documentation

### 2. **Publication-Quality Output**
- All figures saved at 300 DPI for print quality
- Consistent styling across all visualizations
- Professional color palette optimized for academic publications
- Proper font sizes and line weights for readability

### 3. **Comprehensive Visualizations**

#### Figure 1: Grouped Bar Comparison
- Side-by-side comparison of all metrics (ROUGE-1, ROUGE-2, ROUGE-L, BERTScore)
- Clear baseline vs rerank comparison across all three configurations
- Value labels on bars for precise reading

#### Figure 2: Improvement Heatmap
- Visual representation of percentage improvements
- Color-coded to show magnitude of improvement
- Easy to identify best-performing configurations

#### Figure 3: Radar Charts
- Multi-dimensional comparison for each configuration
- Shows overall performance profile
- Filled areas for visual impact

#### Figure 4: Latency Comparison
- Bar chart showing baseline vs rerank latency
- Improvement annotations showing percentage reduction
- Helps assess trade-offs between performance and speed

#### Figure 5: Comprehensive Summary
- Multi-panel dashboard showing:
  - Overall performance average
  - Ranking by performance
  - Metric-specific improvements
  - Latency vs performance tradeoff
  - Statistical summary table

### 4. **Error Handling**
- Graceful handling of missing CSV files
- Fallback to placeholder values when data unavailable
- Clear warning messages

### 5. **Detailed Reporting**
- Generates text summary report with key findings
- Includes statistical analysis
- Easy to copy into manuscript

### 6. **Data Organization**
- Structured data loading from multiple sources
- Flexible configuration for different data formats
- Easy to add new configurations

## Usage

### Basic Usage

```python
python journal_viz.py
```

### Requirements

```bash
pip install pandas matplotlib seaborn numpy
```

### Input Data

The script expects:
1. Config 1 (Jina): Hardcoded values (already included)
2. Config 2 (OpenAI/Alibaba): CSV file `wandb_export_2026-01-25T16_21_02.190+03_00.csv`
3. Config 3 (Cohere): CSV file `wandb_export_2026-01-25T16_33_27.574+03_00.csv`

If CSV files are not found, the script uses placeholder values.

### Output Files

The script generates:
- `figure1_grouped_comparison.png` - Grouped bar chart
- `figure2_improvement_heatmap.png` - Improvement heatmap
- `figure3_radar_chart.png` - Radar charts
- `figure4_latency_comparison.png` - Latency comparison
- `figure5_comprehensive_summary.png` - Comprehensive summary
- `analysis_report.txt` - Text summary report

## Customization

### Adding New Configurations

1. Add data loading logic in `load_config_data()` function
2. Update the `configs` list in `create_comparison_dataframe()`
3. Update the `config_names` global variable

### Modifying Visualizations

Each visualization type has its own function:
- `plot_grouped_bar_comparison()` - Modify bar charts
- `plot_improvement_heatmap()` - Modify heatmap
- `plot_radar_chart()` - Modify radar charts
- `plot_latency_comparison()` - Modify latency chart
- `plot_comprehensive_summary()` - Modify summary dashboard

### Changing Colors

Modify the `COLORS` dictionary at the top of the script:

```python
COLORS = {
    'baseline': '#6c757d',
    'r erank': '#007bff',
    'rouge1': '#e74c3c',
    'rouge2': '#e67e22',
    'rougel': '#27ae60',
    'bertscore': '#3498db',
    'latency': '#9b59b6'
}
```

### Adjusting DPI and Quality

Modify the matplotlib configuration:

```python
plt.rcParams.update({
    'figure.dpi': 300,  # Change to desired DPI
    'savefig.dpi': 300,  # Change to desired DPI
    # ... other settings
})
```

## Metrics Explained

- **ROUGE-1**: Unigram overlap between generated and reference text
- **ROUGE-2**: Bigram overlap between generated and reference text
- **ROUGE-L**: Longest common subsequence overlap
- **BERTScore**: Semantic similarity using BERT embeddings
- **Latency**: Response time in milliseconds

## Best Practices for Journal Articles

1. **Figure Resolution**: All figures are saved at 300 DPI, which is standard for most journals
2. **Color Accessibility**: Colors chosen to be distinguishable in grayscale
3. **Font Size**: Optimized for readability at typical journal figure sizes
4. **Aspect Ratio**: Figures sized to fit common journal layouts (single or double column)
5. **File Format**: PNG format for compatibility with most submission systems

## Troubleshooting

### Missing CSV Files
If CSV files are not found, the script will use placeholder values and display a warning. To use real data:
1. Ensure CSV files are in the same directory as the script
2. Check that CSV filenames match exactly
3. Verify CSV format matches expected structure

### Import Errors
Ensure all required packages are installed:
```bash
pip install pandas matplotlib seaborn numpy
```

### Display Issues
If figures don't display correctly:
1. Check matplotlib backend settings
2. Ensure sufficient memory for high-DPI figures
3. Verify file write permissions in the output directory

## Citation

If you use this visualization code in your research, please cite appropriately.

## License

This code is provided as-is for research purposes.

## Contact

For questions or issues, please contact the development team.
