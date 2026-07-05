import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- COLOR SCHEME from reference image ---
COLOR_STRUCT_VALID = '#1f77b4'   # Dark Blue
COLOR_GCBERT_VALID = '#ff7f0e'   # Dark Orange
COLOR_STRUCT_TEST = '#aec7e8'    # Light Blue
COLOR_GCBERT_TEST = '#ffbb78'    # Light Orange
# -----------------------------------------

# Data
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUROC', 'AUPRC']
struct_only_valid = [0.954, 0.320, 0.530, 0.400, 0.820, 0.300]
gcbert_valid = [0.963, 0.450, 0.660, 0.540, 0.890, 0.450]
struct_only_test = [0.953, 0.310, 0.520, 0.390, 0.810, 0.280]
gcbert_test = [0.965, 0.440, 0.640, 0.520, 0.880, 0.430]

# Create figure with layout to prevent label cutoff
fig, (ax_chart, ax_table) = plt.subplots(2, 1, figsize=(16, 10))
fig.subplots_adjust(top=0.85, hspace=0.35)
ax_table.axis('off')

# Bar positions
x = np.arange(len(metrics))
width = 0.2

# Create 4 grouped bars per metric
bars1 = ax_chart.bar(x - 1.5*width, struct_only_valid, width,
                     label='Struct-only (Valid)', color=COLOR_STRUCT_VALID,
                     edgecolor='white', linewidth=1.5, alpha=0.9)

bars2 = ax_chart.bar(x - 0.5*width, gcbert_valid, width,
                     label='GCBERT+Struct (Valid)', color=COLOR_GCBERT_VALID,
                     edgecolor='white', linewidth=1.5, alpha=0.9)

bars3 = ax_chart.bar(x + 0.5*width, struct_only_test, width,
                     label='Struct-only (Test)', color=COLOR_STRUCT_TEST,
                     edgecolor='white', linewidth=1.5, alpha=0.8)

bars4 = ax_chart.bar(x + 1.5*width, gcbert_test, width,
                     label='GCBERT+Struct (Test)', color=COLOR_GCBERT_TEST,
                     edgecolor='white', linewidth=1.5, alpha=0.8)

# Add value labels with white background boxes
def label_bars(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        label_text = f'{value:.3f}'
        
        ax_chart.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                      label_text, ha='center', va='bottom',
                      fontsize=8, fontweight='bold', color='#333333',
                      bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                               edgecolor='gray', alpha=0.8))

label_bars(bars1, struct_only_valid)
label_bars(bars2, gcbert_valid)
label_bars(bars3, struct_only_test)
label_bars(bars4, gcbert_test)

# Chart styling
ax_chart.set_xlabel('Metric', fontsize=13, fontweight='bold', labelpad=10)
ax_chart.set_ylabel('Score', fontsize=13, fontweight='bold', labelpad=10)
ax_chart.set_title('Model Performance Metrics: Valid vs Test Comparison', 
                   fontsize=16, fontweight='bold', pad=20)
ax_chart.set_xticks(x)
ax_chart.set_xticklabels(metrics, fontsize=11)
ax_chart.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, shadow=True)
ax_chart.grid(axis='y', linestyle='--', alpha=0.6, color='gray')
ax_chart.set_facecolor('#fafafa')
ax_chart.set_ylim(0, 1.05)  # Extra headroom

# Data table
table_data_display = []
for i, metric in enumerate(metrics):
    table_data_display.append([
        metric,
        f"{struct_only_valid[i]:.3f}",
        f"{gcbert_valid[i]:.3f}",
        f"{struct_only_test[i]:.3f}",
        f"{gcbert_test[i]:.3f}"
    ])

table = ax_table.table(cellText=table_data_display,
                       colLabels=['Metric', 'Struct-only (Valid)', 'GCBERT+Struct (Valid)', 
                                 'Struct-only (Test)', 'GCBERT+Struct (Test)'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.20, 0.20, 0.20, 0.20, 0.20])

# Table styling
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 3)

# Header row
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#424242')
    cell.set_text_props(weight='bold', color='white', fontsize=11)

# Data rows with column-based color coding
for i in range(1, len(table_data_display) + 1):
    for j in range(5):
        cell = table[(i, j)]
        
        # Metric column
        if j == 0:
            cell.set_facecolor('#f5f5f5')
            cell.set_text_props(ha='left', fontsize=9, weight='bold')
        
        # Struct-only (Valid) - light blue
        elif j == 1:
            cell.set_facecolor('#e3f2fd')
            cell.set_text_props(ha='center', fontsize=9, weight='normal')
        
        # GCBERT+Struct (Valid) - light orange, bold
        elif j == 2:
            cell.set_facecolor('#ffedd8')
            cell.set_text_props(ha='center', fontsize=9, weight='bold')
        
        # Struct-only (Test) - lighter blue
        elif j == 3:
            cell.set_facecolor('#e1f5fe')
            cell.set_text_props(ha='center', fontsize=9, weight='normal')
        
        # GCBERT+Struct (Test) - lighter orange, bold
        elif j == 4:
            cell.set_facecolor('#fff8e1')
            cell.set_text_props(ha='center', fontsize=9, weight='bold')

plt.tight_layout()
plt.show()