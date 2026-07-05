import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- COLOR SCHEME from previous charts ---
COLOR_STRUCT_VALID = '#1f77b4'   # Dark Blue
COLOR_GCBERT_VALID = '#ff7f0e'   # Dark Orange
COLOR_STRUCT_TEST = '#aec7e8'    # Light Blue
COLOR_GCBERT_TEST = '#ffbb78'    # Light Orange
# -----------------------------------------

# Data - single metric
metric_name = 'Feasible Chains'
struct_only_valid = 0.76
gcbert_valid = 0.84
struct_only_test = 0.74
gcbert_test = 0.82

# Create figure
fig, (ax_chart, ax_table) = plt.subplots(2, 1, figsize=(10, 7))
fig.subplots_adjust(top=0.85, hspace=0.35)
ax_table.axis('off')

# Create single group of bars at position 0
x = 0
width = 0.18

# Create bars with proper spacing
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
def label_bars(bars, value):
    for bar in bars:
        height = bar.get_height()
        ax_chart.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                      f'{value:.2f}', ha='center', va='bottom',
                      fontsize=10, fontweight='bold', color='#333333',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                               edgecolor='gray', alpha=0.8))

label_bars(bars1, struct_only_valid)
label_bars(bars2, gcbert_valid)
label_bars(bars3, struct_only_test)
label_bars(bars4, gcbert_test)

# Chart styling
ax_chart.set_xlabel('Metric', fontsize=13, fontweight='bold', labelpad=10)
ax_chart.set_ylabel('Score', fontsize=13, fontweight='bold', labelpad=10)
ax_chart.set_title('Chain Feasibility', fontsize=16, fontweight='bold', pad=20)
ax_chart.set_xticks([0])
ax_chart.set_xticklabels([metric_name], fontsize=11)
ax_chart.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
ax_chart.grid(axis='y', linestyle='--', alpha=0.6, color='gray')
ax_chart.set_facecolor('#fafafa')
ax_chart.set_ylim(0, 0.95)  # Adjusted for feasibility scores

# Data table
table_data_display = [['Feasible Chains', '0.76', '0.84', '0.74', '0.82']]

table = ax_table.table(cellText=table_data_display,
                       colLabels=['Metric', 'Struct-only (Valid)', 'GCBERT+Struct (Valid)', 
                                 'Struct-only (Test)', 'GCBERT+Struct (Test)'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.25, 0.1875, 0.1875, 0.1875, 0.1875])

# Table styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Header row
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#424242')
    cell.set_text_props(weight='bold', color='white', fontsize=11)

# Data row with color coding
for j in range(5):
    cell = table[(1, j)]
    
    if j == 0:  # Metric column
        cell.set_facecolor('#f5f5f5')
        cell.set_text_props(ha='left', fontsize=10, weight='bold')
    elif j == 1:  # Struct-only (Valid) - light blue
        cell.set_facecolor('#e3f2fd')
        cell.set_text_props(ha='center', fontsize=10)
    elif j == 2:  # GCBERT+Struct (Valid) - light orange, bold
        cell.set_facecolor('#ffedd8')
        cell.set_text_props(ha='center', fontsize=10, weight='bold')
    elif j == 3:  # Struct-only (Test) - lighter blue
        cell.set_facecolor('#e1f5fe')
        cell.set_text_props(ha='center', fontsize=10)
    elif j == 4:  # GCBERT+Struct (Test) - lighter orange, bold
        cell.set_facecolor('#fff8e1')
        cell.set_text_props(ha='center', fontsize=10, weight='bold')

plt.tight_layout()
plt.show()