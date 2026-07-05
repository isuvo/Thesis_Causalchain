import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- COLOR SCHEME from previous charts ---
COLOR_STRUCT_VALID = '#1f77b4'   # Dark Blue
COLOR_GCBERT_VALID = '#ff7f0e'   # Dark Orange
COLOR_STRUCT_TEST = '#aec7e8'    # Light Blue
COLOR_GCBERT_TEST = '#ffbb78'    # Light Orange
# -----------------------------------------

# Data for plotting
metrics = ['IPA rate', 'Both(call+ret)', 'Mean call depth']
struct_only_valid = [0.618, 0.402, 1.27]
gcbert_valid = [0.708, 0.486, 1.32]
struct_only_test = [0.603, 0.389, 1.24]
gcbert_test = [0.691, 0.471, 1.30]

# Create figure with proper layout
fig, (ax_chart, ax_table) = plt.subplots(2, 1, figsize=(14, 9))
fig.subplots_adjust(top=0.85, hspace=0.35)  # Prevent label cutoff
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
        label_text = f'{value:.3f}' if value < 1 else f'{value:.2f}'
        
        ax_chart.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                      label_text, ha='center', va='bottom',
                      fontsize=9, fontweight='bold', color='#333333',
                      bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                               edgecolor='gray', alpha=0.8))

label_bars(bars1, struct_only_valid)
label_bars(bars2, gcbert_valid)
label_bars(bars3, struct_only_test)
label_bars(bars4, gcbert_test)

# Chart styling
ax_chart.set_xlabel('Interprocedural Metric', fontsize=13, fontweight='bold', labelpad=10)
ax_chart.set_ylabel('Score / Depth', fontsize=13, fontweight='bold', labelpad=10)
ax_chart.set_title('Interprocedural Structure in Predicted Chains', 
                   fontsize=16, fontweight='bold', pad=20)
ax_chart.set_xticks(x)
ax_chart.set_xticklabels(metrics, fontsize=11)
ax_chart.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, shadow=True)
ax_chart.grid(axis='y', linestyle='--', alpha=0.6, color='gray')
ax_chart.set_facecolor('#fafafa')
ax_chart.set_ylim(0, 1.45)  # Adjusted for max value (1.32)

# Data table
table_data_display = [
    ['Valid', 'Struct-only', '0.618', '0.402', '1.27'],
    ['Valid', 'GCBERT+Struct', '0.708', '0.486', '1.32'],
    ['Test', 'Struct-only', '0.603', '0.389', '1.24'],
    ['Test', 'GCBERT+Struct', '0.691', '0.471', '1.30']
]

table = ax_table.table(cellText=table_data_display,
                       colLabels=['Split', 'Variant', 'IPA rate', 'Both(call+ret)', 'Mean call depth'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.12, 0.25, 0.21, 0.21, 0.21])

# Table styling
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 3)

# Header row
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#424242')
    cell.set_text_props(weight='bold', color='white', fontsize=11)

# Data rows with color coding
for i in range(1, len(table_data_display) + 1):
    is_gcbert = 'GCBERT+Struct' in table_data_display[i-1][1]
    
    for j in range(5):
        cell = table[(i, j)]
        
        # Variant column with method colors
        if j == 1:
            if is_gcbert:
                cell.set_facecolor('#ff7f0e')  # Dark orange
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#1f77b4')  # Dark blue
                cell.set_text_props(weight='bold', color='white')
        
        # Split column - light gray
        elif j == 0:
            cell.set_facecolor('#f5f5f5')
            cell.set_text_props(weight='bold')
            
        # Value columns - tinted by method
        else:
            cell.set_text_props(weight='bold' if is_gcbert else 'normal')
            cell.set_facecolor('#ffedd8' if is_gcbert else '#e3f2fd')

plt.tight_layout()
plt.show()