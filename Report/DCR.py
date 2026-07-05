import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Extracted Color Scheme from Reference Image ---
# Use a color picker tool to replace these with exact values if needed
COLOR_STRUCT_VALID = '#1f77b4'   # Dark Blue
COLOR_GCBERT_VALID = '#ff7f0e'   # Dark Orange
COLOR_STRUCT_TEST = '#aec7e8'    # Light Blue
COLOR_GCBERT_TEST = '#ffbb78'    # Light Orange
# ----------------------------------------------

# Data
segments = ['Guard', 'Unbind', 'Sink']
struct_only_valid = [0.78, 0.81, 0.93]
gcbert_valid = [0.82, 0.85, 0.95]
struct_only_test = [0.76, 0.79, 0.91]
gcbert_test = [0.81, 0.83, 0.94]

# Create figure with adjusted layout to prevent cutoff
fig, (ax_chart, ax_table) = plt.subplots(2, 1, figsize=(14, 10))
fig.subplots_adjust(top=0.85, hspace=0.35)  # Increased top margin and spacing
ax_table.axis('off')

# Bar positions
x = np.arange(len(segments))
width = 0.2

# Create bars with your color scheme
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
def label_bars(bars):
    for bar in bars:
        height = bar.get_height()
        # Position labels slightly higher and centered
        ax_chart.text(bar.get_x() + bar.get_width() / 2., height + 0.012,
                      f'{height:.2f}', ha='center', va='bottom',
                      fontsize=9, fontweight='bold', color='#333333',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                               edgecolor='gray', alpha=0.8))

label_bars(bars1)
label_bars(bars2)
label_bars(bars3)
label_bars(bars4)

# Chart styling
ax_chart.set_xlabel('Edit Type', fontsize=13, fontweight='bold', labelpad=10)
ax_chart.set_ylabel('Score', fontsize=13, fontweight='bold', labelpad=10)
ax_chart.set_title('Directional consistency rate (DCR; higher is better)', 
                   fontsize=16, fontweight='bold', pad=20)
ax_chart.set_xticks(x)
ax_chart.set_xticklabels(segments, fontsize=11)
ax_chart.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, shadow=True)
ax_chart.grid(axis='y', linestyle='--', alpha=0.6, color='gray')
ax_chart.set_facecolor('#fafafa')
ax_chart.set_ylim(0, 1.02)  # Extra headroom

# Data table
table_data_display = [
    ['Valid', 'Struct-only', '0.78', '0.81', '0.93'],
    ['Valid', 'GCBERT+Struct', '0.82', '0.85', '0.95'],
    ['Test', 'Struct-only', '0.76', '0.79', '0.91'],
    ['Test', 'GCBERT+Struct', '0.81', '0.83', '0.94']
]

table = ax_table.table(cellText=table_data_display,
                       colLabels=['Split', 'Variant', 'Guard', 'Unbind', 'Sink'],
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
            
        # Score columns - tinted by method
        else:
            if is_gcbert:
                cell.set_facecolor('#ffedd8')  # Light orange
            else:
                cell.set_facecolor('#e3f2fd')  # Light blue
            cell.set_text_props(weight='bold' if is_gcbert else 'normal')

plt.tight_layout()
plt.show()