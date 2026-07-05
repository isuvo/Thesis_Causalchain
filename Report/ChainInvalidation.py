import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- COLOR SCHEME from previous image ---
COLOR_STRUCT_VALID = '#1f77b4'   # Dark Blue
COLOR_GCBERT_VALID = '#ff7f0e'   # Dark Orange
COLOR_STRUCT_TEST = '#aec7e8'    # Light Blue
COLOR_GCBERT_TEST = '#ffbb78'    # Light Orange
# ----------------------------------------

# Data for bar chart (intervention rates only)
segments = ['Inv. Guard', 'Inv. Unbind', 'Inv. Sink']
struct_only_valid = [0.46, 0.54, 0.91]
gcbert_valid = [0.55, 0.61, 0.94]
struct_only_test = [0.44, 0.52, 0.89]
gcbert_test = [0.53, 0.60, 0.93]

# Create figure
fig, (ax_chart, ax_table) = plt.subplots(2, 1, figsize=(16, 10))
fig.subplots_adjust(top=0.85, hspace=0.35)
ax_table.axis('off')

# Bar positions
x = np.arange(len(segments))
width = 0.2

# Create bars
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

# Add value labels with background boxes
def label_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax_chart.text(bar.get_x() + bar.get_width() / 2., height + 0.008,
                      f'{height:.2f}', ha='center', va='bottom',
                      fontsize=9, fontweight='bold', color='#333333',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                               edgecolor='gray', alpha=0.8))

label_bars(bars1)
label_bars(bars2)
label_bars(bars3)
label_bars(bars4)

# Chart styling
ax_chart.set_xlabel('Intervention Type', fontsize=13, fontweight='bold', labelpad=10)
ax_chart.set_ylabel('Invalidation Rate', fontsize=13, fontweight='bold', labelpad=10)
ax_chart.set_title('Chain invalidation rates and score deltas under interventions', 
                   fontsize=16, fontweight='bold', pad=20)
ax_chart.set_xticks(x)
ax_chart.set_xticklabels(segments, fontsize=11)
ax_chart.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, shadow=True)
ax_chart.grid(axis='y', linestyle='--', alpha=0.6, color='gray')
ax_chart.set_facecolor('#fafafa')
ax_chart.set_ylim(0, 1.02)  # Extra headroom for labels

# Data table (includes ΔScore column)
table_data_display = [
    ['Valid', 'Struct-only', '0.46', '0.54', '0.91', '–1.27'],
    ['Valid', 'GCBERT+Struct', '0.55', '0.61', '0.94', '–1.41'],
    ['Test', 'Struct-only', '0.44', '0.52', '0.89', '–1.21'],
    ['Test', 'GCBERT+Struct', '0.53', '0.60', '0.93', '–1.36']
]

table = ax_table.table(cellText=table_data_display,
                       colLabels=['Split', 'Variant', 'Inv. Guard', 'Inv. Unbind', 'Inv. Sink', 'ΔScore (sink)'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.12, 0.25, 0.175, 0.175, 0.175, 0.175])

# Table styling
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 3)

# Header row
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#424242')
    cell.set_text_props(weight='bold', color='white', fontsize=11)

# Data rows
for i in range(1, len(table_data_display) + 1):
    is_gcbert = 'GCBERT+Struct' in table_data_display[i-1][1]
    
    for j in range(6):
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
            
        # Value columns - tinted by method, bold for GCBERT
        else:
            cell.set_text_props(weight='bold' if is_gcbert else 'normal')
            if is_gcbert:
                cell.set_facecolor('#ffedd8')  # Light orange
            else:
                cell.set_facecolor('#e3f2fd')  # Light blue

plt.tight_layout()
plt.show()