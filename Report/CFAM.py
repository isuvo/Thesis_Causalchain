import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- DISTINCT COLOR SCHEME ---
# Use a color picker tool to replace with your exact colors
COLOR_STRUCT_VALID = '#4A90E2'  # Replace with your Struct-only color
COLOR_GCBERT_VALID = '#F5A623' 
COLOR_STRUCT_TEST = '#1f77b4'   # Light Blue
COLOR_GCBERT_TEST = '#d62728'    # Light Red
# -----------------------------



# Data
data = {
    'Segment': ['Overall (all)', 'Source (Src)', 'Sanitizer (San)', 'Propagation (Prop)', 'Sink'],
    'Struct-only (Valid)': [0.42, 0.10, 0.09, 0.13, 0.10],
    'GCBERT+Struct (Valid)': [0.51, 0.12, 0.12, 0.16, 0.11],
    'Struct-only (Test)': [0.40, 0.09, 0.09, 0.12, 0.10],
    'GCBERT+Struct (Test)': [0.49, 0.11, 0.11, 0.15, 0.12]
}
df = pd.DataFrame(data)

# Create figure with generous top margin
fig, (ax_chart, ax_table) = plt.subplots(2, 1, figsize=(16, 10))
fig.subplots_adjust(top=0.90, hspace=0.3)  # Prevent label cutoff
ax_table.axis('off')

# Bar positions
x = np.arange(len(df))
width = 0.2

# Create bars with distinct colors
bars1 = ax_chart.bar(x - 1.5*width, df['Struct-only (Valid)'], width,
                     label='Struct-only (Valid)', color=COLOR_STRUCT_VALID,
                     edgecolor='white', linewidth=1.5, alpha=0.9)

bars2 = ax_chart.bar(x - 0.5*width, df['GCBERT+Struct (Valid)'], width,
                     label='GCBERT+Struct (Valid)', color=COLOR_GCBERT_VALID,
                     edgecolor='white', linewidth=1.5, alpha=0.9)

bars3 = ax_chart.bar(x + 0.5*width, df['Struct-only (Test)'], width,
                     label='Struct-only (Test)', color=COLOR_STRUCT_TEST,
                     edgecolor='white', linewidth=1.5, alpha=0.8)

bars4 = ax_chart.bar(x + 1.5*width, df['GCBERT+Struct (Test)'], width,
                     label='GCBERT+Struct (Test)', color=COLOR_GCBERT_TEST,
                     edgecolor='white', linewidth=1.5, alpha=0.8)

# Add value labels with background box
def label_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax_chart.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                      f'{height:.2f}', ha='center', va='bottom',
                      fontsize=8, fontweight='bold', color='#333333',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))

label_bars(bars1)
label_bars(bars2)
label_bars(bars3)
label_bars(bars4)

# Chart styling
ax_chart.set_xlabel('Code Segment', fontsize=13, fontweight='bold', labelpad=10)
ax_chart.set_ylabel('Performance Score', fontsize=13, fontweight='bold', labelpad=10)
ax_chart.set_title('Segment-wise Performance: Valid vs Test Comparison', 
                   fontsize=16, fontweight='bold', pad=20)
ax_chart.set_xticks(x)
ax_chart.set_xticklabels(df['Segment'], fontsize=11)
ax_chart.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, shadow=True)
ax_chart.grid(axis='y', linestyle='--', alpha=0.6)
ax_chart.set_facecolor('#fafafa')

# Increase y-limit to prevent label cutoff
ax_chart.set_ylim(0, 0.60)

# Data table
table_data = []
for _, row in df.iterrows():
    table_data.append([
        row['Segment'],
        f"{row['Struct-only (Valid)']:.2f}",
        f"{row['GCBERT+Struct (Valid)']:.2f}",
        f"{row['Struct-only (Test)']:.2f}",
        f"{row['GCBERT+Struct (Test)']:.2f}"
    ])

table = ax_table.table(cellText=table_data,
                       colLabels=['Segment', 'Struct-only (Valid)', 'GCBERT+Struct (Valid)', 
                                 'Struct-only (Test)', 'GCBERT+Struct (Test)'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.30, 0.175, 0.175, 0.175, 0.175])

# Table styling
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.8)

# Header row
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor("#4C4B4B")
    cell.set_text_props(weight='bold', color='white', fontsize=10)

# Data rows
for i in range(1, len(table_data) + 1):
    segment_name = table_data[i-1][0]
    
    # Segment column
    cell = table[(i, 0)]
    if 'Overall' in segment_name:
        cell.set_facecolor("#2fd394")
        cell.set_text_props(ha='left', fontsize=9, weight='bold', color='white')
    else:
        cell.set_facecolor("#e95f47")
        cell.set_text_props(ha='left', fontsize=9)
    
    # Value columns
    for j in range(1, 5):
        cell = table[(i, j)]
        if 'Overall' in segment_name:
            cell.set_facecolor("#9DE4B4")
        else:
            if j == 1: cell.set_facecolor('#E3F2FD')    # Blue tint
            elif j == 2: cell.set_facecolor("#ECB3B3")  # Red tint
            elif j == 3: cell.set_facecolor('#E1F5FE')  # Light blue
            else: cell.set_facecolor("#ECB3B3")         # Light red
        
        cell.set_text_props(ha='center', fontsize=9, weight='bold')

plt.tight_layout()
plt.show()