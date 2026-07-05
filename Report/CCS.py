import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- INSERT YOUR COLORS HERE ---
# Replace these hex codes with colors from your image
# You can use a color picker tool like:
# - Windows: PowerToys Color Picker
# - Mac: Digital Color Meter
# - Online: imagecolorpicker.com
COLOR_STRUCT_ONLY = '#4A90E2'  # Replace with your Struct-only color
COLOR_GCBERT = '#F5A623'       # Replace with your GCBERT+Struct color
# ------------------------------

# Data
data = {
    'Edit Type': ['Guard', 'Unbind', 'Sink'],
    'Struct-only': [0.004, 0.009, 0.11],
    'GCBERT+Struct': [0.004, 0.011, 0.13],
    'Interpretation': [
        'Stable to non-causal edits',
        'Moderate sensitivity',
        'Large shift → sink is causal'
    ]
}
df = pd.DataFrame(data)

# Create figure with better proportions
fig, (ax_chart, ax_table) = plt.subplots(2, 1, figsize=(12, 8), 
                                         gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})
ax_table.axis('off')

# Bar positions
x = np.arange(len(df))
width = 0.35

# Create bars with custom colors
bars1 = ax_chart.bar(x - width/2, df['Struct-only'], width,
                     label='Struct-only', color=COLOR_STRUCT_ONLY,
                     edgecolor='white', linewidth=1.5, alpha=0.9)
bars2 = ax_chart.bar(x + width/2, df['GCBERT+Struct'], width,
                     label='GCBERT+Struct', color=COLOR_GCBERT,
                     edgecolor='white', linewidth=1.5, alpha=0.9)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax_chart.text(bar.get_x() + bar.get_width()/2., height + max(df['Struct-only'].max(), df['GCBERT+Struct'].max()) * 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')

# Styling
ax_chart.set_xlabel('Edit Type', fontsize=14, weight='bold', labelpad=10)
ax_chart.set_ylabel('Sensitivity Value', fontsize=14, weight='bold', labelpad=10)
ax_chart.set_title('Model Sensitivity Analysis', fontsize=16, weight='bold', pad=20)
ax_chart.set_xticks(x)
ax_chart.set_xticklabels(df['Edit Type'], fontsize=12)
ax_chart.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
ax_chart.grid(axis='y', linestyle='--', alpha=0.5)
ax_chart.set_facecolor('#fafafa')

# Adjust y-axis for space
max_val = max(df['Struct-only'].max(), df['GCBERT+Struct'].max())
ax_chart.set_ylim(0, max_val * 1.25)

# Create table
table_data = []
for _, row in df.iterrows():
    table_data.append([
        row['Edit Type'],
        f"{row['Struct-only']:.3f}",
        f"{row['GCBERT+Struct']:.3f}",
        row['Interpretation']
    ])

table = ax_table.table(cellText=table_data,
                       colLabels=['Edit Type', 'Struct-only', 'GCBERT+Struct', 'Interpretation'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.15, 0.15, 0.15, 0.55])

# Table styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Header style
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#333333')
    cell.set_text_props(weight='bold', color='white', fontsize=11)

# Data cell styles
for i in range(1, len(table_data) + 1):
    table[(i, 0)].set_facecolor('#f0f0f0')  # Edit Type
    table[(i, 1)].set_facecolor('#E3F2FD')  # Struct-only
    table[(i, 2)].set_facecolor('#FFF3E0')  # GCBERT+Struct
    table[(i, 3)].set_facecolor('#f9f9f9')  # Interpretation
    table[(i, 3)].set_text_props(ha='left', fontsize=10)

plt.tight_layout()
plt.show()