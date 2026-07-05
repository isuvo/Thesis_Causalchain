import matplotlib.pyplot as plt
import numpy as np

# --- 1. DEFINE STANDARD COLORS ---
# Standard colors: Blue (Train), Orange (Valid), Green (Test)
STANDARD_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']

# --- 2. DATA CALCULATION ---

# Raw Data (from original tables)
splits = ['Train', 'Valid', 'Test']
records = [185791, 23224, 23224]
non_vuln_files = [180259, 22503, 22554]
vuln_files = [5532, 721, 670]
graphs = [3438, 2905, 2915]

data_caller = [12.59, 13.10, 12.97]
data_callee = [28.95, 29.26, 29.17]
data_both   = [8.72, 9.21, 9.03]

# Calculate Vulnerability Percentages (for Pie Charts & Table)
# Note: Vuln% is provided as 2.98, 3.10, 2.88 in the original data,
# but we are calculating the total distribution of records and graphs.

# To show vulnerability in a pie chart, we need a different approach:
# Pie Chart 1: Distribution of Total Records (185k vs 23k vs 23k) -> Kept
# Pie Chart 2: Distribution of Graph Instances (3.4k vs 2.9k vs 2.9k) -> Kept

# --- 3. CREATE FIGURE ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
fig.subplots_adjust(top=0.92, hspace=0.35, wspace=0.3, left=0.08, right=0.95)

# ==========================================
# Chart 1: File Records Distribution (PIE CHART)
# ==========================================
wedges1, texts1, autotexts1 = ax1.pie(records, labels=splits, autopct='%1.1f%%',
                                      colors=STANDARD_COLORS, startangle=90,
                                      textprops={'fontsize': 11})

for autotext in autotexts1:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax1.set_title('File Records Distribution by Split', fontsize=12, fontweight='bold')
ax1.legend(wedges1, splits, title="Split", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))


# ==========================================
# Chart 2: Graph Instances Distribution (PIE CHART)
# ==========================================
wedges2, texts2, autotexts2 = ax2.pie(graphs, labels=splits, autopct='%1.1f%%',
                                      colors=STANDARD_COLORS, startangle=90,
                                      textprops={'fontsize': 11})

for autotext in autotexts2:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax2.set_title('Graph Instances Distribution by Split', fontsize=12, fontweight='bold')
ax2.legend(wedges2, splits, title="Split", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))


# ==========================================
# Chart 3: Interprocedural Connectivity Metrics (Grouped Bar)
# ==========================================
x = np.arange(len(['Caller%', 'Callee%', 'Both%']))
width = 0.25

rects1 = ax3.bar(x - width, [data_caller[0], data_callee[0], data_both[0]], width, 
                 label='Train', color=STANDARD_COLORS[0])
rects2 = ax3.bar(x, [data_caller[1], data_callee[1], data_both[1]], width, 
                 label='Valid', color=STANDARD_COLORS[1])
rects3 = ax3.bar(x + width, [data_caller[2], data_callee[2], data_both[2]], width, 
                 label='Test', color=STANDARD_COLORS[2])

ax3.set_ylabel('Percentage (%)', fontsize=11)
ax3.set_title('Interprocedural Connectivity', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(['Caller%', 'Callee%', 'Both%'], fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(axis='y', linestyle='--', alpha=0.3)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax3.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)


# ==========================================
# Chart 4: Consolidated Summary Table with Vulnerability Percentages
# ==========================================
ax4.axis('off')
ax4.set_title('Comprehensive Data Summary', fontsize=12, fontweight='bold', y=1.0)

# Calculate Vulnerable and Non-Vulnerable Percentages for the table
# Vuln% = (Vuln_Files / Total_Records) * 100
# Non-Vuln% = 100 - Vuln%
vuln_percs = [(v / r) * 100 for v, r in zip(vuln_files, records)]
non_vuln_percs = [100 - p for p in vuln_percs]

table_data = [
    ['Metric', 'Train', 'Valid', 'Test'],
    ['Total Records', f'{records[0]:,}', f'{records[1]:,}', f'{records[2]:,}'],
    ['Non-Vulnerable (%)', f'{non_vuln_percs[0]:.2f}%', f'{non_vuln_percs[1]:.2f}%', f'{non_vuln_percs[2]:.2f}%'],
    ['Vulnerable (%)', f'{vuln_percs[0]:.2f}%', f'{vuln_percs[1]:.2f}%', f'{vuln_percs[2]:.2f}%'],
    ['Graph Instances', f'{graphs[0]:,}', f'{graphs[1]:,}', f'{graphs[2]:,}'],
    ['Caller (%)', f'{data_caller[0]:.2f}%', f'{data_caller[1]:.2f}%', f'{data_caller[2]:.2f}%'],
    ['Callee (%)', f'{data_callee[0]:.2f}%', f'{data_callee[1]:.2f}%', f'{data_callee[2]:.2f}%'],
    ['Both (%)', f'{data_both[0]:.2f}%', f'{data_both[1]:.2f}%', f'{data_both[2]:.2f}%']
]

the_table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.2, 0.2, 0.2])
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1, 1.8)

for (row, col), cell in the_table.get_celld().items():
    cell.set_linewidth(1)
    cell.set_edgecolor('#cccccc')
    if row == 0:
        if col == 0: cell.set_facecolor('#f0f0f0')
        elif col == 1: cell.set_facecolor(STANDARD_COLORS[0])
        elif col == 2: cell.set_facecolor(STANDARD_COLORS[1])
        elif col == 3: cell.set_facecolor(STANDARD_COLORS[2])
        cell.set_text_props(weight='bold', color='white' if col > 0 else 'black')
    elif col == 0:
        cell.set_facecolor('#f8f9fa')
        cell.set_text_props(weight='bold', ha='left')

plt.show()