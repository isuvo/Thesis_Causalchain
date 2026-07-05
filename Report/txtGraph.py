import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

# --- Color scheme ---
ENTRY_COLOR = '#d4edda'      # Light green
BRANCH_COLOR = '#fff3cd'     # Light yellow
PROPAGATION_COLOR = '#cce5ff' # Light blue
SINK_COLOR = '#f8d7da'       # Light red
DEADEND_COLOR = '#e2e3e5'    # Gray for dead-end paths
EDGE_HIGHLIGHT = '#ff6b35'   # Orange for vulnerable path
# -----------------------------------------

# Build tree structure
G = nx.DiGraph()

# Add nodes with attributes (id, label, prob, file_line, node_type)
nodes = [
    ("root", "fgets(buf, sizeof(buf), stdin)", 0.461, "main.c:15", "entry"),
    ("cond_null", "if (fgets(...) == NULL)", 0.461, "main.c:15", "branch"),
    ("return0", "return 0;", 0.458, "main.c:16", "deadend"),
    ("block1", "}", 0.446, "main.c:17", "propagation"),
    ("strlen", "n = strlen(buf);", 0.520, "main.c:19", "propagation"),
    ("cond_trim", "if (n > 0 && ...)", 0.424, "main.c:20", "branch"),
    ("transition", "n = sizeof(exec_local) - 1;", 0.467, "shell.c:40", "transition"),
    ("null_term", "exec_local[n] = '\\0';", 0.448, "shell.c:44", "propagation"),
    ("sink", "system(exec_local);", 0.547, "shell.c:45", "sink")
]

for node_id, label, prob, file_line, node_type in nodes:
    G.add_node(node_id, label=label, probability=prob, 
               file_line=file_line, node_type=node_type)

# Add edges with types and probabilities
edges = [
    ("root", "cond_null", "CFG", 0.461),
    ("cond_null", "return0", "CFG", 0.458),
    ("cond_null", "block1", "CFG", 0.446),  # Vulnerable path
    ("block1", "strlen", "CFG", 0.520),
    ("strlen", "cond_trim", "CFG/DFG", 0.424),
    ("cond_trim", "transition", "CPG", 0.467),  # Vulnerable path
    ("transition", "null_term", "DFG", 0.448),
    ("null_term", "sink", "CFG", 0.547)
]

for u, v, edge_type, prob in edges:
    G.add_edge(u, v, edge_type=edge_type, probability=prob)

# --- Manual tree positioning ---
pos = {
    "root": (0, 0),
    "cond_null": (1.5, 0),
    "return0": (3, 1.5),      # Upper branch (dead-end)
    "block1": (3, -1.5),      # Lower branch (vulnerable)
    "strlen": (4.5, -1.5),
    "cond_trim": (6, -1.5),
    "transition": (7.5, -3),   # Cross-file transition
    "null_term": (9, -3),
    "sink": (10.5, -3)
}

# --- Plotting ---
fig, (ax_graph, ax_table) = plt.subplots(2, 1, figsize=(18, 10))
ax_table.axis('off')

# Node colors based on type
node_colors = {
    "entry": ENTRY_COLOR,
    "branch": BRANCH_COLOR,
    "deadend": DEADEND_COLOR,
    "propagation": PROPAGATION_COLOR,
    "transition": '#d1ecf1',
    "sink": SINK_COLOR
}

# Draw nodes
for node_id in G.nodes():
    node_type = G.nodes[node_id]['node_type']
    nx.draw_networkx_nodes(G, pos, nodelist=[node_id], ax=ax_graph,
                           node_color=node_colors[node_type],
                           node_size=4000, alpha=0.9, 
                           edgecolors='black', linewidths=2)

# Draw edges with styles
edge_widths = []
edge_colors = []
edge_styles = []
edge_labels = {}

for u, v, data in G.edges(data=True):
    edge_type = data['edge_type']
    prob = data['probability']
    edge_labels[(u, v)] = f"{edge_type}\nP={prob:.3f}"
    
    # Highlight vulnerable path
    if v in ["block1", "transition", "sink"]:
        edge_widths.append(4)
        edge_colors.append(EDGE_HIGHLIGHT)
        edge_styles.append('solid')
    else:
        edge_widths.append(2)
        edge_colors.append('gray')
        edge_styles.append('dashed')

nx.draw_networkx_edges(G, pos, ax=ax_graph, arrowstyle='->', arrowsize=25,
                       edge_color=edge_colors, width=edge_widths, style=edge_styles)

# Node labels (code snippets)
node_labels = {node_id: G.nodes[node_id]['label'] for node_id in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax_graph,
                        font_size=9, font_weight='bold')

# Edge labels
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax_graph,
                             font_size=8, font_color='darkblue')

# --- Summary table ---
table_data = [
    ['Stage', 'File', 'Line', 'Operation Type', 'Probability'],
    ['Root', 'main.c', '15', 'Input entry (fgets)', '0.461'],
    ['Branch', 'main.c', '15-17', 'NULL check → dead-end vs continuation', '0.458→0.446'],
    ['Propagation', 'main.c', '19-20', 'strlen + whitespace trimming', '0.520→0.424'],
    ['Transition', 'shell.c', '40', 'Cross-file semantic entry', '0.467'],
    ['Propagation', 'shell.c', '44', 'String null-termination', '0.448'],
    ['Sink', 'shell.c', '45', 'Command execution (system)', '0.547']
]

table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                       colWidths=[0.15, 0.15, 0.10, 0.35, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Header styling
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#424242')
    cell.set_text_props(weight='bold', color='white', fontsize=11)

# Data styling
for i in range(1, len(table_data)):
    for j in range(5):
        cell = table[(i, j)]
        if j == 0:  # Stage column
            cell.set_facecolor('#e9ecef')
            cell.set_text_props(weight='bold', fontsize=9)
        else:
            cell.set_facecolor('#f8f9fa')
            cell.set_text_props(fontsize=9)

ax_graph.axis('off')

plt.suptitle('Chain Analysis: fgets → system Path', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()