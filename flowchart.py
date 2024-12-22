import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes
nodes = [
    ("Start", "Start"),
    ("Inputs", "Inputs:\nPlain Text, Cipher Text, Key, etc."),
    ("CheckPlainCipher", "Check: Plain Text + Cipher Text + Key?"),
    ("CheckLengths", "Check: Plain Text Length == Cipher Text Length?"),
    ("Symmetric", "Symmetric Algorithm"),
    ("Transposition", "Transposition"),
    ("Asymmetric", "Asymmetric Algorithm"),
    ("Hash", "Hash Algorithm"),
    ("Classifier", "General Model (e.g., RandomForest Classifier)"),
    ("End", "End")
]

# Add edges
edges = [
    ("Start", "Inputs"),
    ("Inputs", "CheckPlainCipher"),
    ("CheckPlainCipher", "CheckLengths", {"label": "Yes"}),
    ("CheckPlainCipher", "Asymmetric", {"label": "Plain Text + Cipher Text + Key1 + Key2"}),
    ("CheckPlainCipher", "Hash", {"label": "Plain Text + Cipher Text"}),
    ("CheckPlainCipher", "Classifier", {"label": "Cipher Text Only"}),
    ("CheckLengths", "Symmetric", {"label": "No"}),
    ("CheckLengths", "Transposition", {"label": "Yes"}),
    ("Symmetric", "End"),
    ("Transposition", "End"),
    ("Asymmetric", "End"),
    ("Hash", "End"),
    ("Classifier", "End")
]

# Add nodes and edges to the graph
for node_id, label in nodes:
    G.add_node(node_id, label=label)

for edge in edges:
    G.add_edge(*edge[:2], **edge[2] if len(edge) > 2 else {})

# Use graphviz_layout for better flowchart style layout
pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

# Retrieve node labels and edge labels
node_labels = nx.get_node_attributes(G, "label")
edge_labels = nx.get_edge_attributes(G, "label")

# Create plot with adjusted figure size
plt.figure(figsize=(12, 12))

# Draw nodes and edges
nx.draw(G, pos, with_labels=False, node_size=5000, node_color="lightblue", edge_color="black", font_size=8, font_weight="bold")
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_family="sans-serif")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# Set title and display the plot
plt.title("Encryption Algorithm Flowchart", fontsize=16)
plt.axis("off")  # Turn off axis
plt.show()
