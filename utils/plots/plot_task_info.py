import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

# Plot my tasks from detailed_task_type
def plot_task_distribution(data):
    titles = []
    categories = []
    focus_areas = []

    for item in data:
        for title, details in item.items():
            if details["Categories"] and details["FocusArea"]:
                for category in details["Categories"]:
                    for focus_area in details["FocusArea"]:
                        titles.append(str(title))
                        categories.append(str(category))
                        focus_areas.append(str(focus_area))
            else:
                continue

    # Create a DataFrame
    df = pd.DataFrame({
        "Title": titles,
        "Categories": categories,
        "FocusArea": focus_areas
    })

    # Count the occurrences of each category and focus area
    category_counts = df["Categories"].value_counts()
    focusarea_counts = df["FocusArea"].value_counts()

    # Plot the data
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot category counts
    axes[0].bar(category_counts.index, category_counts.values, color='skyblue')
    axes[0].set_title('Category Counts')
    axes[0].set_xlabel('Categories')
    axes[0].set_ylabel('Counts')
    axes[0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

    # Plot focus area counts
    axes[1].bar(focusarea_counts.index, focusarea_counts.values, color='lightgreen')
    axes[1].set_title('Focus Area Counts')
    axes[1].set_xlabel('Focus Areas')
    axes[1].set_ylabel('Counts')
    axes[1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

    plt.tight_layout()
    plt.show(block=False)

def plot_statusEdits(statusEdits, idx):
    G = nx.DiGraph()

    # Add edges to the graph
    for edit in statusEdits:
        G.add_edge(edit['from'], edit['to'], createdAt=edit['createdAt'])

    # Define positions for nodes to create a hierarchical layout
    pos = nx.spring_layout(G)

    # Draw the graph
    plt.figure(figsize=(16, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold', arrowsize=20)
    edge_labels = {(u, v): f"{d['createdAt']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title(f'Status Edits Flow Diagram {idx + 1}')
    plt.show()

def plot_status_edits(status_edits, idx):
    """
    Plot a flow diagram of status edits as a directed graph.

    Parameters example:
    status_edits = [
        {'from': 'Open', 'to': 'In Progress', 'createdAt': '2023-05-01'},
        {'from': 'In Progress', 'to': 'Review', 'createdAt': '2023-05-02'},
        {'from': 'Review', 'to': 'Closed', 'createdAt': '2023-05-03'}
    ]
    """
    G = nx.DiGraph()

    for edit in status_edits:
        G.add_edge(edit['from'], edit['to'], createdAt=edit['createdAt'])

    pos = nx.spring_layout(G)

    plt.figure(figsize=(16, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold', arrowsize=20)
    edge_labels = {(u, v): f"{d['createdAt']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title(f'Status Edits Flow Diagram {idx + 1}')
    plt.show()