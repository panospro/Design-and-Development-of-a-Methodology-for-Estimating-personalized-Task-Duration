import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

def plot_commit_task_network(data):
    """
    Plot a bar chart of commit frequencies.
    
    Parameters:
    data -> List of dictionaries containing commit information.
    
    Example:
    data = [
        {
            "keywords": "",
            "title": "Check and release lychte",
            "commits": [
                {
                    "commitId": "commitId1",
                }
    """
    G = nx.Graph()

    # for task in data:
    #     task_node = f"Task: {task['title']}"
    #     G.add_node(task_node, bipartite=0)
    #     for commit in task['commits']:
    #         commit_node = f"Commit: {commit['commitId']}"
    #         G.add_node(commit_node, bipartite=1)
    #         G.add_edge(task_node, commit_node)

    # Anonymed data
    for task_index, task in enumerate(data, start=1):
        task_node = f"Task{task_index}"
        G.add_node(task_node, bipartite=0)
        for commit_index, commit in enumerate(task['commits'], start=1):
            commit_node = f"Commit{commit_index + (task_index - 1) * len(task['commits'])}"
            G.add_node(commit_node, bipartite=1)
            G.add_edge(task_node, commit_node)

    pos = nx.spring_layout(G)
    task_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
    commit_nodes = [n for n in G if n not in task_nodes]

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, nodelist=task_nodes, node_color='skyblue', node_size=700, label='Tasks')
    nx.draw_networkx_nodes(G, pos, nodelist=commit_nodes, node_color='lightgreen', node_size=700, label='Commits')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color='black', font_weight='bold')
    plt.title('Task-Commit Bipartite Graph')
    plt.legend()
    plt.show(block=False)

def plot_commit_frequencies(data):
    """
    Plot a bar chart of commit frequencies.
    
    Parameters:
    data -> List of dictionaries containing commit information.
    
    Example:
    data = [
        {
            "keywords": "",
            "title": "Check and release lychte",
            "commits": [
                {
                    "commitId": "commitId1",
                }
    """
    # Extract commit IDs from the data
    commit_ids = []

    for entry in data:
        commits = entry['commits']
        for item in commits:
            if isinstance(item, dict) and 'commitId' in item:
                commit_ids.append(item['commitId'])
    # Count the frequency of each commit ID
    commit_counts = Counter(commit_ids)
    
    # Plotting the data
    plt.figure(figsize=(10, 5))
    plt.bar(commit_counts.keys(), commit_counts.values(), color='skyblue')
    plt.xlabel('Commit ID')
    plt.ylabel('Number of Occurrences')
    plt.title('Occurrences of Commit IDs')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show(block=False)

def plot_commits(data):
    plot_commit_frequencies(data)
    plot_commit_task_network(data)