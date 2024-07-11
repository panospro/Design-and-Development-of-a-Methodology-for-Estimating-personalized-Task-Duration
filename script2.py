import pandas as pd
import json
import warnings
from sklearn.preprocessing import OneHotEncoder
from utils.plots.plot_models import plot_best_model, plot_heatmap_categories, plot_for_key, plot_csv_models, plot_task_assignment
from utils.bert import get_bert_embeddings, run_bert_for_all_combinations
from utils.model_utils import process_assignees, balance_dataset, filter_data, normalize_data, remove_redundant_features
from utils.model import find_best_model
from utils.distrubute_tasks import distribute_tasks
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Categorize 'burnedPoints' into Fast-Medium-Slow
def categorize_burned_points(x): return 1 if x <= 0.5 else 2 if x <= 2 else 3

def pre_process_data(df):
    # Convert burnedPoints to classes
    df['class'] = df['burnedPoints'].apply(categorize_burned_points)
    # plot_heatmap_categories(df)

    # Convert labels to values and join categories, focus_areas, labels if they are lists
    for col in ['categories', 'focus_areas', 'labels']:
        df[col] = df[col].apply(lambda x: ','.join(sorted(x)) if isinstance(x, list) else x)

    # One-hot encode the specified columns
    encoder = OneHotEncoder(sparse_output=False)
    encoded_df = pd.DataFrame(
        encoder.fit_transform(df[['categories', 'focus_areas', 'labels']]),
        columns=encoder.get_feature_names_out()
    )

    # Combine original dataframe with encoded data and drop the original columns
    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1).drop(columns=['categories', 'focus_areas', 'labels'])

    # Balance and filter the dataset
    df = filter_data(df)

    # plot_for_key(df, 'burnedPoints')

    return balance_dataset(df)

def process_data(data):
    df = pd.DataFrame(data)
    df = pre_process_data(df)
    df = normalize_data(df)
    df = process_assignees(df)

    df = get_bert_embeddings(df, columns=["title", "body", "comments", "commitMessages"], n_components=10, method='norm')
    df = remove_redundant_features(df)
    find_best_model(df)
    return df

def main():
    with open('', 'r', encoding='utf-8') as file: # Load the file data that were returned from script 1
        data = json.load(file)

    df = process_data(data)
    # plot_best_model(pd.read_csv('bagged_trees_comparisonAll.csv'))

    # plot_csv_models('bagged_trees_comparison.csv', 'Dataset')

    assignees = ["assignee1", "assignee2", "assignee3"]
    assignee_names = {
        "assignee1Id1": "AssigneeName1",
        "assignee1Id2": "AssigneeName2",
        "assignee1Id3": "AssigneeName3",
    }

    processed_tasks = df.to_dict(orient='records')
    task_assignment = distribute_tasks(processed_tasks, assignees)
    # plot_task_assignment(task_assignment, assignee_names, categorize_burned_points)
    
    input("\nPress Enter to close the graphs...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main.py: {e}")
