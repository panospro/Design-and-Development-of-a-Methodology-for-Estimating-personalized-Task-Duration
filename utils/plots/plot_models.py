import matplotlib.pyplot as plt
import ast
import plotly.express as px
import seaborn as sns
from scipy.stats import mode
import numpy as np
import pandas as pd

# Plot different models cv accuracies and their best average parameters
def plot_best_model(data):
    """
        data: 
            Model,Best Parameters,Average CV Score,Run,Comment
            Medium Tree,"{'max_depth': 7, 'min_samples_leaf': 1, 'min_samples_split': 10}",0.7093078381795195,1,Running multiple configurations without dropped lines
            Boosted Trees,"{'learning_rate': 0.01, 'n_estimators': 200}",0.6737357774968393,1,Running multiple configurations without dropped lines
    """
    def parse_best_parameters(param_str):
        return ast.literal_eval(param_str)

    def calculate_average_best_parameters(data):
        best_params = data['Best Parameters'].apply(parse_best_parameters)
        average_params = {}
        
        for params in best_params:
            for key, value in params.items():
                if key not in average_params:
                    average_params[key] = []
                average_params[key].append(value)
        
        for key in average_params:
            average_params[key] = sum(average_params[key]) / len(average_params[key])
        
        return average_params

    plt.figure(figsize=(10, 6))
    for model in data['Model'].unique():
        subset = data[data['Model'] == model]
        plt.plot(subset['Run'], subset['Average CV Score'], label=model)
        
        # Calculate the average best parameters
        avg_params = calculate_average_best_parameters(subset)
        
        # Annotate the average best parameters on the plot
        last_run = subset.iloc[-1]
        plt.annotate(
            f"Avg Params:\n{avg_params}",
            xy=(last_run['Run'], last_run['Average CV Score']),
            xytext=(last_run['Run'], last_run['Average CV Score'] + 0.01),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.5)
        )

    plt.title('Model Comparison Across Multiple Runs')
    plt.xlabel('Run')
    plt.ylabel('Average CV Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_comparison.png')
    plt.show(block=False)

# Combined heatmap plotting
def plot_heatmap_categories(df):
    most_frequent = lambda x: mode(x)[0]
    df_exploded = df.explode('categories').explode('focus_areas')
    agg_funcs = {'mean': 'Heatmap of Categories and Focus Areas with Class Averages', 
                 most_frequent: 'Heatmap of Categories and Focus Areas with Most Frequent Class'}

    for aggfunc, title in agg_funcs.items():
        pivot_data = df_exploded.pivot_table(index='categories', columns='focus_areas', values='class', aggfunc=aggfunc, fill_value=0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", cbar=True)
        plt.title(title)
        plt.ylabel("Categories")
        plt.xlabel("Focus Areas")
        plt.show(block=False)

# Plotting the histogram for burnedPoints
def plot_for_key(df, key):
    fig = px.histogram(df, x=key, nbins=100, title=f"Distribution of {key} Points")
    fig.update_traces(marker_line_width=1.5, marker_line_color="black")
    fig.update_layout(xaxis_title='Expected Points', yaxis_title='Number of Tasks')
    fig.show()

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances for {model_name}")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show(block=False)

def plot_csv_models(file_path, filter_phrase):
    # Read the CSV data from the file
    df = pd.read_csv(file_path)

    # Extract the parameters from the 'Best Parameters' column
    df['Best Parameters'] = df['Best Parameters'].apply(ast.literal_eval)

    # Filter the data based on the filter_phrase in the 'Comment' column
    filtered_df = df[df['Comment'].str.contains(filter_phrase, case=False)]

    # Get unique comments from the filtered data
    unique_comments = filtered_df['Comment'].unique()

    # Set up color map
    color_map = plt.get_cmap('tab20')
    colors = [color_map(i) for i in range(len(unique_comments))]

    # Plotting the data for each unique comment
    plt.figure(figsize=(12, 8))

    for index, comment in enumerate(unique_comments):
        subset = filtered_df[filtered_df['Comment'] == comment]
        avg_cv_score = subset['Average CV Score'].mean()
        plt.plot(subset['Run'], subset['Average CV Score'], label=f'{comment} (Avg: {avg_cv_score:.4f})', color=colors[index], marker='o')

    plt.title('Bagged Trees Model Performance')
    plt.xlabel('Run')
    plt.ylabel('Average CV Score')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

def plot_task_assignment(task_assignment, assignee_names, categorize_burned_points):
    def prepare_data(task_assignment):
        records = [entry for value in task_assignment.values() for entry in value]
        df = pd.DataFrame(records)
        df['assigned_to'] = df['assigned_to'].map(assignee_names)
        df['expected_class'] = df['expectedPoints'].apply(categorize_burned_points)
        return df

    def plot_task_count(df):
        task_count = df['assigned_to'].value_counts()
        plt.figure(figsize=(10, 6))
        task_count.plot(kind='bar')
        plt.xlabel('Assignee')
        plt.ylabel('Number of Tasks')
        plt.title('Number of Tasks per Assignee')
        plt.xticks(rotation=45)
        plt.show(block=False)

    def plot_time_summary(df):
        classes_sum = df.groupby('assigned_to').agg({
            'best_class_estimated': 'sum',
            'class': 'sum',
            'expected_class': 'sum'
        })
        classes_sum.plot(kind='bar', figsize=(10, 6))
        plt.xlabel('Assignee')
        plt.ylabel('Classes')
        plt.title('Estimated Time, Actual Time, and Expected Time per Assignee')
        plt.xticks(rotation=45)
        plt.legend(['Estimated Time', 'Actual Time', 'Expected Time'])
        plt.show(block=False)

    df = prepare_data(task_assignment)
    plot_task_count(df)
    plot_time_summary(df)
