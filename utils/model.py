import csv
import argparse
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from imblearn.ensemble import RUSBoostClassifier
import warnings 
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Run different models, with model_name i can choose what models to run e.g ['Bagged Trees', 'Medium Tree'] 
def different_models_comparison(df, model_names):
    # Assuming 'class' is the name of your target variable
    X = df.drop(columns=['class'])
    y = df['class']

    # Define hyperparameter grids
    param_grids = {
        'Medium Tree': {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
        'Boosted Trees': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
        'Bagged Trees': {'n_estimators': [10, 50, 100], 'estimator__max_depth': [3, 5, 7]},
        'RUSBoosted Trees': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
    }

    base_tree = DecisionTreeClassifier()
    models = {
        'Medium Tree': DecisionTreeClassifier(),
        'Boosted Trees': AdaBoostClassifier(estimator=base_tree),
        'Bagged Trees': BaggingClassifier(estimator=base_tree),
        'RUSBoosted Trees': RUSBoostClassifier(estimator=base_tree)
    }

    # Filter models and param_grids based on model_names argument
    models = {name: model for name, model in models.items() if name in model_names}
    param_grids = {name: params for name, params in param_grids.items() if name in model_names}

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Model evaluation with optional comments.')
    parser.add_argument('--comment', type=str, default='', help='A comment about the changes made.')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs with different random splits.')
    args = parser.parse_args()
    comment, num_runs = args.comment, args.runs

    # File to save the results
    results_file = 'bagged_trees_comparison.csv'

    # Initialize CSV file with headers if not already done
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Model', 'Best Parameters', 'Average CV Score', 'Run', 'Comment'])

    # Train and evaluate models with hyperparameter tuning for different configurations
    for run in range(num_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run)

        for name, model in models.items():
            print(f"Hyperparameter tuning for {name}, Run {run + 1}")
            
            # Perform Grid Search
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            
            # Evaluate the best model
            y_pred = best_model.predict(X_test)
            print(f"Best {name} Classification Report:")
            print(classification_report(y_test, y_pred))
            
            # Cross-validation scores
            scores = cross_val_score(best_model, X, y, cv=5)
            average_cv_score = scores.mean()
            print(f"Cross-validation scores for best {name}: {scores}")
            print(f"Average cross-validation score for best {name}: {average_cv_score}\n")
            
            # Save the results to the file
            with open(results_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, grid_search.best_params_, average_cv_score, run + 1, comment])

            # Plot feature importances for tree-based models
            # plot_feature_importance(best_model, X.columns, name)

def find_best_model(df):
    # df = df.drop(columns=["burnedPoints"]) # Remove solution
    X = df.drop(columns=['class'])
    y = df['class']

    # Define hyperparameter grid for Bagged Trees
    param_grid = {'n_estimators': [10, 50, 100], 'estimator__max_depth': [3, 5, 7]}

    # Base Decision Tree
    base_tree = DecisionTreeClassifier()
    
    # Bagged Trees Model
    model = BaggingClassifier(estimator=base_tree)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Model evaluation with optional comments.')
    parser.add_argument('--comment', type=str, default='', help='A comment about the changes made.')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs with different random splits.')
    args = parser.parse_args()
    comment, num_runs = args.comment, args.runs

    # File to save the results
    results_file = 'bagged_trees_comparison.csv'

    # Initialize CSV file with headers if not already done
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Model', 'Best Parameters', 'Average CV Score', 'Run', 'Comment'])

    # Train and evaluate models with hyperparameter tuning for different configurations
    best_model = None
    best_params = None
    best_score = 0

    for run in range(num_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run)

        print(f"Hyperparameter tuning for Bagged Trees, Run {run + 1}")
        
        # Perform Grid Search
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for Bagged Trees: {grid_search.best_params_}")
        
        # Evaluate the best model
        y_pred = grid_search.best_estimator_.predict(X_test)
        print(f"Best Bagged Trees Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Display Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.best_estimator_.classes_)
        disp.plot()
        plt.show()

        # Cross-validation scores
        scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5)
        average_cv_score = scores.mean()
        print(f"Cross-validation scores for best Bagged Trees: {scores}")
        print(f"Average cross-validation score for best Bagged Trees: {average_cv_score}\n")
        
        # Save the best model if it has the highest average CV score
        if average_cv_score > best_score:
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = average_cv_score
        
        # Save the results to the file
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Bagged Trees', grid_search.best_params_, average_cv_score, run + 1, comment])

    # Save the best model and feature names to disk
    joblib.dump((best_model, X.columns.tolist()), 'best_bagged_trees_model.pkl')
    return best_model, best_params, best_score