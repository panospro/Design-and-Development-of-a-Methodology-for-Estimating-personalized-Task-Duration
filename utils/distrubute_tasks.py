import heapq
import joblib
import pandas as pd

def load_model():
    return joblib.load('best_bagged_trees_model.pkl')

# Predict execution time for a task given an assignee
def predict_execution_time(task, assignee, model, feature_names):
    # Encode the assignee in the task
    task_with_assignee = task.copy()
    for key in feature_names:
        if key == assignee:
            task_with_assignee[key] = 1
        elif key not in task_with_assignee:
            task_with_assignee[key] = 0
    task_features = pd.DataFrame([task_with_assignee], columns=feature_names)
    return model.predict(task_features)[0]

# Greedy Algorithm for Task Distribution considering assignee's performance
def distribute_tasks(tasks, assignees):
    def adjust_predicted_time(predicted_time):
        if predicted_time == 1:
            return 0.25
        elif predicted_time == 2:
            return 1.25
        elif predicted_time == 3:
            return 2.5
        return predicted_time  # If it's not 1, 2, or 3, return the original value

    model, feature_names = load_model()
    
    # Initialize a dictionary to keep track of the total execution time for each assignee
    assignee_times = {assignee: 0 for assignee in assignees}
    
    # Initialize task assignment dictionary
    task_assignment = {assignee: [] for assignee in assignees}
    
    # For each task, find the best assignee
    for task in tasks:
        best_assignee = None
        best_score = float('inf')
        
        # Evaluate each assignee for the current task
        for assignee in assignees:
            predicted_time = predict_execution_time(task, assignee, model, feature_names)
            adjusted_time = adjust_predicted_time(predicted_time)
            total_time_with_new_task = assignee_times[assignee] + adjusted_time
            
            if total_time_with_new_task < best_score:
                best_score = total_time_with_new_task
                best_assignee = assignee
        
        # Update the total execution time for the best assignee
        assignee_times[best_assignee] += adjusted_time

        # Find the original assignee by checking key lengths (assume all assignee IDs have the same length)
        original_assignee = [k for k, v in task.items() if len(k) == 24 and v == 1]

        # Assign the task to the best assignee found
        task_assignment[best_assignee].append({
            'assigned_to': best_assignee,
            'burnedPoints': task['burnedPoints'],
            'expectedPoints': task['expectedPoints'],
            'class': task['class'],
            'original_assignee': original_assignee,
            'best_class_estimated': predicted_time
        })
    
    return task_assignment
