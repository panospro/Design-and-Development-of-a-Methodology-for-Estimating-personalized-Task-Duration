from utils.plots.plot_task_info import plot_task_distribution, plot_statusEdits
from utils.plots.plot_commit_info import plot_commits
from utils.plots.plot_people_info import plot_people_results, plot_average_points_per_task
from utils.queries import process_queries, categorize_task_query, enhance_task_query, categorize_commits_query
from utils.utils import get_tasks, get_commits, add_commits_to_tasks, filter_enhanced_tasks

# Some tasks are deleted, for various reasons e.g. the llm_output, or they are filtered. Option 1 is for enrich, 2 is for task_commits
def return_all_tasks(original_tasks, filtered_tasks, option):
    original_task_dict = {task["_id"]: task for task in original_tasks}
    filtered_task_dict = {task["_id"]: task for task in filtered_tasks}

    # Identify missing tasks
    missing_tasks = [task for task_id, task in original_task_dict.items() if task_id not in filtered_task_dict]

    # Update missing tasks with empty fields and add to filtered_tasks
    for task in missing_tasks:
        if option == 1:
            task.update({"focus_areas": [], "categories": []})
        if option == 2:
            task.update({"commits": []})

        filtered_tasks.append(task)
    return filtered_tasks

# Filter task titles to get only 'Code-Related'
def get_code_tasks(tasks, file_name):
    def validate_task_type(task_type):
        if isinstance(task_type, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in task_type.items()):
            return list(task_type.items())
        else:
            raise ValueError("Output structure is not as expected")

    # Normalize quotes by replacing single quotes with double quotes
    # NOTE: Idk which is correct or where is used is correct, when it crashes use the other or something
    def normalize_quotes(title):
        return title.replace("'", '"')
        return title.replace('"',"'")

    task_titles = [normalize_quotes(task['title']) for task in tasks]
    task_types_list = process_queries(task_titles, categorize_task_query, file_name)

    task_type_dict = {
        normalize_quotes(task): task_type 
        for task_types in task_types_list 
        for task, task_type in validate_task_type(task_types)
    }

    return [task for task in tasks if task_type_dict.get(normalize_quotes(task['title'])) == 'Code-Related']

# Add categories and area of focus to tasks
def enrich_tasks(tasks, file_name):
    def normalize_quotes(title):
        # return title.replace("'", '"')
        return title.replace('"', "'")

    def update_task_data(task_data, details):
        task_data.update({
            'categories': details.get("Categories", []),
            'focus_areas': details.get("FocusArea", [])
        })
        return task_data

    enriched_tasks = []
    tasks_titles = {normalize_quotes(task['title']): task for task in tasks}

    detailed_task_type_list = process_queries(tasks_titles, enhance_task_query, file_name)
    detailed_task_type_list = filter_enhanced_tasks(detailed_task_type_list) # We lose some tasks from this

    # plot_task_distribution(detailed_task_type_list)
    for detailed_task_type in detailed_task_type_list:
        for title, details in detailed_task_type.items():
            if title in tasks_titles:
                enriched_tasks.append(update_task_data(tasks_titles[title], details))

    return return_all_tasks(tasks, enriched_tasks, 1)

# Add information about the possible commits of the task
def correlate_tasks_with_commits(tasks, commits, file_name):
    # Filter commits based on combined_llm_output and keep only tasks with commits 
    def filter_tasks(tasks, combined_llm_output):
        task_dict = {task['title']: task for task in tasks}
        filtered_tasks = []

        for output in combined_llm_output:
            for item in output:
                title, commit_ids = item['title'], set(item['commit_ids'])

                if title in task_dict:
                    task = task_dict[title]
                    task['commits'] = [commit for commit in task['commits'] if commit['commitId'] in commit_ids]

                    if task['commits']:
                        filtered_tasks.append(task)

        return filtered_tasks

    def combine_tasks_with_commits(filtered_tasks, tasks):
        task_map = {task['title']: task for task in tasks}

        for task in filtered_tasks:
            task.pop('keywords', None)
            task_title = task['title']
            
            # Update the task with information from the project task map if it exists
            if task_title in task_map:
                task_info = task_map[task_title]
                task.update(task_info)
                task['commits'] = task.get('commits', [])

        return filtered_tasks

    combined_llm_output = []
    tasks_with_commits = add_commits_to_tasks(tasks, commits, f'{file_name}add_commits_to_tasks.json')

    # plot_commits(tasks_with_commits)
    for i, task in enumerate(tasks_with_commits):
        llm_output = process_queries([task], categorize_commits_query, f'{file_name}task_{i}')
        combined_llm_output.extend(llm_output)

    filtered_tasks = filter_tasks(tasks_with_commits, combined_llm_output)
    return combine_tasks_with_commits(filtered_tasks, tasks)

def main():
    users = {
        'user1': { 'id': 'userId1', 'git_name': 'githubUserName1', 'base_path': 'test/user1/'}, # It can either be a user to see specific results or a whole organization
        'user2': { 'id': 'userId2', 'git_name': 'githubUserName2', 'base_path': 'test/user2/'},
    }

    name = 'user1' # Change name to see different results
    user_details = users.get(name)
    if user_details:
        id, base_path, git_name = user_details['id'], user_details['base_path'], user_details['git_name']
        tasks = get_tasks(id)[:10]
        print(len(tasks))
        commits = get_commits(git_name)

        # Filter tasks and keep only code tasks
        code_tasks = get_code_tasks(tasks, f'{base_path}task_type')

        # Add categories and areas of focus to tasks
        enriched_tasks = enrich_tasks(code_tasks, f'{base_path}detailed_task_type')

        # Add commits to tasks
        tasks_with_commits = correlate_tasks_with_commits(enriched_tasks, commits, base_path)
        tasks_with_commits = return_all_tasks(enriched_tasks, tasks_with_commits, 2) 

        # # plot_people_results(tasks_with_commits)
        input("\nPress Enter to close the graphs...")
    else:
        print(f"No details found for name: {name}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main.py: {e}")
