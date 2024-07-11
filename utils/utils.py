from datetime import datetime, timedelta
import json
import re
from collections import Counter
import spacy
import os
nlp = spacy.load('en_core_web_sm')

def load_json_file(filename):
    path = '' # Place the main folder of your data
    with open(f'{path}{filename}', 'r', encoding='utf8') as file:
        return json.load(file)

# Function that returns the commits of an author. You can choose an author to get only his commits
def get_commits(author=''):
    commits = load_json_file('') # Place the data file in that main folder for commits
    if author:
        return [commit for commit in commits if commit.get('author') == author]
    return commits

# Function that returns the tasks of an author. You can choose an author to get only his tasks
# Remove statusEdits = []
def get_tasks(assignee_id=''):
    tasks = load_json_file('') # Place the data file in that main folder for tasks
    
    cleaned_tasks = [task for task in tasks if task.get('statusEdits')]

    return [task for task in cleaned_tasks if assignee_id in task.get('assignees', [])] if assignee_id else cleaned_tasks

def preprocess_text(text, max_words=20):
    """Clean and prepare text and get the max_words most common words. Lowercase, remove mentions, URLs, 
    non-alphanumeric characters (except for spaces), stopwords, and normalizing whitespace."""
    CLEAN_TEXT = re.compile(r'@\w+|https?://\S+|[^\w\s]|_')
    cleaned_text = CLEAN_TEXT.sub(' ', text.lower()).strip()
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    
    # Use spaCy for tokenization and stopword removal
    doc = nlp(cleaned_text)
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    
    # Extract the most common words
    word_counts = Counter(tokens)
    most_common_keywords = [word for word, count in word_counts.most_common(max_words)]
    
    return ' '.join(most_common_keywords)

# Find the important dates of tasks to look for commits, have as start_data -> createdAt and the last time if last statusEdits.to is accepted, take the previous one, else take the last
# NOTE: CreatedAt shouldnt have an initial entry, first the task is created and then we search for commits
def extract_dates(tasks):
    def get_date_variations(date_str):
        date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        return [(date - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"), (date + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")]

    # Get createdAt for a specific key
    def extract_from_nested(task, key):
        FLOW = ["Backlog", "Sprint Planning", "In Progress", "Delivered", "Accepted"]
        is_valid_transition = lambda from_status, to_status: FLOW.index(from_status) < FLOW.index(to_status) if from_status in FLOW and to_status in FLOW else False

        def filter_valid_status_edits(status_edits):
            valid_items, last_status = [], None
            for item in status_edits:
                current_status = item.get('to')
                if last_status is None or is_valid_transition(last_status, current_status):
                    valid_items.append(item)
                    last_status = current_status
            return valid_items

        items = task.get(key, [])
        if key == 'statusEdits':
            items = filter_valid_status_edits(items)

        return [get_date_variations(item['createdAt']) for item in items if 'createdAt' in item]

    def get_end_date(task):
        status_edits = task['statusEdits']
        last_edit = status_edits[-1]
        end_date_str = (status_edits[-2]['createdAt'] if last_edit['to'] == "Accepted" and len(status_edits) > 1 else last_edit['createdAt'])
        return datetime.strptime(end_date_str, "%Y-%m-%dT%H:%M:%S.%fZ")

    # Merge overlapping time ranges
    def merge_overlapping_ranges(ranges):
        if not ranges:
            return []

        parse_date = lambda date_str: datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        ranges.sort(key=lambda x: parse_date(x[0]))
        merged_ranges = [ranges[0]]

        for start, end in ranges[1:]:
            last_end = merged_ranges[-1][1]
            if parse_date(start) <= parse_date(last_end):
                merged_ranges[-1][1] = max(last_end, end, key=parse_date)
            else:
                merged_ranges.append([start, end])

        return merged_ranges

    # Get only the ranges inside start and end date
    def filter_and_adjust_ranges(start_date, end_date, date_ranges):
        parse_date = lambda date_str: datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        format_date = lambda date: date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        filtered_ranges = []
        for start, end in date_ranges:
            start_of_range = parse_date(start)
            end_of_range = parse_date(end)
            
            if start_of_range < start_date < end_of_range:
                filtered_ranges.append([format_date(start_date), end])
            elif start_date <= start_of_range <= end_date:
                filtered_ranges.append([start, end])

        return filtered_ranges

    all_dates = []
    for task in tasks:
        start_date = datetime.strptime(task['createdAt'], "%Y-%m-%dT%H:%M:%S.%fZ")
        end_date = get_end_date(task) + timedelta(hours=3)

        date_ranges = []
        for key in ['pointsBurnedEdits', 'statusEdits']:
            date_ranges.extend(extract_from_nested(task, key))

        filtered_date_ranges = filter_and_adjust_ranges(start_date, end_date, date_ranges)
        comments_text = " ".join(comment['body'] for comment in task.get('comments', []) if 'body' in comment)

        all_dates.append({
            'id': task.get('_id'), 
            'date_ranges': merge_overlapping_ranges(filtered_date_ranges),
            'keywords': preprocess_text(f"{task.get('body', '')} {comments_text}"),
            'title': task.get('title', ''),
        })

    return all_dates

# Get the important dates of a task, find the commits of those dates, filter task data and filter tasks where len(commits) > 0
def add_commits_to_tasks(code_tasks, commits, file_name):
    def filter_task_data(results):
        extracted_data = []
        for result in results:
            filtered_result = {
                "keywords": result["keywords"],
                "title": result["title"],
                "commits": [
                    {   
                        "commitId": commit['_id'],
                        "message": preprocess_text(commit["message"]),
                        "files": [{key: value for key, value in file.items() if key != "_id"} for file in commit["files"]]
                    }
                    for commit in result["commits"]
                ]
            }
            extracted_data.append(filtered_result)
        return extracted_data

    def is_commit_in_date_range(commit_date_str, date_ranges):
        commit_date = datetime.strptime(commit_date_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        for date_range in date_ranges:
            start_date = datetime.strptime(date_range[0], '%Y-%m-%dT%H:%M:%S.%fZ')
            end_date = datetime.strptime(date_range[1], '%Y-%m-%dT%H:%M:%S.%fZ')
            if start_date <= commit_date <= end_date:
                return True
        return False

    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            tasks_with_commits = json.load(file)
    else:
        try:
            task_important_dates = extract_dates(code_tasks)
        except Exception as e:
            print(f"Error in extract_dates: {e}")

        for task in task_important_dates:
            task['commits'] = []
            for commit in commits:
                if is_commit_in_date_range(commit['createdAt'], task['date_ranges']):
                    task['commits'].append(commit)
                    
        enhanced_tasks = filter_task_data(task_important_dates)

        tasks_with_commits = [task for task in enhanced_tasks if len(task['commits']) > 0]

        with open(file_name, 'w') as file:
            json.dump(tasks_with_commits, file)
    return tasks_with_commits

def filter_enhanced_tasks(tasks):
    # Allowed categories and focus areas
    allowed_categories = [
        "Bug Fixes",
        "Testing & Code Review",
        "Optimization",
        "Feature",
        "Code Refactoring",
        "Dependencies",
        "Documentation & General"
    ]

    allowed_focus_areas = [
        "Frontend",
        "Backend",
        "DevOps & Cloud",
        "Database",
        "Security",
        "AI",
        "Embedded"
    ]

    # Filter tasks
    filtered_tasks = [
        task for task in tasks
        if all(category in allowed_categories for task_name, details in task.items() for category in details["Categories"]) and
           all(area in allowed_focus_areas for task_name, details in task.items() for area in details["FocusArea"])
    ]


    return filtered_tasks
