import copy
from .api import count_tokens, load_or_request_data

# Message to categorize the tasks into code-related and non-code-related
# REMEMBER: Don't change the query tabbing, because it will change the results 
def categorize_task_query(task_titles):
    return [  
        {
            "role": "system",
            "content": "You are an all-knowing product owner. Your goal is to accurately categorize task titles related to software development."
        }, 
        {
        "role": "user",
        "content": f"""
        I will give you several task titles related to software development. You will categorize each title as "Code-Related" (requiring code commits) or "Non-Code-Related" (solved without code commits). Please do not explain your reasoning and do not include any additional text. Only provide the categorizations in the specified format.

        Format the output as shown below:

        Example if there are 4 titles:
        {{
        "Name of the 1st title": "Code-Related",
        "Name of the 2nd title": "Code-Related",
        "Name of the 3rd title": "Code-Related",
        "Name of the 4th title": "Non-Code-Related"
        }}

        The titles are:
        {task_titles}
        """
        }
    ]

# Message to enhance tasks with their categories and area of focus
def enhance_task_query(code_task_titles):
    return [
        {
            "role": "system",
            "content": "You are an all-knowing product owner. Your goal is to accurately categorize task titles related to software development."
        },
        {
        "role": "user",
        "content": f"""
        I will give you several task titles related to software development. You will categorize each title based on the provided categories and areas of focus. Each task title should be associated with one or more categories and one or more areas of focus. Try not to return any false positives. Please do not explain your reasoning and do not include any additional text. Only provide the categorizations in the specified format.

        Categories:
        1. Bug Fixes
        2. Testing & Code Review
        3. Optimization
        4. Feature
        5. Code Refactoring
        6. Dependencies
        7. Documentation & General

        Areas of Focus:
        1. Frontend
        2. Backend
        3. DevOps & Cloud
        4. Database
        5. Security
        6. AI
        7. Embedded

        Format the output as shown below and property name enclosed in double quotes:

        Example if there are 2 titles:
        {{
        "Name of the 1st title": {{ "Categories": ["Testing"], "FocusArea": ["Frontend"]}},
        "Name of the 2nd title": {{ "Categories": ["Testing"], "FocusArea": ["Backend"]}},
        }}

        The titles are:
        {code_task_titles}
        """
        }
    ]

def categorize_commits_query(tasks):
    return [
        {
            "role": "system",
            "content": "You are an all-knowing developer. Your goal is to accurately determine if commits belong to a specific task."
        },
        {
            "role": "user",
            "content": f"""
            I will give you a task with a title, keywords, and several commits. You will return the task title along with the _ids of commits that belong to the task. Only provide the result in the specified format without any additional text.

            Consider the following when determining if a commit belongs to a task:
            1. **File Paths**: Look at the commit's file paths (commit.files.filename). For example, if a task is about "fix on php", prioritize filenames such as "php/analyzer/analyze.js".
            2. **Keywords Matching**: Use the task title and keywords to match with the commit messages and file paths. Strong matches are crucial.
            3. **Commit Messages**: Check if the commit message contains keywords from the task title or description.
            4. **Strict Criteria**: Return only the commits that strongly match the task title, keywords, and file paths. Discard any commits with weak or ambiguous matches.

            Example:
            Task Title: "Fix PHP Analyzer Bug"
            Keywords: ["fix", "PHP", "analyzer", "bug"]
            Commits:
            - Commit 1: Message: "Fixed bug in PHP analyzer", Files: ["php/analyzer/analyze.js"]
            - Commit 2: Message: "Updated README", Files: ["README.md"]
            - Commit 3: Message: "Refactored PHP analyzer code", Files: ["php/analyzer/refactor.js"]

            In this case, only Commit 1 and Commit 3 belong to the task.

            Format the output as shown below:

            Example if there are commits that belong:
            ```
            [
                {{
                    "title": "Task Title",
                    "commit_ids": [
                        "commit_id_1",
                        "commit_id_2"
                    ]
                }}
            ]
            ```

            If no commits belong:
            ```
            [
                {{
                    "title": "Task Title",
                    "commit_ids": []
                }}
            ]
            ```

            The tasks are:
            {tasks}
            """
        }
    ]

# Break queries if they exceed a token count, if nothing can be done return empty
def process_queries(items, query_function, path, should_write_file=False, max_tokens=8192):
    def process_with_max_tokens(max_tokens):
        all_queries = []
        current_items = []

        def format_files_for_categorize(commits):
            for commit in commits:
                commit["files"] = [
                    f"+{file['additions']} -{file['deletions']} {file['filename']}" 
                    for file in commit["files"]
                ]
            return commits

        for item in items:
            temp_items = current_items + [item]
            temp_query = query_function(temp_items)
            temp_token_count = sum(count_tokens(query_item['content']) for query_item in temp_query)

            if temp_token_count > max_tokens:
                if current_items:
                    all_queries.append(query_function(current_items))
                current_items = [item]

                # Apply format if tokens exceeded and make commits be (+ additions - deletions filename)
                if query_function == categorize_commits_query:
                    item_copy = copy.deepcopy(item)
                    item_copy['commits'] = format_files_for_categorize(item_copy['commits'])
                    current_items = [item_copy]
            else:
                current_items.append(item)

        if current_items:
            all_queries.append(query_function(current_items))

        return all_queries

    def retry_with_different_tokens(initial_tokens):
        for tokens in initial_tokens:
            all_queries = process_with_max_tokens(tokens)
            if all_queries and not all(len(query) == 0 for query in all_queries):
                return all_queries
            
            if len(initial_tokens) == 2:
                print(f"Retrying with max tokens: {tokens}")
        return []
    
    # Set initial max_tokens based on query_function
    if query_function == enhance_task_query:
        initial_tokens = [600, max_tokens]
    elif query_function == categorize_task_query:
        initial_tokens = [700, max_tokens]
    else:
        initial_tokens = [max_tokens]
    
    all_queries = retry_with_different_tokens(initial_tokens)
    llm_output = load_or_request_data(all_queries, path, should_write_file)

    return llm_output if isinstance(llm_output, list) else [{}]
