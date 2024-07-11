# Design and Development of a Methodology for Estimating personalized Task Duration

## General Info
In an age of rapid technology development, increasing market demands create the need for fast and qualitative software delivery. This is a major issue for software project managers, who are responsible for leading their teams to success.

In the context of this diploma thesis, a mechanism is designed and developed for calculating the delivery time of tasks assigned to team members. By leveraging task completion data for each team, features can be extracted to estimate the completion time of new tasks by the same or new team members. Additionally, this information helps create a mechanism to optimize the distribution of these tasks, aiming to reduce the overall software delivery time by assigning them to the most appropriate team members.

## Installation 
```

pip install -r requirements.txt
npm install

```


## Dataset Creation
Data was collected through the Cyclopt platform. This data includes task completion times, team member contributions, and task details. The initial dataset is created by processing the raw data, extracting relevant features, and correlating the commits of team members with their respective tasks. The enriched dataset serves as the foundation for training a machine learning model. If you want to extract that data from the platform you can execute the following command:

``` node path/to/script1.js ```


### Preprocessing
To preprocess the data, the following steps are performed:

* Data Cleaning: Removing any incomplete or irrelevant data entries.
* Feature Extraction: Extracting features such as task complexity, member experience, and previous task completion times.
* Dataset Enrichment: Utilizing large language models to correlate team member commits with tasks, adding more context to the dataset.

Preprocessing is handled by executing the script:

``` python path/to/script1.py ```


## Model Training
The preprocessed dataset is used to train a machine learning model that predicts task completion times. The training process involves:

* Model Selection: Choosing appropriate machine learning algorithms. Originally MATLAB, Classification Learner App was used and after that we focused on the 4 best performing models. 
* Training: Training the model on the dataset.
* Validation: Validating the model's performance.

## Task Distribution Optimization
An algorithm is developed to optimize task distribution among team members. This algorithm considers factors such as team member expertise, task complexity, and current workload to assign tasks in a way that minimizes overall delivery time.

Model training and the optimization process is executed with the script:

``` python path/to/script2.py ```

## Conclusion
This thesis presents a comprehensive mechanism for predicting and optimizing task delivery times in software development teams. By leveraging machine learning and data-driven insights, the proposed system aims to enhance productivity and streamline project management.


### P.S.
Get an api key for llama and add it on env.

Save the results of script1.js and script1.py. The output from script1.js can be used as the input data for script1.py, and the output from script1.py can then be used for script2.py.

To run the scripts:

* script1.js: Ensure you have access to the Cyclopt database.
* script1.py: Follow the comments within the script and update the paths accordingly.
* script2.py: Similar to script1.py, update the paths as needed. Additionally, to view the results of task distribution, use a smaller subset of tasks, such as 10-20 tasks.