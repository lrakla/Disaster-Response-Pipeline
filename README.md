# Disaster-Response-Pipeline
A ML pipeline project which categorizes tweets received during disaster time period using data provided by Figure Eight. 

# Project Components

1. ETL Pipeline
A Python script, process_data.py which 

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
A Python script, train_classifier.py, that :

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App

A simple web application which can categorize new messages.
