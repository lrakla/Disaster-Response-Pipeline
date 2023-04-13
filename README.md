# Disaster-Response-Pipeline
This is a NLP project which categorizes tweets received during disaster time period using data provided by Figure Eight using
supervised machine learning. The end result of the project is a web dashboard which can categorize new messages.

# Motivation
In real life disasters, millions of text messages and tweets are received either directly or via social media right at the time when the disaster response organizations have the least capacity to filter and then pull out the messages which are the most important. There might be one out of every thousand message that might be relevant to the disaster response professionals.

At the time of disaster, different organizations take care of different parts of the problem. For e.g. one organization may care about water, another would take care about blocked roads and some other would take care of medical supplies etc. In the dataset, these kind of categories are pulled out from these datasets. The datasets from different disasters have been combined and consistently labeled with categories with help of Figure Eight, the human and machine learning enabled data annotation service.

Talking about the nature of the messages and the category assigned to them, it is possible that the exact word that describes the category for e.g. 'water' may not be present in the text of the message and that category still needs to be inferred based on the presence of other relevant words and phrases for e.g. 'thirsty'. Therefore, a simple keyword matching may not be effective way to categorize the messages. Supervised machine learning model can be more helpful to tackle this challenge effectively.

# Approach
1. Firstly, an ETL pipeline is run which combines the messages and categories dataset, removes duplicates, turns categories into
separate columns and stores the clean data in an SQLite database.
2. Secondly, a ML pipeline uses the cleaned data. It tokenizes the messages and various Multi-Classification algorithms are run on this 
data. The parameters are selected using GridSearch. The whole procedure is an ML pipeline. The output is a classification report 
of metrics like F1 score, precision, recall of various individual categories. The trained model is stored as a pickle file.
3. The web application uses the trained model and provides a dashboard to categorize new messages. It provides visualisations of the training data as well.
![Web application](https://github.com/lrakla/Disaster-Response-Pipeline/blob/master/Web%20application%20dahboard.jpg?raw=True "Web application")
![Message categories](https://github.com/lrakla/Disaster-Response-Pipeline/blob/master/Bar%20plot%20of%20message%20categories.jpg?raw=True "Message Categories")
![Message Genres](https://github.com/lrakla/Disaster-Response-Pipeline/blob/master/Bar%20plot%20of%20message%20genres.jpg?raw=True "Message genres")

# Important :
The different message categories are not equal in number. For e.g, cold, fire, earthquake have only ~100 to 200 messages whereas offer, related, request have messages > 5000. It is important to focus on recall parameter for the categories which are few in number because
recall provides an indication of missed positive predictions which in this case are more important for aid agencies. Recall should be as close to 1.

# Instructions to run project
1. The project depends on several libraries including plotly, pandas, nltk, flask, sklearn, sqlalchemy, numpy, re, pickle.

2. Run the following commands in the project's root directory to set up SQLite database and supervised machine learning model to categorize messages.
 - To run ETL pipeline that cleans data and stores in database ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
 - To run ML pipeline that trains classifier and saves ```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```
 - Run the following command in the app's directory to run your web app. ```python run.py```

Open http://0.0.0.0:3001/ in browser

# Future scope of project
In the coming times I am trying to implement the following in the project :
1. Include more vidualisations in the web app using Altair's visualisation library.
2. Add relevant agencies in the app based on classification of messages
3. Deploy the web app to a cloud service provider.

