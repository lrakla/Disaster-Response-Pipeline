#import necessary packages and libraries
import sys
import pandas as pd
import numpy as np
from nltk import pos_tag
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet') # download for lemmatization
import string
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    '''
    This function takes in the database location and splits the data into features(X) i.e messages and
    labels(y) i.e categories
    Input:
    - database_filepath: String <- the location of filepath where the database file is located
    Output : 
    - Input data for ML model X
    - Target/categories to be prediced y
    - categorie names of messages for the ML model
    
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_data', engine)
    df[df.columns[4:]] = df[df.columns[4:]].astype(bool).astype(int)
    for column in df.columns[4:]:
        if df[column].sum() == 0:
           df = df.drop([column], axis=1)   #drop columns with no positive message
    df.insert(loc=len(df.columns), column="unknown_category", value=0)
    df.loc[df[df.columns[4:]].sum(axis=1) == 0, "unknown_category"] = 1
    X = df["message"].values
    y = df[df.columns[4:]]
    category_names = y.columns

    return X,y,category_names


def tokenize(text):
    """
    Tokenizes,normalizes and lemmatizes a given text.
    Input:
        text: text string
    Output:
    -  array of lemmatized and normalized tokens
    """
    def is_noun(tag):
        return tag in ['NN', 'NNS', 'NNP', 'NNPS']


    def is_verb(tag):
        return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


    def is_adverb(tag):
        return tag in ['RB', 'RBR', 'RBS']


    def is_adjective(tag):
        return tag in ['JJ', 'JJR', 'JJS']


    def penn_to_wn(tag):
        if is_adjective(tag):
            return wn.ADJ
        elif is_noun(tag):
            return wn.NOUN
        elif is_adverb(tag):
            return wn.ADV
        elif is_verb(tag):
            return wn.VERB
        return wn.NOUN
    
    tokens = word_tokenize(text.lower()) #split words into tokens and turn thwm into lower case
    tokens = [w for w in tokens if (w not in stopwords.words("english") and w not in string.punctuation)] # remove stopwords and punctuation
    tagged_words = pos_tag(tokens) #tag the tokens
    lemmed = [WordNetLemmatizer().lemmatize(w.lower(), pos=penn_to_wn(tag)) for (w,tag) in tagged_words] #lemmatize the tagged words
    if len(lemmed) == 0: #no lemmatized word should have zero length
        return ["error"]
    return lemmed
    


def build_model():
    """
    Builds the pipeline to transform the input data as natural language into numerical attributes
    to apply machine learning algorithm.
    Output:
    - model
    """
    # pipeline = Pipeline([
    #     ('vect', CountVectorizer(tokenizer=tokenize)),
    #     ('tfidf', TfidfTransformer(sublinear_tf=True)),
    #     ('clf', MultiOutputClassifier(RandomForestClassifier())
    #     ])
    # parameters = {'tfidf__use_idf': (True, False), 
    #          'clf__estimator__n_estimators': [15], #[100,500],
    #         'clf__estimator__min_samples_split': [2, 4]}
    #
    # cv =GridSearchCV(pipeline, param_grid=parameters)
    # return cv
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator = SVC()))
    ])
    parameters = {#'tfidf__use_idf': (True, False),
              'clf__estimator__gamma' : [0.1, 1],
               #“clf__estimator_C”: [100, 200, 500]}
               }
    

    cv = GridSearchCV(pipeline, param_grid=parameters,scoring='f1_weighted', n_jobs=1, verbose=1)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluates the ML model by comparing the predictions of the test
    and prints the classification report (Precision, recall and F1 score)
    cases
    Input:
    - model <- trained ML model
    - X_test <- dataframe of test messages which need to be categorized
    - y_test <- dataframe of actual categories of the messages
    - category_names - list of categories of messages
    Output :
    - Prints classification report
    """
    y_pred = model.predict(X_test)
    print(classification_report (model, X_test, y_test, target_names = category_names))


def save_model(model, model_filepath):
    """
    Saves the ML model to pickle file at specified filepath location.
    Input:
    - model <- sklearn model to be saved as Pickle file
    - model_filepath <- location where the file is to be saved 
    """
    pickle.dump(model,model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 54)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()