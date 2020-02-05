import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''
    input:
        messages_filepath: The path of messages dataset.
        categories_filepath: The path of categories dataset.
    output:
        df: The merged dataset
    '''
    messages = pd.read_csv(messages_filepath)     #load messages data from csv
    categories = pd.read_csv(categories_filepath) #load categories data from csv
    df = pd.merge(messages,categories) #merge categories and messages into one dataset
    return df
    
    


def clean_data(df):
    '''
    Transforms the dataframe containing messages and categories.
    Converts the categories assigned to message in the list format under single column to
    numeric value under individual columns.
    input:
        df: The merged dataset from loading data.
    output:
        df: Dataset after cleaning.
    '''
    categories = df.categories.str.split(pat = ';',expand = True) # create a dataframe of the individual category columns
    row = categories.iloc[0,:]                              # select the first row of the categories dataframe
    category_colnames = row.apply(lambda x : x[:-2]) # use this row to extract a list of new column names for categories.
    categories.columns = category_colnames  #set category column names
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1] 
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    df.drop(columns = 'categories',inplace =True) #remove original categories column
    df = pd.concat([df,categories], axis = 1) # concatenate the original dataframe with the new `categories` dataframe
    df.drop_duplicates(inplace = True)  #drop duplicates
    return df
    
    

def save_data(df, database_filename):
    """
    Saves given dataframe into an table in SQLite database file.
    Input:
    - df: DataFrame <- Pandas DataFrame containing cleaned data of messages and categories
    - database_filename: String <- Location of file where the database file is to be stored    
    """
    
     engine = create_engine('sqlite:///' + database_filename)
     df.to_sql('disaster_data', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()