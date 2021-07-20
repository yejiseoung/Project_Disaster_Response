import sys
import pandas as pd
from sqlalchemy import create_engine

def category_to_numeric(categories: pd.DataFrame) -> pd.DataFrame:
    """ Function to change categorical data to numeric.
    Args (DataFrame): categories
    Returns: DataFrame
    """

    for col in categories:
        categories[col] = categories[col].map(
            lambda x: 1 if int(x.split("-")[1]) > 0 else 0
        )
    return categories


def split_values(categories: pd.DataFrame) -> pd.DataFrame:
    """ Function to split right values (0, 1) from categorical values
    Args (DataFrame): categories
    Returns: DataFrame
    """
    categories = categories['categories'].str.split(';', expand=True)
    row = categories.iloc[[1]].values[0]
    categories.columns = [x.split("-")[0] for x in row]
    categories = category_to_numeric(categories)
    return categories


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """ Function to load 2 datasets (categories and messages)
    Args: messages_filepath: Path to the messages.csv file
          categories_filepath: PAth to the categories.csv file

    Returns: DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = split_values(pd.read_csv(categories_filepath))

    return pd.concat([messages, categories], join='inner', axis=1)
    

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Function to clean data
    Args: DataFrame
    Returns: DataFrame
    """
    return df.drop_duplicates()
    

def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """ Function to save the data in sql file
    Args: df: DataFrame
          database_filename: Path to save sql database
          
    Returns: None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster', engine, if_exists='replace', index=False)



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n      CATEGORIES: {}'.format(
            messages_filepath, categories_filepath))
        
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n     DATABASE: {}'.format(database_filepath))
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