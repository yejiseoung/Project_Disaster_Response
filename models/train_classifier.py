import sys

def load_data(database_filepath, database_filename):
    """ Function to load the database into pandas DataFrame
    Args: database_filepath: Path for the database
          database_filename: Name for the database
    Returns: X: features (messages)
             y: categories
             An ordered list of categories
    """
    # Loading database into pandas DataFrame
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM {}".format(database_filename), engine)

    # Creating DataFrame for x variables
    X = df['message']

    # Creating DataFrame for y variables
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1).astype(float)
    categories = y.columns.values
    return X, y, categories 


def tokenize(text):
    """ Function to tokenize text
    Args: Text
    Returns: List of tokens
    """
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w).lower().strip() for w in tokens]


def build_model()->GridSearchCV:
    """ Function to build pipeline and GridSearch
    Args: None
    Returns: Model
    """
    # Pipeline for transforming data, fitting to model and predict the model
    pipeline = Pipeline([
        ('cvect': CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])

    # Parameters for GridSearch 
    parameters = {
        'clf__n_estimators': [20, 40, 60],
        'clf__max_depth': [5, 10, None],
        'clf__max_samples_leaf': [2, 4, 5],
        'clf__max_samples_split': [2, 5, 10],
    }

    # GridSearch with parameters above
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring='f1_micro', verbose=1, n_jobs=1)

    return pipeline


def evaluate_model(model: GridSearchCV, X_test: pd.DataFrame, y_test: pd.DataFrame, category_names: List)->None:
    """ Function to evaluate model by printing a classification report

    Args: model, features, labels to evaluate, and a list of categories
    Returns: Classification report
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))    
    print('--------------------------------------------------')
    print('\n', classification_report(y_train.values, y_pred_train, target_names=y.columns.values))



def save_model(model: GridSearchCV, model_filepath: str)-> None:
    """ Function to save the model as pickle file
    Args: Model, filepath
    Returns: None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = building_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()