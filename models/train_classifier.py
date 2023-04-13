"""
This script is a part of the Disaster Response Pipeline project in the Udacity Data Science Nanodegree program. Its purpose is to train a machine learning classifier that can categorize messages related to disasters.

To use the script, you need to provide two arguments when executing it in the command line:

The file path to the SQLite database where the cleaned data is stored. For example, "disaster_response_db.db".
The file path and name where you want to save the trained machine learning model in pickle format. For example, "classifier.pkl".

The script will read the cleaned data from the SQLite database, train a machine learning model, and save the trained model to the specified pickle file.
"""

# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import sys # for system-specific parameters and functions
import os # for interacting with the operating system
import re # for regular expression operations
from sqlalchemy import create_engine # for SQL database connection
import pickle # for saving trained model

from scipy.stats import gmean # for geometric mean calculation
from sklearn.pipeline import Pipeline, FeatureUnion # for creating pipelines
from sklearn.model_selection import train_test_split # for splitting dataset
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, make_scorer # for evaluating model
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier # for classification models
from sklearn.model_selection import GridSearchCV # for hyperparameter tuning
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer # for text feature extraction
from sklearn.multioutput import MultiOutputClassifier # for multi-output classification
from sklearn.base import BaseEstimator,TransformerMixin # for creating custom transformers in pipelines


def load_data_from_db(database_filepath):
    """
    Load data from a SQLite database and return features, labels, and category names.

    Arguments:
        database_filepath (str): The file path to the SQLite database.
        
    Returns:
        X (pandas.DataFrame): A dataframe containing features (messages).
        y (pandas.DataFrame): A dataframe containing labels (categories).
        category_names (list): A list of category names.
    """
    
    # Create a database engine and read the data from the specified table.
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)

    # Drop the 'child_alone' column since it has all zeros only.
    df = df.drop(['child_alone'],axis=1)

    # Replace value 2 in the 'related' field with 1 to consider it a valid response.
    # Alternatively, we could have assumed it to be 0 also. In the absence of information I have gone with majority class.
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)

    # Separate features (messages) and labels (categories) from the loaded data.
    X = df['message']
    y = df.iloc[:,4:]  # The labels start from the 5th column in the loaded dataframe.

    # Save the column names of the label dataframe for later use in visualization.
    category_names = y.columns

    return X, y, category_names


def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Tokenize the text by extracting word tokens and normalizing them by lemmatization.

    Arguments:
        text (str): The text message that needs to be tokenized.
        url_place_holder_string (str): The placeholder string to replace all URLs with.
        
    Returns:
        clean_tokens (list): A list of tokens extracted from the provided text.
    """
    
    # Replace all URLs with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)

    # Lemmatize the word tokens to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens

# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_pipeline():
    """
    Build Pipeline function
    
    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.
        
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline

def multioutput_fscore(y_true,y_pred,beta=1):
    """
    The multioutput_fscore() function is a performance metric used for measuring the performance of a multi-label/multi-class classification problem. It calculates the F1 score, which is the harmonic mean of precision and recall, and then takes the geometric mean of all the calculated F1 scores.

    The function takes three arguments - y_true (list of actual labels), y_pred (list of predicted labels) and beta (value of beta used to calculate F1 score). It can be used as a scorer for GridSearchCV.

    The function first checks if the y_pred and y_true are pandas dataframes, and if so, extracts the values from them. It then calculates the F1 score for each column of the y_true and y_pred arrays, and appends the scores to a list. The F1 scores are calculated using the fbeta_score() function from the scikit-learn library, with the weighted average parameter to account for class imbalances.

    After calculating all the F1 scores, the function removes any score equal to 1, to exclude trivial solutions. It then takes the geometric mean of all the remaining F1 scores, which is the final output of the function.

    The function is designed to deliberately underestimate the standard F1 score average to avoid issues when dealing with multi-class/multi-label imbalanced cases.
    """
    
    # If provided y predictions is a dataframe then extract the values from that
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    
    # If provided y actuals is a dataframe then extract the values from that
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    
    f1score_list = []
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta=beta,average='weighted', zero_division=0)

        f1score_list.append(score)
        
    f1score = np.asarray(f1score_list)
    f1score = f1score[f1score<1]
    
    # Get the geometric mean of f1score
    f1score = gmean(f1score)
    return f1score

def evaluate_pipeline(pipeline, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)
    
    Arguments:
        pipeline -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    Y_pred = pipeline.predict(X_test)
    
    multi_f1 = multioutput_fscore(Y_test,Y_pred, beta = 1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%'.format(multi_f1*100))

    # Print the whole classification report.
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))


def save_model_as_pickle(pipeline, pickle_filepath):
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        pipeline -> GridSearchCV or Scikit Pipelin object
        pickle_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))

def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        print('Loading data from {} ...'.format(database_filepath))
        X, Y, category_names = load_data_from_db(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building the pipeline ...')
        pipeline = build_pipeline()
        
        print('Training the pipeline ...')
        pipeline.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_pipeline(pipeline, X_test, Y_test, category_names)

        print('Saving pipeline to {} ...'.format(pickle_filepath))
        save_model_as_pickle(pipeline, pickle_filepath)

        print('Trained model saved!')

    else:
         print("Please provide the arguments correctly: \n\n"
      "Sample Script Execution:\n"
      "> python train_classifier.py <database_filepath> <model_filepath>\n\n"
      "Arguments Description: \n"
      "1) Path to SQLite destination database file (e.g. data/disaster_response_db.db)\n"
      "2) Path to pickle file where ML model needs to be saved (e.g. models/classifier.pkl)\n")

if __name__ == '__main__':
    main()