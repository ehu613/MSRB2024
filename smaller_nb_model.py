#This is gridsearch done on 2,000+ documents because gridsearch on all 10,000+ documents was unsuccessful due to insufficient RAM
#displays the best set of parameters, which then can be used by nb_classifier.py to train the 10,000+documents
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import pandas as pd

import os
curr_dir = os.getcwd()

rel_path = 'all_text_files'
folder_path = os.path.join(curr_dir, rel_path)
print(folder_path)
table = pd.read_csv('acfr_train_combined20240611.csv')
print(table['LABEL'].value_counts())

dict = table.set_index('FILE_ID')['LABEL'].to_dict()
file_list = os.listdir(folder_path)

#must create this in order to choose a balanced set of entries from each label
new_labels = []
for file_name in file_list:
    new_labels.append(dict.get(file_name[:-4]))

import random

new_df = pd.DataFrame({'Name':file_list, 'Label':new_labels})
subset = new_df.groupby('Label').apply(lambda x: x.sample(min(500, len(x)))).reset_index(drop = True)
print(subset['Label'].value_counts())

file_contents = []
for file_name in subset['Name']:
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'r', encoding = "utf8") as file:
        contents = file.read()
        contents = contents.lower()
    file_contents.append(contents)

small_sample_labels = subset['Label']

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')

file_contents = [word_tokenize(x) for x in file_contents]

no_stop_words = []
#for every indiviual text file
for text in file_contents:
    cleaned_text = []
    for word in text:
        if word not in stop_words:
            cleaned_text.append(word)
    no_stop_words.append(cleaned_text)

import string

no_punct = []
to_be_removed = string.punctuation + '•,“”¨'
for text in no_stop_words:

    cleaned_text = []
    for word in text:
        if word not in to_be_removed:
            cleaned_text.append(word)
    no_punct.append(cleaned_text)

import re
no_nums = []
for text in no_punct:
    new_string = [token for token in text if not re.search(r'\d', token)]
    no_nums.append(new_string)

import numpy as np
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
# vectorizing function to able to call on list of tokens
lemmatize_words = np.vectorize(lemmatizer.lemmatize)

nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
import pandas as pd

# define function to lemmatize tokens
def lemmatize_tokens(tokens):
    # convert POS tag to WordNet format
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    # lemmatize tokens
    lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]

    return lemmas


nltk.download('omw-1.4')
# apply lemmatization function to column of dataframe
lemmatized = []
for text in no_nums:
   lemmatized.append(lemmatize_tokens(text))

joined = []
for text in lemmatized:
    joining = ' '.join(text)
    joined.append(joining)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import string
import random


# Pipeline & Gridsearch setup
# TFIDF pipeline setup
from sklearn.pipeline import Pipeline

tvc_pipe = Pipeline([
    ('tvec', TfidfVectorizer()),
('mb', MultinomialNB())
])

# Setting params for TFIDF Vectorizer gridsearch
tf_params = {
'tvec__max_features':[1000, 2000, 3000],
'tvec__stop_words': ['english'],
'tvec__min_df': (0.10, 0.15),
'tvec__max_df': (0.55, 0.60, 0.65),
'tvec__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3)]
}

# Setting up GridSearch for TFIDFVectorizer
from sklearn.model_selection import GridSearchCV
tvc_gs = GridSearchCV(tvc_pipe, param_grid=tf_params, cv = 5, verbose =1, n_jobs = -1)

X_train, X_test, y_train, y_test = train_test_split(joined, small_sample_labels, test_size=0.2, random_state=42)

# Fitting TVC GS
tvc_gs.fit(X_train, y_train)

# Scoring Training data on TFIDFVectorizer
train_score = tvc_gs.score(X_train, y_train)
print('Training score: ', train_score)
# Scoring Test data on TFIDFVectorizer
test_score = tvc_gs.score(X_test, y_test)
print('Test score: ', test_score)

y_pred = tvc_gs.predict(X_test)
y_probs = tvc_gs.predict_proba(X_test)
prob_predictions = pd.DataFrame(data = y_probs, columns = tvc_gs.classes_)


best_pipeline = tvc_gs.best_estimator_
print(best_pipeline)

# Extract the TfidfVectorizer
best_tfidf = best_pipeline.named_steps['tvec']
classifier = best_pipeline.named_steps['mb']

import pickle
pickle.dump(tvc_gs, open('final_smaller_nb_model_v1.1.pk1', 'wb'))
pickle.dump(best_tfidf , open('vectorizer-classifying-audited-financials.pk1' , 'wb'))
pickle.dump(classifier , open('classifier-classifying-audited-financials.pk1' , 'wb'))

#this represents the probabilities that a document is in each class, i.e. 5% class A
print(prob_predictions)

#Evaluate the classifier
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))