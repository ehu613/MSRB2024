#this is the main classifier, takes the txt files obtained from get_text and classifies them using the csv files
#creates a classifier model object using the best parameters found in smaller_nb_model.py and trains on 10,000+docs, save model into a pickle object
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import numpy as np

curr_dir = os.getcwd()
rel_path = 'all_text_files'
folder_path = os.path.join(curr_dir, rel_path)
table = pd.read_csv('acfr_train_combined20240611.csv')
dict = table.set_index('FILE_ID')['LABEL'].to_dict()
file_list = os.listdir(folder_path)

#getting all the labels, this is needed when I was evenly separating the classes
new_labels = []
for file_name in file_list:
    new_labels.append(dict.get(file_name[:-4]))

#only needed for small sampling
# new_df = pd.DataFrame({'Name':file_list, 'Label':new_labels})
# print(new_df)
# subset = new_df.groupby('Label').apply(lambda x: x.sample(min(65, len(x)))).reset_index(drop = True)
# print(subset)

#for small sampling only
# file_contents = []
# for file_name in subset['Name']:
#     file_path = os.path.join(folder_path, file_name)
#
#     with open(file_path, 'r', encoding = "utf8") as file:
#         contents = file.read()
#
#         contents = contents.lower()
#
#     file_contents.append(contents)
#
# small_sample_labels = subset['Label']
# dict_series = pd.Series(dict.values())
# print(dict_series.value_counts(ascending = False))
# small_sample_labels = []
#
#
# for curr_file in small_sample:
#     if curr_file[:-4] in dict:
#         small_sample_labels.append(dict[curr_file[:-4]])

# small_labels_series = pd.Series(small_sample_labels)
# print(small_labels_series.value_counts(ascending = False))
#print out how many i have of each class in small samples labels list

#must use the method of iteration because five files were not copied over from the csv
file_contents = []
for name_of_file in file_list:
    file_path = os.path.join(folder_path, name_of_file)
    with open(file_path, 'r', encoding = "utf8") as file:
        contents = file.read()
        contents = contents.lower()
        file_contents.append(contents)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')

file_contents = [word_tokenize(x) for x in file_contents]

#getting rid of stopwords
no_stop_words = []
for text in file_contents:
    cleaned_text = [word for word in text if word not in stop_words]
    no_stop_words.append(cleaned_text)

import string
no_punct = []
to_be_removed = string.punctuation + '•,“”¨'
for text in no_stop_words:
    cleaned_text = [word for word in text if word not in to_be_removed]
    no_punct.append(cleaned_text)


import re
no_nums = []
for text in no_punct:
    new_string = [token for token in text if not re.search(r'\d', token)]
    no_nums.append(new_string)


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
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
lemmatized = []
for text in no_nums:
   lemmatized.append(lemmatize_tokens(text))

joined = []
for text in lemmatized:
    joining = ' '.join(text)
    joined.append(joining)

# joined = []
# for text in no_nums:
#     joining = ' '.join(text)
#     joined.append(joining)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import string
import random


# Pipeline & Gridsearch setup
# TFIDF pipeline setup
from sklearn.pipeline import Pipeline

# tvc_pipe = Pipeline([
#     ('tvec', TfidfVectorizer()),
# ('mb', MultinomialNB())
# ])


tvc_gs = Pipeline(steps=[('tvec',
                 TfidfVectorizer(max_df=0.6, max_features=2000, min_df=0.15,
                                 ngram_range=(2, 2), stop_words='english')),
                ('mb', MultinomialNB())])

# Setting params for TFIDF Vectorizer gridsearch
# tf_params = {
# 'tvec__max_features':[1000, 2000, 3000],
# 'tvec__stop_words': ['english'],
# 'tvec__min_df': (0.05, 0.10, 0.15),
# 'tvec__max_df': (0.55, 0.60, 0.65),
# 'tvec__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3)]
# }

# Setting up GridSearch for TFIDFVectorizer
# from sklearn.model_selection import GridSearchCV
# tvc_gs = GridSearchCV(tvc_pipe, param_grid=tf_params, cv = 5, verbose =1, n_jobs = -1)


X_train, X_test, y_train, y_test = train_test_split(joined, new_labels, test_size=0.2, random_state=42)

# Fitting TVC GS
tvc_gs.fit(X_train, y_train)

# Scoring Training data on TFIDFVectorizer
train_score = tvc_gs.score(X_train, y_train)
print('Training score: ', train_score)
# Scoring Test data on TFIDFVectorizer
test_score = tvc_gs.score(X_test, y_test)
print('Test score: ', test_score)


# best_estimator = tvc_gs.best_estimator_  # Retrieve the best estimator
# best_estimator.fit(X_train, y_train)

# predictions = tvc_gs.predict(X_test)  # Make predictions on test data


# X_train = vectorizer.fit_transform(X_train)
# X_test = vectorizer.transform(X_test)
#
# #Train the classifier
# classifier = MultinomialNB()
# classifier.fit(X_train, y_train)

y_pred = tvc_gs.predict(X_test)
y_probs = tvc_gs.predict_proba(X_test)
prob_predictions = pd.DataFrame(data = y_probs, columns = tvc_gs.classes_)

# best_pipeline = tvc_gs.best_estimator_
best_tfidf = tvc_gs.named_steps['tvec']
classifier = tvc_gs.named_steps['mb']

import pickle
pickle.dump(tvc_gs , open('final-model-1.1.pk1' , 'wb'))

#this represents the probabilities that a document is in each class, i.e. 5% class A
print(prob_predictions)
#Evaluate the classifier
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict = True)
df = pd.DataFrame(report).transpose()
df.to_csv('classification_report_v1.1.csv')


train_df = pd.DataFrame({'X_train': X_train, 'y_train': y_train})
train_df.to_csv('train_cleaned_data_v1.1.csv')
test_df = pd.DataFrame({'X_test': X_test, 'y_test': y_test})
test_df.to_csv('test_cleaned_data_v1.1.csv')


# Get the feature names
feature_names = best_tfidf.get_feature_names_out()

#frequency of tokens per class
class_a_token_count = classifier.feature_count_[0]
class_b_token_count = classifier.feature_count_[1]
class_c_token_count = classifier.feature_count_[2]
class_d_token_count = classifier.feature_count_[3]
class_e_token_count = classifier.feature_count_[4]
class_f_token_count = classifier.feature_count_[5]

tokens = pd.DataFrame(
    {'token': feature_names, 'Class_A': class_a_token_count, 'Class_B': class_b_token_count, 'Class_C': class_c_token_count, 'Class_D': class_d_token_count, 'Class_E': class_e_token_count, 'Class_F': class_f_token_count}).set_index('token')

# add 1 to Class 1 and Class 2 counts to avoid dividing by 0
tokens['Class_A'] = tokens.Class_A + 1
tokens['Class_B'] = tokens.Class_B + 1
tokens['Class_C'] = tokens.Class_C + 1
tokens['Class_D'] = tokens.Class_D + 1
tokens['Class_E'] = tokens.Class_E + 1
tokens['Class_F'] = tokens.Class_F + 1

#convert to frequencies
tokens['Class_A frequencies'] = tokens.Class_A / classifier.class_count_[0]
tokens['Class_B frequencies'] = tokens.Class_B / classifier.class_count_[1]
tokens['Class_C frequencies'] = tokens.Class_C / classifier.class_count_[2]
tokens['Class_D frequencies'] = tokens.Class_D / classifier.class_count_[3]
tokens['Class_E frequencies'] = tokens.Class_E / classifier.class_count_[4]
tokens['Class_F frequencies'] = tokens.Class_F / classifier.class_count_[5]

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

tokens = tokens.T
freq = tokens.iloc[-6:]

#add frequences for each of top five terms, word cloud other visualization..
most_freq_class_A = freq.iloc[0, :].sort_values(ascending=False)
print('Most frequent word in class A: ')
print(most_freq_class_A[0:20])

most_freq_class_B = freq.iloc[1, :].sort_values(ascending=False)
print('Most frequent word in class B: ')
print(most_freq_class_B[0:20])

most_freq_class_C = freq.iloc[2, :].sort_values(ascending=False)
print('Most frequent word in class C: ')
print(most_freq_class_C[0:20])

most_freq_class_D = freq.iloc[3, :].sort_values(ascending=False)
print('Most frequent word in class D: ')
print(most_freq_class_D[0:20])

most_freq_class_E = freq.iloc[4, :].sort_values(ascending=False)
print('Most frequent word in class E: ')
print(most_freq_class_E[0:20])

most_freq_class_F = freq.iloc[5, :].sort_values(ascending=False)
print('Most frequent word in class F: ')
print(most_freq_class_F[0:20])

print(best_tfidf)

#donut plot for one of the documents, better part of this is in results.py
import matplotlib.pyplot as plt

import matplotlib
import numpy as np
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())

# print("50th results: ")
# print(prob_predictions.iloc[50])
#
# size_of_groups=prob_predictions.iloc[50].tolist()
#
# colors = ['#FF0000', '#0000FF', '#FFFF00',
# 		'#ADFF2F', '#FFA500', 'E6E6FA']
#
# # explosion
# explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
#
# # Pie Chart
# plt.pie(size_of_groups, colors=colors, labels=tvc_gs.classes_,
# 		autopct='%1.1f%%', pctdistance=0.85,
# 		explode=explode)
#
# my_circle=plt.Circle( (0,0), 0.7, color='white')
# p=plt.gcf()
# # Adding Circle in Pie chart
# p.gca().add_artist(my_circle)
# plt.title('Prediction per class')
# plt.show()


#confusion matrix heat map

from sklearn.metrics import confusion_matrix
import seaborn as sns

class_names = tvc_gs.classes_
cf_matrix = confusion_matrix(y_test, y_pred)

sns.set(font_scale=1.4) # for label size
sns.heatmap(cf_matrix, annot=True, annot_kws={"size": 16}, xticklabels=class_names, yticklabels=class_names) # font size
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('final-heat-map.png')
plt.show()


#wordcloud test
from wordcloud import WordCloud

def make_wordcloud(data, class_name):

    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate_from_frequencies(data)
    # Display the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(class_name)
    plt.savefig(class_name+'.png')
    plt.show()

make_wordcloud(most_freq_class_A[0:20], 'Class_A')
make_wordcloud(most_freq_class_B[0:20], 'Class_B')
make_wordcloud(most_freq_class_C[0:20], 'Class_C')
make_wordcloud(most_freq_class_D[0:20], 'Class_D')
make_wordcloud(most_freq_class_E[0:20], 'Class_E')
make_wordcloud(most_freq_class_F[0:20], 'Class_F')