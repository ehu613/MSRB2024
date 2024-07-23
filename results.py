import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#
curr_dir = os.getcwd()
# rel_path = 'new files'
rel_path = 'all_text_files'
folder_path = os.path.join(curr_dir, rel_path)
#
# file_list = os.listdir(folder_path)
table = pd.read_csv('file_sample_for_NB_validation(in).csv')
table_file_names = table['FILE_ID']
table_file_labels = table['Summary Category Letter']

file_contents = []

for file_name in table_file_names:
    path_to_txt = os.path.join(folder_path, file_name+'.txt')

    with open(path_to_txt, 'r', encoding="utf8") as file:
        contents = file.read()
        contents = contents.lower()
        file_contents.append(contents)

print(len(file_contents))

#
#
# def extract_text(textDirectory, fileID, pdf_path):
#
#     import pdfplumber
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#
#             all_pages = []
#             for page in pdf.pages:
#                 page_text = page.extract_text()
#                 all_pages.append(page_text)
#             # text = '\n'.join([page.extract_text() for page in pdf.pages])
#
#             text = '\n'.join(all_pages)
#
#         write_to_this_path = str(textDirectory + '\\' + fileID[:-4] +  '.txt')
#         with open(write_to_this_path, 'w', encoding = "utf-8") as file:
#             # Write the text to the file
#             file.write(text)
#     except:
#         print('PDF could not be processed'+ fileID[:-4])
#
#
#
#
# for file_name in file_list:
#     pdf_file_path = os.path.join(folder_path, file_name)
#     extract_text(folder_path, file_name, pdf_file_path)
#     os.remove(pdf_file_path)
#
#
# txt_list = os.listdir(folder_path)
# print('txt list: ')
# print(txt_list)
#
# file_contents = []
# for txt_name in txt_list:
#     print('hi')
#     print('Reading in: '+ txt_name)
#     file_path = os.path.join(folder_path, txt_name)
# #maybe this line below is the reason it keeps deleting
#     # os.mkdir(file_path)
#
    # with open(file_path, 'r', encoding = "utf8") as file:
    #     contents = file.read()
    #     contents = contents.lower()
    #     file_contents.append(contents)
#
# print("conetens")
# print(file_contents[0])
#
# #
# # curr_dir = os.getcwd()
# # rel_path = 'all_text_files'
# # folder_path = os.path.join(curr_dir, rel_path)
# # table = pd.read_csv('acfr_train_combined20240611.csv')
# #
# # dict = table.set_index('FILE_ID')['LABEL'].to_dict()
# # file_list = os.listdir(folder_path)
# #
# # new_labels = []
# # for file_name in file_list:
# #     new_labels.append(dict.get(file_name[:-4]))
# #
# # new_df = pd.DataFrame({'Name':file_list, 'Label':new_labels})
# # print(new_df)
# # subset = new_df.groupby('Label').apply(lambda x: x.sample(min(65, len(x)))).reset_index(drop = True)
# # print(subset)
# #
# # file_contents = []
# # for file_name in subset['Name']:
# #     file_path = os.path.join(folder_path, file_name)
# #
# #     with open(file_path, 'r', encoding = "utf8") as file:
# #         contents = file.read()
# #         contents = contents.lower()
# #
# #     file_contents.append(contents)
# #
# # small_sample_labels = subset['Label']
#
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')

file_contents = [word_tokenize(x) for x in file_contents]


no_stop_words = []
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
    # [word1, word2, word3, ...]
    for word in text:
        if word not in to_be_removed:
            cleaned_text.append(word)
    no_punct.append(cleaned_text)

import re

no_nums = []
for text in no_punct:
    new_string = [token for token in text if not re.search(r'\d', token)]
    no_nums.append(new_string)


#lemmatizing


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

    # return lemmatized tokens as a list
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

#
#
# joined = []
# for text in no_nums:
#     joining = ' '.join(text)
#     joined.append(joining)



# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
#
# X_train, X_test, y_train, y_test = train_test_split(joined, small_sample_labels, test_size=0.2, random_state=42)

import pandas as pd
import pickle
loaded_model = pickle.load(open('final-model-1.0.pk1', 'rb'))
# loaded_model_accuracy = loaded_model.score(X_test, y_test)

key_vals = {'A': 'Governmental - Non - GAAP', 'B': 'Non - Governmental - Nonprofit', 'C':'Non - Governmental - For Profit', 'D': 'Government - GAAP', 'E': 'Supplemental Material', 'F': '10K'}
y_pred = loaded_model.predict(joined)
# print('Prediction: '+key_vals.get(y_pred[0]))
y_probs = loaded_model.predict_proba(joined)
# print('Prob prediction: ')
# print(y_probs)
# prob_predictions = pd.DataFrame(data = y_probs, columns = loaded_model.classes_)
#
print(f"Accuracy: {accuracy_score(table_file_labels, y_pred) * 100:.2f}%")
print(classification_report(table_file_labels, y_pred))

predictions_names = []
for p in y_pred:
    predictions_names.append(key_vals.get(p))


report = classification_report(table_file_labels, y_pred, output_dict = True)
df = pd.DataFrame(report).transpose()
df.to_csv('file_sample_classification_report_v1.1.csv')


table['Predicted Category'] = y_pred
table['Predicted Category Name'] = predictions_names
table.to_csv('file_sample_with_predictions_v1.1.csv')



#not sure what this is printing
# print(loaded_model.cv_results_)

# best_pipeline = loaded_model.best_estimator_
# print(best_pipeline)


#uncomment starting HERERERERERERERERERE
#
# best_tfidf = loaded_model.named_steps['tvec']
# classifier = loaded_model.named_steps['mb']
#
# idf_df = pd.DataFrame({'IDF value':best_tfidf.idf_})
# print(idf_df.sort_values(by = 'IDF value', ascending = False))
#
# feature_names = best_tfidf.get_feature_names_out()
# print(feature_names)
#
# #wordcloud test
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
#
#
#
# #
# # #donut plot for one of the documents
# #
import matplotlib
#
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
# # print("Switched to:",matplotlib.get_backend())
# #
# # # print("50th results: ")
# # # print(prob_predictions.iloc[50])
# #
# # # size_of_groups=prob_predictions.iloc[50].tolist()
# #
# #
# #
# # colors = ['#FF0000', '#0000FF', '#FFFF00',
# # 		'#ADFF2F', '#FFA500', '#E6E6FA']
# # # explosion
# # explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
# #
# # # Pie Chart
# # # plt.pie(size_of_groups, colors=colors, labels=loaded_model.classes_,
# # # 		autopct='%1.1f%%', pctdistance=0.85,
# # # 		explode=explode)
# # plt.pie(y_probs[0], colors=colors, labels=key_vals.values(),
# # 		autopct='%1.1f%%', pctdistance=0.85,
# # 		explode=explode)
# #
# # # draw circle
# # my_circle=plt.Circle( (0,0), 0.7, color='white')
# # p=plt.gcf()
# # # Adding Circle in Pie chart
# # p.gca().add_artist(my_circle)
# #
# # plt.title('Prediction per class')
# #
# # plt.show()

#confusion matrix heat map
#
from sklearn.metrics import confusion_matrix
import seaborn as sns

class_names = loaded_model.classes_
#
cf_matrix = confusion_matrix(table_file_labels, y_pred)
print(cf_matrix)
#
sns.set(font_scale=1.4) # for label size
sns.heatmap(cf_matrix, annot=True, annot_kws={"size": 16}, xticklabels=class_names, yticklabels=class_names, cmap = 'crest', fmt='d') # font size
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('heat-map-file-sample-1_1.png')

plt.show()

#
# class_a_token_count = classifier.feature_count_[0]
# class_b_token_count = classifier.feature_count_[1]
# class_c_token_count = classifier.feature_count_[2]
# class_d_token_count = classifier.feature_count_[3]
# class_e_token_count = classifier.feature_count_[4]
# class_f_token_count = classifier.feature_count_[5]
# #
# # # create a DataFrame of tokens with their separate Class 1 and Class 2 counts
# #
# tokens = pd.DataFrame(
#     {'token': feature_names, 'Class_A': class_a_token_count, 'Class_B': class_b_token_count, 'Class_C': class_c_token_count, 'Class_D': class_d_token_count, 'Class_E': class_e_token_count, 'Class_F': class_f_token_count}).set_index('token')
#
#
# # # add 1 to Class 1 and Class 2 counts to avoid dividing by 0
# #
# tokens['Class_A'] = tokens.Class_A + 1
# tokens['Class_B'] = tokens.Class_B + 1
# tokens['Class_C'] = tokens.Class_C + 1
# tokens['Class_D'] = tokens.Class_D + 1
# tokens['Class_E'] = tokens.Class_E + 1
# tokens['Class_F'] = tokens.Class_F + 1
# #
# # tokens.sample(5, random_state=6)
# #
# # # convert the Class 1 and Class 2 counts into frequencies
#
# tokens['Class_A frequencies'] = tokens.Class_A / classifier.class_count_[0]
# tokens['Class_B frequencies'] = tokens.Class_B / classifier.class_count_[1]
# tokens['Class_C frequencies'] = tokens.Class_C / classifier.class_count_[2]
# tokens['Class_D frequencies'] = tokens.Class_D / classifier.class_count_[3]
# tokens['Class_E frequencies'] = tokens.Class_E / classifier.class_count_[4]
# tokens['Class_F frequencies'] = tokens.Class_F / classifier.class_count_[5]
#
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
#
# tokens = tokens.T
# freq = tokens.iloc[-6:]
# print(freq.iloc[:, 25:30])
#
#
# #add frequences for each of top five terms, word cloud other visualization..
# most_freq_class_A = freq.iloc[0, :].sort_values(ascending=False)
# print('Most frequent word in class A: ')
# print(most_freq_class_A[0:20])
#
# most_freq_class_B = freq.iloc[1, :].sort_values(ascending=False)
# print('Most frequent word in class B: ')
# print(most_freq_class_B[0:20])
#
# most_freq_class_C = freq.iloc[2, :].sort_values(ascending=False)
# print('Most frequent word in class C: ')
# print(most_freq_class_C[0:20])
#
# most_freq_class_D = freq.iloc[3, :].sort_values(ascending=False)
# print('Most frequent word in class D: ')
# print(most_freq_class_D[0:20])
#
# most_freq_class_E = freq.iloc[4, :].sort_values(ascending=False)
# print('Most frequent word in class E: ')
# print(most_freq_class_E[0:20])
#
# most_freq_class_F = freq.iloc[5, :].sort_values(ascending=False)
# print('Most frequent word in class F: ')
# print(most_freq_class_F[0:20])
#
#
# #wordcloud test
# from wordcloud import WordCloud
#
# def make_wordcloud(data, class_name):
#
#     wordcloud = WordCloud(width=800, height=400, background_color ='white').generate_from_frequencies(data)
#
#     # Display the word cloud
#     plt.figure(figsize=(10, 6))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.title(class_name)
#     plt.savefig(class_name+'.png')
#     plt.show()
#
#
# make_wordcloud(most_freq_class_A[0:20], 'Class_A')
# # make_wordcloud(most_freq_class_B[0:20], 'Class_B')
# # make_wordcloud(most_freq_class_C[0:20], 'Class_C')
# # make_wordcloud(most_freq_class_D[0:20], 'Class_D')
# # make_wordcloud(most_freq_class_E[0:20], 'Class_E')
# # make_wordcloud(most_freq_class_F[0:20], 'Class_F')