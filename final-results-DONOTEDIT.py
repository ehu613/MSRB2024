#deploying of the nb classifier model by dropping in pdf files into a folder called 'new files' and classifying them
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
curr_dir = os.getcwd()
rel_path = 'new files'
folder_path = os.path.join(curr_dir, rel_path)
file_list = os.listdir(folder_path)

#extracts the text of one document
def extract_text(textDirectory, fileID, pdf_path):

    import pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:

            all_pages = []
            for page in pdf.pages:
                page_text = page.extract_text()
                all_pages.append(page_text)
            # text = '\n'.join([page.extract_text() for page in pdf.pages])

            text = '\n'.join(all_pages)

        write_to_this_path = str(textDirectory + '\\' + fileID[:-4] +  '.txt')
        with open(write_to_this_path, 'w', encoding = "utf-8") as file:
            # Write the text to the file
            file.write(text)
    except:
        print('PDF could not be processed'+ fileID[:-4])

for file_name in file_list:

    pdf_file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(os.path.join(folder_path, file_name[:-4]+".txt")):
        extract_text(folder_path, file_name, pdf_file_path)
    os.remove(pdf_file_path)

txt_list = os.listdir(folder_path)


file_contents = []
for txt_name in txt_list:
    print('Reading in: '+ txt_name)
    file_path = os.path.join(folder_path, txt_name)

    with open(file_path, 'r', encoding = "utf8") as file:
        contents = file.read()
        contents = contents.lower()
        file_contents.append(contents)

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
    return lemmas

nltk.download('omw-1.4')
lemmatized = []
for text in no_nums:
   lemmatized.append(lemmatize_tokens(text))

joined = []
for text in lemmatized:
    joining = ' '.join(text)
    joined.append(joining)

#
# joined = []
# for text in no_nums:
#     joining = ' '.join(text)
#     joined.append(joining)

# joined = file_contents
import pandas as pd
import pickle
loaded_model = pickle.load(open('final-model-1.0.pk1', 'rb'))
# loaded_model_accuracy = loaded_model.score(X_test, y_test)

key_vals = {'A': 'Governmental - Non - GAAP', 'B': 'Non - Governmental - Nonprofit', 'C':'Non - Governmental - For Profit', 'D': 'Government - GAAP', 'E': 'Supplemental Material', 'F': '10K'}
y_pred = loaded_model.predict(joined)

values = [key_vals.get(key) for key in y_pred]
print(values)
print('Prediction:\n')
results_frame = pd.DataFrame({'File name': txt_list, 'Prediction': values})
print(results_frame)
y_probs = loaded_model.predict_proba(joined)
print('Prob prediction: ')
print(y_probs)
prob_predictions = pd.DataFrame(data = y_probs, columns = loaded_model.classes_)

#donut plot for one of the documents
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
def make_donut(prediction, filen):

    colors = ['#FF0000', '#0000FF', '#FFFF00',
            '#ADFF2F', '#FFA500', '#E6E6FA']
    # explosion
    explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    plt.pie(prediction, colors=colors, labels=key_vals.values(),
            autopct='%1.1f%%', pctdistance=0.85,
            explode=explode)
    # draw circle
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    p=plt.gcf()
    # Adding Circle in Pie chart
    p.gca().add_artist(my_circle)
    plt.title('Prediction for: '+filen)
    plt.show()

count = 0
for p in y_probs:
    make_donut(p, txt_list[count][:-4])
    count+=1