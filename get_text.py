#grabs the files from the internet and turns them into txt files
from lib.file_handling import extract_text
import os as os

import sys as sys
import pandas as pd

#reading in url data from csv file
fileList = pd.read_csv(os.path.join(os.getcwd(),'file_sample_for_NB_validation(in).csv'))
fileList = fileList.values.tolist()

cwd = os.getcwd()

textDownloadsFolder = 'all_text_files'
textFileDirectory = os.path.join(cwd, textDownloadsFolder)


from datetime import datetime
# at start of a cycle
theStart = datetime.now()
timestamps = []

for f in fileList:

    txt_path = str(textFileDirectory + '\\' + f[0] + '.txt')
    if not os.path.exists(txt_path):
        # Create the textfile directory if it does not exist
        fileUrl = str('https://emma.msrb.org/' + f[0] + '.pdf')
        time_result = extract_text(f[0], fileUrl, textFileDirectory, datetime)
        timestamps.append(time_result)

# label_series = pd.Series(labels)
# labelsDirectory = os.path.join(labelsFolderDirectory, 'labels.csv')
# label_series.to_csv(labelsDirectory, index = False)


# at end of cycle
endDatetime = datetime.now()
cycleTimeElapsed = (endDatetime - theStart).total_seconds()
# Printing a timestamp
fmtStartDatetime = theStart.strftime("%m/%d/%Y %H:%M")

print('Start: ', theStart)
print('End: ', endDatetime)
print('Elapsed: ', cycleTimeElapsed)

dir_down = 'C:\\Users\\ehu\\OneDrive - MUNICIPAL SECURITIES RULEMAKING BOARD\\Documents\\projects\\accounting classifier\\downloaded pdfs'
if os.path.exists(dir_down):
    import shutil
    shutil.rmtree(dir_down)

print(len(timestamps))
for time in timestamps:
    print('\nDownload elapsed time ', time[3])
    print('Text extraction elapsed time ', time[5])
    print('Total elapsed time: ', time[6])
    print('------------------------')