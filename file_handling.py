#used by get_text, downloads the pdfs and extracts the text and puts them into txt files
import requests
requests.packages.urllib3.disable_warnings()
import time
def getPDF_with_retry(url):
    max_retries = 3
    delay = 1
    retries = 0
    while retries < max_retries:
        try:
            # Perform the operation here
            result = getPDF(url)
            return result  # Return the result if successful
        except Exception as e:
            print(f"Attempt {retries + 1} failed. Error: {e}")
            retries += 1
            time.sleep(delay)  # Wait for a specified delay before retrying
    raise Exception("Maximum number of retries reached. Operation failed.")

def getPDF(url):
    import requests
    import os
    from pathlib import Path

    certLoc = Path('C:\\Users\\ehu\\ca-bundle.crt')
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

    response = requests.get(url, headers = headers, verify=certLoc)

    # isolate PDF filename from URL
    pdf_file_name = os.path.basename(url)
    if response.status_code == 200:

        dirpath = os.path.join(os.getcwd(), "downloaded pdfs")

        if not os.path.exists(dirpath):
            # Create the directory if it does not exist
            os.makedirs(dirpath)# print("New directory created")

        filepath = os.path.join(dirpath, pdf_file_name)

        with open(filepath, 'wb') as pdf_object:
            pdf_object.write(response.content)

        # print(f'{pdf_file_name} was successfully saved!')
        return filepath
    else:
        print(f'Uh oh! Could not download {pdf_file_name},')
        # print(f'HTTP response status code: {response.status_code}')
        return 'Unsuccessful'

def extract_text(fileID, pdfUrl, textDirectory, datetime):


    timestampAppend = []
    timestampAppend.append(fileID)

    import pdfplumber

    startDatetime = datetime.now()
    timestampAppend.append(startDatetime)

    # Example usage
    try:
        file_path = getPDF_with_retry(pdfUrl)
        print("Operation successful. Result:", file_path)
    except Exception as e:
        print("Operation failed:", e)

    afterDownloadDatetime = datetime.now()
    timestampAppend.append(afterDownloadDatetime)
    #download elapsed time
    downloadElapsedTime = afterDownloadDatetime - startDatetime
    timestampAppend.append(downloadElapsedTime)

    #skips over a page that is cannot read in
    try:
        with pdfplumber.open(file_path) as pdf:

            all_pages = []
            for page in pdf.pages:

                page_text = page.extract_text()
                all_pages.append(page_text)
                # text = '\n'.join([page.extract_text() for page in pdf.pages])

            text = '\n'.join(all_pages)

        write_to_this_path = str(textDirectory + '\\' + fileID +  '.txt')
        with open(write_to_this_path, 'w', encoding = "utf-8") as file:
            # Write the text to the file
            file.write(text)
    except:
        print('PDF could not be processed'+ str(file_path))

    afterTextExtractionDatetime = datetime.now()
    timestampAppend.append(afterTextExtractionDatetime)

    textExtractionElapsedTime = afterTextExtractionDatetime - afterDownloadDatetime
    timestampAppend.append(textExtractionElapsedTime)

    totalElapsedTime = downloadElapsedTime + textExtractionElapsedTime
    timestampAppend.append(totalElapsedTime)

    return timestampAppend