import os
import pandas as pd
import re
import nltk
import requests
import glob
from bs4 import BeautifulSoup
import sys

#get the stopwords data from nltk
nltk.download('stopwords',quiet=True)
#get the punkt data
nltk.download('punkt',quiet=True)
# Get the stopwords set from nltk
nltk_stopwords = set(nltk.corpus.stopwords.words('english'))


cwd = os.getcwd()
excel_file = pd.read_excel("Input.xlsx")
##########################################################


# Function to read a file with a list of stopwords and return a set of those stopwords
def read_stopwords(file_name):
    # Open the file in read mode and read its content as a string
    with open(file_name, "r") as file:
        content = file.read()

    # Split the content by newline characters and strip any whitespace from each word
    words = content.split("\n")
    words = [word.strip() for word in words]

    # Return a set of those words for faster lookup
    return set(words)

def get_stop_word_list(file_names):
   # Create an empty set to store all the stopwords from different files
    stopwords = set()

    # Loop through each file name in file_names
    for file_name in file_names:
        #add the stopwords to the stopwords set
        stopwords.update(read_stopwords(file_name))
    return stopwords



# Function to remove the stopwords from a text and return a cleaned text
def remove_stopwords(text, stopwords):

    # Split the text into words and remove punctuation and numbers
    words = re.findall (r"\w+", text)
    words = [word for word in words if word.isalpha()]

    # Filter out the words that are in the stopwords set
    words = [word for word in words if word.lower() not in stopwords]

    cleaned_text = " ".join(words)

    #Cleaned text
    return cleaned_text

# Function to count the number of syllables in a word
def count_syllables(word):
  # Use a regular expression to split the word into vowels and consonants
  word = word.lower()
  vowels = "aeiouy"
  syllables = re.findall (f"[{vowels}]+", word)

  # Count the number of syllables and apply some rules
  count = len(syllables)

  if count == 0:
    return 1
  if word.endswith ("e"):
    count -= 1
  if word.endswith ("le"):
    count += 1
  if count == 0:
    count += 1
  return count

# Function to compute the average number of words per sentence in a text
def avg_sentence_length(text):
  # Split the text into sentences and words and remove punctuation and numbers
  sentences = re.split (r"[.?!]+", text)
  sentences = [sentence for sentence in sentences if sentence]
  words = re.findall (r"\w+", text)
  words = [word for word in words if word.isalpha()]

  # Compute the 
  avg_words = len(words) / len(sentences)

  # Return the average number of words per sentence as a rounded value
  return round(avg_words, 2)


# Function to check if a word is complex or not
def is_complex(word):
  # Word is complex if it has more than two syllables
  return count_syllables(word) > 2

# Function to compute the percentage of complex words in a text
def percentage_complex(text):
  # Split the text into words and remove punctuation and numbers
  words = re.findall (r"\w+", text)
  words = [word for word in words if word.isalpha()]

  # Count the number of complex words and total words
  complex_count = sum(map(is_complex, words))
  total_count = len(words)

  # Percentage of complex words
  percentage = complex_count / total_count * 100

  # Percentage as a rounded value
  return round(percentage, 2)

# Function to compute the fog index of a text
def fog_index(avg_sentence_length,percentage_complex):
  # 0.4 * (average sentence length + percentage of complex words)

  fog = (avg_sentence_length + percentage_complex) * 0.4
  return fog


# Function to read a file with a list of words and return a set of those words
def read_word_list(file_name):
    with open(file_name, "r") as file:
        content = file.read()

    # Split the content by newline characters and strip any whitespace
    words = content.split("\n")
    words = [word.strip() for word in words]

    # Return as a set faster lookup
    return set(words)

# Function to compute the positive score
def positive_score(text, positive_words):
    #nltk function to split the text into words
    tokens = nltk.word_tokenize(text)
    all_tokens = " ".join(tokens)
    positive_words = set(positive_words)

    # Split the text into words and remove punctuation and numbers
    words = re.findall (r"\w+", all_tokens)
    words = [word for word in words if word.isalpha()]

    # Count the number of positive words in the words
    positive_count = sum(map(lambda word: word in positive_words, words))

    return positive_count

# Function to compute the negative score of a text based on a file with negative words
def negative_score(text, negative_words):
    #nltkfunction to split the text into words
    tokens = nltk.word_tokenize(text)
    all_tokens = " ".join(tokens)
    # Read the file with negative words and store them in a set
    negative_words = set(negative_words)

    # Split text into words and remove punctuation and numbers
    words = re.findall (r"\w+", all_tokens)
    words = [word for word in words if word.isalpha()]

    # Count the number of negative words
    negative_count = sum(map(lambda word: word in negative_words, words))

    return negative_count

# Function to compute the average word length of a text
def avg_word_length(text):
  

    # Split the text into words , remove punctuation and numbers
    words = re.findall (r"\w+", text)
    words = [word for word in words if word.isalpha()]

    # Compute the total number of words in the text and total length
    length_sum = sum(map(len,words))
    word_count = len(words)

    #average word length as the ratio of length_sum to word count
    avg_word_length = length_sum / word_count

    # Return as rounded value
    return round(avg_word_length,2)

# Function to compute the polarity score of a text using TextBlob library
def polarity_score(text,positive_score,negative_score):

    polarity =(positive_score - negative_score)/ ((positive_score + negative_score) + 0.000001)

    return polarity

# Function to compute the subjectivity score of a text using TextBlob library
def subjectivity_score(text,positive_score,negative_score):
    subjectivity_score = (positive_score + negative_score)/(word_count(text) + 0.000001)
    return subjectivity_score

# Function to compute the complex word count of a text
def complex_word_count(text):
  # Split the text into words and remove punctuation and numbers
  words = re.findall (r"\w+", text)
  words = [word for word in words if word.isalpha()]

  # Count the number of complex words in the words
  complex_count = sum(map(is_complex, words))

  # Return the complex word count
  return complex_count

# Function to count the words in a text
def word_count(text):
    
    # to convert the text into text nltk object
    text = nltk.Text(nltk.word_tokenize(text))

    # filter words that are not alphabetic or words that are stopwords
    words = [word for word in text if word.isalpha() and word.lower() not in nltk_stopwords]

    return len(words)
        

# Function to compute the syllable per word of a text
def syllable_per_word(text):
  # Split the text into words and remove punctuation and numbers
  words = re.findall (r"\w+", text)
  words = [word for word in words if word.isalpha()]

  # To Remove the words that end with "es" or "ed"
  words = [word for word in words if not word.endswith(("es", "ed"))]

  # Compute the syllable count and word count in the text
  syllable_count = sum(map(count_syllables, words))
  word_count = len(words)
  # Compute the syllable per word 
  syllable_per_word = syllable_count / word_count

  # Return as a rounded value
  return round(syllable_per_word,2)

# Function to count the personal pronouns in a text
def personal_pronouns(text):
    # To store the personal pronouns in a set
    personal_pronouns = {"I","we","my","ours","us"}

    # Split the text into words and remove punctuation and numbers
    words = re.findall (r"\w+", text)
    words = [word for word in words if word.isalpha()]
    # Count the number of personal pronouns in the text,exclude the word "US" when in Capitals
    personal_pronouns_count = sum(map(lambda word: word.lower() in personal_pronouns and word != "US", words))

    return personal_pronouns_count



def get_master_dictionary():
    directory = f"{cwd}/MasterDictionary"

   # to find files that have positive or negative in their names
    positive_files = glob.glob(directory + "/*positive*")
    negative_files = glob.glob(directory + "/*negative*")

    #dictionary with key positive and negative with value of  list of files
    file_dict = {"positive": positive_files, "negative": negative_files}
    return file_dict
  


def start(excel_file):
    
    #List to store the results
    results = []
    for _ , row in excel_file.iterrows():
        # Get the URL_ID and URL from the row
        url_id = row["URL_ID"]
        url = row["URL"]
       
        print('Processing URL_ID:',url_id)
        # Send a GET request to the URL
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            article_text = []
            #since text title is inside a <h1> element with class "entry"
            title_head = soup.find("h1", class_="entry-title")
            if title_head is not None:
               title = title_head.text.strip()
               article_text.append(title+'\n')
            #text article is inside a <div> element with class "td-post-content tagdiv-type"
            text_div = soup.find("div", class_="td-post-content tagdiv-type")
            if text_div is not None:
                # Find all the p elements inside the div element since data is in p elements
                p_tags = text_div.findAll("p")
                if len(p_tags) > 0:
                    # Loop through each p element
                    for p in p_tags:
                        # Get the text content of the p element and strip any whitespace
                        text = p.text.strip()
                        article_text.append(text+'\n')

                text = " ".join(article_text)
                output_article_files_path=f"{cwd}/article_text_files"
                os.makedirs(output_article_files_path, exist_ok=True)

                with open(os.path.join(output_article_files_path,str(url_id)+'.txt'),'w',encoding='utf-8') as file:
                   file.write(text)
                
                # Remove the stopwords from the text using a list of files with stopwords 
                files_with_stopwords = [f"{cwd}/StopWords/StopWords_Currencies.txt", f"{cwd}/StopWords/stopwords_Datesandnumbers.txt", f"{cwd}/StopWords/stopwords_generic.txt", f"{cwd}/StopWords/stopwords_auditor.txt", f"{cwd}/StopWords/stopwords_genericLong.txt", f"{cwd}/StopWords/stopwords_geographic.txt", f"{cwd}/StopWords/stopwords_names.txt"]
                stop_word_list =  get_stop_word_list(files_with_stopwords)
                text = remove_stopwords(text, stop_word_list)

                
                master_dictionary = get_master_dictionary()

                #Computing the variables for the text content
                positive_score_value = positive_score(text, master_dictionary["positive"])
                negative_score_value = negative_score(text, master_dictionary["negative"])
                
                polarity_score_value = polarity_score(text,positive_score_value,negative_score_value)
                subjectivity_score_value = subjectivity_score(text,positive_score_value,negative_score_value)
                
                avg_sentence_length_value = avg_sentence_length(text)
                percentage_complex_words_value = percentage_complex(text)
                fog_index_value = fog_index(avg_sentence_length_value,percentage_complex_words_value)
                
                avg_number_of_words_per_sentence_value = avg_sentence_length(text)
                complex_word_count_value = complex_word_count(text)

                word_count_value = word_count(text)
                syllable_per_word_value = syllable_per_word(text)
                personal_pronouns_value = personal_pronouns(text)
                avg_word_length_value = avg_word_length(text)

                # Create a dictionary with the above variables and their values
                result = {
                "URL_ID": url_id,
                "URL": url,
                "POSITIVE SCORE": positive_score_value,
                "NEGATIVE SCORE": negative_score_value,
                "POLARITY SCORE": polarity_score_value,
                "SUBJECTIVITY SCORE": subjectivity_score_value,
                "AVG SENTENCE LENGTH": avg_sentence_length_value,
                "PERCENTAGE OF COMPLEX WORDS": percentage_complex_words_value,
                "FOG INDEX": fog_index_value,
                "AVG NUMBER OF WORDS PER SENTENCE": avg_number_of_words_per_sentence_value,
                "COMPLEX WORD COUNT": complex_word_count_value,
                "WORD COUNT": word_count_value,
                "SYLLABLE PER WORD": syllable_per_word_value,
                "PERSONAL PRONOUNS": personal_pronouns_value,
                "AVG WORD LENGTH": avg_word_length_value
                }

                # Append the result dictionary to the results list
                results.append(result)
                results_df = pd.DataFrame(results)
                # Save the DataFrame as an excel file with the name "output.xlsx"
                results_df.to_excel(f"{cwd}/output.xlsx", index=False)
            else:
                #If div element not present,print a message indicating that the div element was not found
                print("Div element not found.")
        else:
            # Print a message indicating that the response status code was not 200 (OK)
            print(f"Response status code was not 200 (OK) for URL {url}.")

    # Print a message indicating that the excel file was created successfully
    print("Output Excel file completed successfully.")


#entry point for the code

start(excel_file)