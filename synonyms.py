'''Semantic Similarity: starter code

Original Author: Michael Guerzhoy, University of Toronto. October, 2014.
Modified with permission by Marcus Gubanyi, Concordia University-Nebraska. October, 2024.
Finished by Anna Chen as Foundation of Data Science course project. October, 2024.
'''

import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Download necessary resources (if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Contractions to expand (I want to preserve the "no" in the sentence)
contractions = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "won't": "will not",
    "can't": "cannot",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "mustn't": "must not",
    "mightn't": "might not",
    "needn't": "need not",
    "shan't": "shall not",
    "aren't": "are not",
    "ain't": "is not",  # Colloquial
}

def norm(vec):
    '''Return the norm of a vector stored as a dictionary,
    as described in the handout for Project 2.
    '''
    
    sum_of_squares = 0.0  # floating point to handle large numbers
    for x in vec:
        sum_of_squares += vec[x] * vec[x]
    
    return math.sqrt(sum_of_squares)


def cosine_similarity(vec1, vec2):
    '''Return the cosine similarity of sparse vectors vec1 and vec2,
    stored as dictionaries as described in the handout for Project 2.
    '''
    
    dot_product = 0.0  # floating point to handle large numbers
    for x in vec1:
        if x in vec2:
            dot_product += vec1[x] * vec2[x]
    
    return dot_product / (norm(vec1) * norm(vec2))




def expand_contractions(text, contractions):
    """ Expand contractions to make the processed text easier to handle."""
    
    for contraction, expansion in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text)
    
    return text


def get_sentence_lists(text):
    """Takes in a string text and returns a list that contains lists of strings, one list for each sentence.
        
        Negatives such as "no" and "not" are preserved, since they have a semantic impact to the words.

        0. Convert words to lowercase and expand contractions
        1. Break the text into sentences based on ., ?, or !.
        2. Tokenize each sentence into words and dropping unneeded punctuation
        3. Remove stop words (common words like "the", "is", etc.).
        4. Perform stemming and lemmatization to change words to their base form and improve performance.
    """

    # Define stop words but keep "no", "not", and "nor"
    stop_words = set(stopwords.words('english')) - {"no", "not", "nor"}
    
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Step 0: Convert everything to lowercase first and expand contractions
    text = text.lower()
    text = expand_contractions(text, contractions)
    
    # Step 1: Split the text into sentences based on ".", "?", or "!"
    sentences = re.split(r'[.!?]', text)
    
    sentence_lists = []
    
    for sentence in sentences:
        # Step 2: Tokenize sentence into words, ignoring unwanted punctuations
        words = re.findall(r'\b\w+\b', sentence)
        
        # Step 3: Remove stop words (excluding "no", "not", and "nor")
        words = [word for word in words if word not in stop_words]
        
        # Step 4: Perform stemming and lemmatization
        words = [stemmer.stem(word) for word in words]
        words = [lemmatizer.lemmatize(word) for word in words]
        
        # Add the processed sentence to the list if it's not empty
        if words:
            sentence_lists.append(words)
    
    return sentence_lists



def get_sentence_lists_from_files(filenames):
    """Accepts a list of file names and returns sentence lists for each file."""
    
    all_sentence_lists = []  # To store sentence lists from all files
    
    for filename in filenames:
        try:
            # Step 1: Open and read the file
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Step 2: Use the get_sentence_lists function to process the text
            sentence_lists = get_sentence_lists(text)
            
            # Step 3: Add the sentence lists to the result
            all_sentence_lists.append(sentence_lists)
        
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    return all_sentence_lists



def build_semantic_descriptors(sentences):
    pass


def most_similar_word(word, choices, semantic_descriptors):
    pass


def run_similarity_test(filename, semantic_descriptors):
    pass

if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog. But, isn't it the smartest animal? No! Don't underestimate it."
    result = get_sentence_lists(text)
    print(result)

