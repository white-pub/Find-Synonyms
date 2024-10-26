'''
Find Synonyms Program
    
    Anna Chen
    October 2024  

This program processes text data and calculates semantic similarity between words 
based on their co-occurrence in sentences, using a bag-of-words model and cosine similarity.

Key features include:
    - Text preprocessing: expanding contractions, stemming, lemmatization, and removing stop words.
    - Building semantic descriptors: creating word co-occurrence contexts.
    - Calculating cosine similarity: measuring word relations.
    - Running similarity tests: checking program accuracy using a test file.

This program builds on "Semantic Similarity" starter code.

Starter Code
Original Author: Michael Guerzhoy, University of Toronto, October 2014.
Modified with permission by Marcus Gubanyi, Concordia University-Nebraska, October 2024.
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
    """
    Accepts a list of file names and returns one sentence list
    that has all the sentence from all the files.

    Accept:["file1", "file2","file3", ...]
    Return:[['not', 'agre', 'either'], ['like', 'plant', 'snake'],...]
    """
    
    all_sentence_lists = []  # To store sentence lists from all files
    
    for filename in filenames:
        try:
            # Open and read the file
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Use the get_sentence_lists function to process the text
            sentence_lists = get_sentence_lists(text)
            
            # Add the sentence lists to the result 
            all_sentence_lists.extend(sentence_lists)
        
        # Handle exceptions (errors)
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    return all_sentence_lists


def build_semantic_descriptors(sentences):
    """Builds semantic descriptors for each word in a list of sentences.

    Accepts:
        sentences (list of lists): Each sublist represents a sentence, containing words as strings.
        Example [['not', 'agre', 'either'], ['like', 'plant', 'snake'],...]

    Returns:
        A dictionary where each key is a word, and the value is a dictionary
            representing the semantic descriptor (context counts) of that word.
        Example {
            'man': {'i': 3, 'am': 3, 'sick': 1, 'an': 1, 'unattractive': 1,...},
            'liver': {'i': 1, 'believe': 1, 'my': 1, 'is': 1, 'diseased': 1},
            ...
            }

    """
    
    # Outer dictionary to hold all word contexts
    # Each key value pair inside is a unique word and its context(bag of word)
    semantic_descriptors = {}  

    # Go through each word and build the word context (co-occurring words word count)
    for sentence in sentences:
        
        # Get unique words to avoid double counting in a sentence
        unique_words = set(sentence)  

        for word in unique_words:
            # Initialize inner dictionary (word_context) for the word if it doesn't exist
            if word not in semantic_descriptors:
                semantic_descriptors[word] = {}

            # Remember the inner dictionary that matches this word
            word_context = semantic_descriptors[word]  

            # Count co-occurrences with other words in the sentence
            for companion_word in unique_words:
                if companion_word != word:  # Skip the word itself
                    # Update the co-occurrence count in word_context
                    if companion_word in word_context:
                        word_context[companion_word] += 1
                    else:
                        # Initialize if this is the first occurrence
                        word_context[companion_word] = 1 

    return semantic_descriptors



def most_similar_word(word, choices, semantic_descriptors):
    """
    Given a word, a list of choices, and semantic descriptors, returns the word
        in choices with the highest semantic similarity to the given word. 
    Ties are resolved by choosing the smallest index.
    
    Parameters:
        word (str): The word to compare against choices.
        choices (list of str): A list of words to compare for similarity.
        semantic_descriptors (dict): A dictionary where each key is a word, and
            the value is a dictionary representing the semantic context of that word.
                                     
    Returns:
        best_synonym (str): The choice with the highest semantic similarity to the word.
            If the similarity is uncomputable, it defaults to −1.
    """
    
    # Ensure the word exists in semantic_descriptors
    if word not in semantic_descriptors:
        # Return first choice if word has no descriptor (solve the tie by index)
        return choices[0]  
    
    word_descriptor = semantic_descriptors[word]
    max_similarity = -1 # default for when the similarity can't be computed
    best_choice = None
    
    for choice in choices:
        # Only calculate similarity if the choice has a descriptor
        if choice in semantic_descriptors:
            similarity = cosine_similarity(word_descriptor, semantic_descriptors[choice])

            # Update max and best choice when encountering a higher similarity
            if similarity > max_similarity:
                max_similarity = similarity
                best_choice = choice
            
            # If there is a ties, will only update the first time: from None to current choice
            # Ensures ties are resolved by choosing the smallest index option
            elif similarity == max_similarity and best_choice is None:
                best_choice = choice
    
    # Return the best choice found, or the first choice if no valid similarity was found
    return best_choice if best_choice is not None else choices[0]


def run_similarity_test(filename, semantic_descriptors):
    """
    Takes in a filename (str) and returns the percentage of questions on which 
    most_similar_word() guesses the answer correctly using the semantic descriptors.

    The expected format of test.txt is:
    word correct_answer choice1 choice2 choice3 ...

    Returns the percentage of correct answer.
    """
    
    total_questions = 0
    correct_guesses = 0
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # Strip white space and split the line into parts (individual phrases)
                parts = line.strip().split()
                
                word = parts[0]
                correct_answer = parts[1]
                choices = parts[2:]
                
                # Call most_similar_word to get the predicted answer
                predicted_answer = most_similar_word(word, choices, semantic_descriptors)
                
                # Count total questions
                total_questions += 1
                
                # Check if the predicted answer is correct
                if predicted_answer == correct_answer:
                    correct_guesses += 1

        # Calculate percentage of correct guesses
        percentage_correct = (correct_guesses / total_questions) * 100
        print(f"test file: {filename}")
        print(f"The percentage of correct guesses is {percentage_correct} %" )
        
        return percentage_correct
    
    # Handle error
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return 0.0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0.0

def run_program(training_data, test_file):
    """
    This function accepts a list of traning_data and a test_file, then 
    run the find synonym program and print the result of the model's prediction.

    Accept: (example)
        training_data - ["train1.txt", "train2.txt", ...]
        test_file - "test.txt"
    """
    
    divider = "----------------"
    print(divider)

    print("Find synonym program started...\n")

    # Get words and sentences from training data
    processed_sentence_list = get_sentence_lists_from_files(training_data)
    print(f"Finished getting words and sentences from file (training data)")
    
    print(divider)
    
    # Get word context
    all_word_context = build_semantic_descriptors(processed_sentence_list)
    print(f"Finished getting word_context")

    print("    ======")

    run_similarity_test(test_file, all_word_context)
    print(f"The training data used: {training_data}")

    print(divider)

if __name__ == "__main__":
    """ Runs the program with different training set."""
    
    # train with Swann’s Way
    run_program(["Swann’s Way by Marcel Proust.txt"], "test.txt")

    # train with War and Peace
    run_program(["War and Peace by Leo Tolstoy.txt"], "test.txt")

    print("\n============== ⬇️⬇️ Use the altered test file ⬇️⬇️ ============================\n")
    # I altered the test.txt file to make the format fit the description 
    # on the assignment instruction.

    # train with Swann’s Way
    run_program(["Swann’s Way by Marcel Proust.txt"], "test altered.txt")

    # train with War and Peace
    run_program(["War and Peace by Leo Tolstoy.txt"], "test altered.txt")



