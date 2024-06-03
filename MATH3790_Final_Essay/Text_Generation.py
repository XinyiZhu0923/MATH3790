import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import hmm
import random
from collections import defaultdict

# Make sure NLTK resources are downloaded (if not already)
nltk.download('punkt')
nltk.download('stopwords')

# Read the file
# file_path = 'E:/2024 Spring/MATH 3790/Assignment/Final project 2/Amozon_women_dress/NLP_DataSet.csv'  # Replace with your actual file path
file_path = 'E:/2024 Spring/MATH 3790/Assignment/Final project 2/Amozon_women_dress/Women Dresses Reviews Dataset .csv'
df = pd.read_csv(file_path)

# Delete row with NaN or empty string 
df.dropna(subset=['review_text'], inplace=True)
df['review_text'] = df['review_text'].astype(str).str.strip()  # Convert all text to a string and remove leading and trailing spaces

# pre process the text
def preprocess_text(text):
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text.lower().strip())
    # take tokens
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens  # Returns the list after word segmentation

# Preprocess the text
df['processed_text'] = df['review_text'].apply(preprocess_text)

# HMM Part-of-Speech Tagger
tagged_sents = [nltk.pos_tag(tokens) for tokens in df['processed_text']]
tagger = hmm.HiddenMarkovModelTrainer().train(tagged_sents)

# Function to filter meaningless words
def filter_meaningless_words(tagged_tokens):
    # Define the set of part-of-speech tags that need to be removed
    meaningless_tags = {'DT', 'CC', 'IN', 'TO', 'PDT', 'RP'}
    # Filter out meaningless words
    filtered_tokens = [(token, tag) for token, tag in tagged_tokens if tag not in meaningless_tags]
    return filtered_tokens

# Applying filter functions
df['filtered_tokens'] = df['processed_text'].apply(lambda tokens: filter_meaningless_words(tagger.tag(tokens)))

# # aave into excel file
# output_file = 'processed_reviews.xlsx'
# with pd.ExcelWriter(output_file) as writer:
#     # Save the processed data frame
#     df[['review_text', 'processed_text', 'filtered_tokens']].to_excel(writer, sheet_name='Processed Data', index=False)

#     # Save words and their part-of-speech tags
#     for index, row in df.iterrows():
#         token_data = [(token, tag) for token, tag in row['filtered_tokens']]
#         token_df = pd.DataFrame(token_data, columns=['Token', 'Tag'])
#         token_df.to_excel(writer, sheet_name=f'Review {index + 1} Tokens', index=False)

# print(f"Processed data and tokens saved to {output_file}")

# Building a bigram model function
def build_bigram_model(data):
    bigrams = defaultdict(list)
    for tokens in data:
        for w1, w2 in zip(tokens[:-1], tokens[1:]):
            bigrams[w1].append(w2)
    return bigrams

# Building a bigram model function
positive_data = df[df['recommend_index '] == 1]['processed_text']
negative_data = df[df['recommend_index '] == 0]['processed_text']

positive_bigram_model = build_bigram_model(positive_data)
negative_bigram_model = build_bigram_model(negative_data)

#List of common negative emotion words
negative_words = set(['bad', 'worst', 'terrible', 'awful', 'disappointing', 'poor', 'negative', 'horrible', 'dislike', 'hate', 'fail', 'failed'])

# Sentence Function
def generate_sentence(bigram_model, negative=False):
    for _ in range(100):  # Try 100 times to generate a suitable sentence
        start_word = random.choice(list(bigram_model.keys()))
        sentence = [start_word]

        while True:
            current_word = sentence[-1]
            next_word = random.choice(bigram_model.get(current_word, ["."]))
            if next_word == "." or len(sentence) >= 15:  # Limit the length of sentences to 15 words or less
                break
            sentence.append(next_word)

        generated_sentence = " ".join(sentence).capitalize() + "."

        if not negative or any(word in negative_words for word in sentence):
            return generated_sentence

    return "Failed to generate a negative sentence."


print(generate_sentence(positive_bigram_model))
print(generate_sentence(negative_bigram_model, negative=True))
# Generate multiple sentences with positive and negative sentiment
num_sentences = 10
print("Positive Sentences:")
for _ in range(num_sentences):
    print(generate_sentence(positive_bigram_model))
    print()
print("==========================================================================================================")
print("\nNegative Sentences:")
for _ in range(num_sentences):
    print(generate_sentence(negative_bigram_model, negative=True))
    print()
