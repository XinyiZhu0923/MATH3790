# Bayesian
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.metrics import r2_score
import time
start_time=time.time()
# read cse
# file_path = 'E:/2024 Spring/MATH 3790/Assignment/Final project 2/Amozon_women_dress/NLP_DataSet.csv'
file_path = 'E:/2024 Spring/MATH 3790/Assignment/Final project 2/Amozon_women_dress/Women Dresses Reviews Dataset .csv'
df = pd.read_csv(file_path)

# Training the model
df = df.replace(r'^\s*$', np.nan, regex=True)

# Remove rows containing NaN
df = df.dropna()

# Text preprocessing functions
def preprocess_text(text):
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    # make tokens
    tokens = word_tokenize(text)
    # remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)  # return string

# precess
df['processed_text'] = df['review_text'].apply(preprocess_text)

#Feature extractor that can vectorize text features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Bayesian classification
nb_classifier = MultinomialNB()

# Create a voting ensemble classifier, containing only the Bayesian classifier
voting_classifier = VotingClassifier(estimators=[
    ('nb', nb_classifier)
], voting='soft')

# define Pipeline
pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('voting', voting_classifier)
])

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
# Run 10 model training and evaluations
for i in range(10):
    # Prepare training and testing data
    X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['recommend_index '], test_size=0.3, random_state=i)
    # Training the model
    pipeline.fit(X_train, y_train)
    # prediction
    y_pred = pipeline.predict(X_test)
    # Evaluating Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    # save
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print(f'Run {i+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')

# built DataFrame
result_df = pd.DataFrame({
    "Run": range(1, 11),
    "Accuracy": accuracy_list,
    "Precision": precision_list,
    "Recall": recall_list,
    "F1": f1_list
})

# save
# excel_file = "Results_B.xlsx"
# result_df.to_excel(excel_file, index=False)

# print(f"Metrics results saved to {excel_file}")
total_time = time.time() - start_time
print(total_time/10)

# # decision  tree
# import numpy as np
# import pandas as pd
# import re
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import VotingClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,mean_squared_error
# from sklearn.metrics import r2_score
# import time
# start_time=time.time()
# # csv file
# # file_path = 'E:/2024 Spring/MATH 3790/Assignment/Final project 2/Amozon_women_dress/NLP_DataSet.csv'
# file_path = 'E:/2024 Spring/MATH 3790/Assignment/Final project 2/Amozon_women_dress/Women Dresses Reviews Dataset .csv'
# # use pandas read_csvto read file
# df = pd.read_csv(file_path)

# # substitude empty into NaN
# df = df.replace(r'^\s*$', np.nan, regex=True)

# # delete nan row
# df = df.dropna()

# # preprocess
# def preprocess_text(text):
#     # Remove punctuation and special characters
#     text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
#     # token
#     tokens = word_tokenize(text)
#     # remove stop word 
#     stop_words = set(stopwords.words('english'))
#     tokens = [token for token in tokens if token not in stop_words]
#     return ' '.join(tokens)  # 返回字符串而不是列表

# # preprocess
# df['processed_text'] = df['review_text'].apply(preprocess_text)

# # Feature extractor that can vectorize text features using TF-IDF
# tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# # Decision Tree Classifier
# dt_classifier = DecisionTreeClassifier()

# # Create a voting ensemble classifier that only contains decision tree classifiers
# voting_classifier = VotingClassifier(estimators=[
#     ('dt', dt_classifier)
# ], voting='soft')

# # define Pipeline
# pipeline = Pipeline([
#     ('tfidf', tfidf_vectorizer),
#     ('voting', voting_classifier)
# ])

# accuracy_list = []
# precision_list = []
# recall_list = []
# f1_list = []


# # Run 10 model training and evaluations
# for i in range(10):
#     # Prepare training and testing data
#     X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['recommend_index '], test_size=0.3, random_state=i)
#     # Training the model
#     pipeline.fit(X_train, y_train)
#     # predictuin
#     y_pred = pipeline.predict(X_test)
#     # evaluate the accuracy
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     

#     # save
#     accuracy_list.append(accuracy)
#     precision_list.append(precision)
#     recall_list.append(recall)
#     f1_list.append(f1)
#     cm_list.append(cm)
#     r_squared_list.append(r_squared)
#     mse_list.append(mse)
#     rmse_list.append(rmse)
#     rse_list.append(rse)

#     print(f'Run {i+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, R²={r_squared:.4f}')

# # create DataFrame
# result_df = pd.DataFrame({
#     "Run": range(1, 11),
#     "Accuracy": accuracy_list,
#     "Precision": precision_list,
#     "Recall": recall_list,
#     "F1": f1_list
# })

# # save
# # excel_file = "Results_D.xlsx"
# # result_df.to_excel(excel_file, index=False)

# # print(f"Metrics results saved to {excel_file}")

# total_time = time.time() - start_time
# print(total_time/10)


# # 2 classifier
# import numpy as np
# import pandas as pd
# import re
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import VotingClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,mean_squared_error
# from sklearn.metrics import r2_score
# import time
# start_time=time.time()
# # cse file path
# # file_path = 'E:/2024 Spring/MATH 3790/Assignment/Final project 2/Amozon_women_dress/NLP_DataSet.csv'
# file_path = 'E:/2024 Spring/MATH 3790/Assignment/Final project 2/Amozon_women_dress/Women Dresses Reviews Dataset .csv'
# # use pandas to read csv
# df = pd.read_csv(file_path)

# # substitute 
# df = df.replace(r'^\s*$', np.nan, regex=True)

# # Remove rows containing NaN
# df = df.dropna()

# # preprocess
# def preprocess_text(text):
#     # Remove punctuation and special characters
#     text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
#     # tokens
#     tokens = word_tokenize(text)
#     # delete stop words
#     stop_words = set(stopwords.words('english'))
#     tokens = [token for token in tokens if token not in stop_words]
#     return ' '.join(tokens)  # return string

# # processed
# df['processed_text'] = df['review_text'].apply(preprocess_text)



# # Feature extractor that can vectorize text features using TF-IDF
# tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# # Decuision Tree Classifier
# dt_classifier = DecisionTreeClassifier()

# # Bayesian Classifier
# nb_classifier = MultinomialNB()

# # built voting classifier
# voting_classifier = VotingClassifier(estimators=[
#     ('dt', dt_classifier), 
#     ('nb', nb_classifier)
# ], voting='soft')

# # define  Pipeline
# pipeline = Pipeline([
#     ('tfidf', tfidf_vectorizer),
#     ('voting', voting_classifier)
# ])
# accuracy_list = []
# precision_list = []
# recall_list = []
# f1_list = []


# # run ten time 
# for i in range(10):
#     # training set and test set 
#     X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['recommend_index '], test_size=0.3, random_state=i)
#     # train the model
#     pipeline.fit(X_train, y_train)
#     # prediction
#     y_pred = pipeline.predict(X_test)
#     # evaluate the accurate
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')

#     # save
#     accuracy_list.append(accuracy)
#     precision_list.append(precision)
#     recall_list.append(recall)
#     f1_list.append(f1)
#     

#     print(f'Run {i+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')

# # build DataFrame
# result_df = pd.DataFrame({
#     "Run": range(1, 11),
#     "Accuracy": accuracy_list,
#     "Precision": precision_list,
#     "Recall": recall_list,
#     "F1": f1_list,
#     
# })

# # save
# # excel_file = "Results_2.xlsx"
# # result_df.to_excel(excel_file, index=False)

# # print(f"Metrics results saved to {excel_file}")
# total_time = time.time() - start_time
# print(total_time/10)






# # #ensemble model
# import numpy as np
# import pandas as pd
# import re
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import VotingClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,mean_squared_error
# from sklearn.metrics import r2_score
# import time
# start_time = time.time()
# # csv path
# # file_path = 'E:/2024 Spring/MATH 3790/Assignment/Final project 2/Amozon_women_dress/NLP_DataSet.csv'
# file_path = 'E:/2024 Spring/MATH 3790/Assignment/Final project 2/Amozon_women_dress/Women Dresses Reviews Dataset .csv'
# # use pandas to read the file 
# df = pd.read_csv(file_path)

# # substiture the empty
# df = df.replace(r'^\s*$', np.nan, regex=True)

# # Remove rows containing NaN
# df = df.dropna()

# #Text preprocessing functions
# def preprocess_text(text):
#     # Text preprocessing functions
#     text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
#     # token
#     tokens = word_tokenize(text)
#     # Remove stop words
#     stop_words = set(stopwords.words('english'))
#     tokens = [token for token in tokens if token not in stop_words]
#     return ' '.join(tokens)  # 返回字符串而不是列表

# # Remove stop words
# df['processed_text'] = df['review_text'].apply(preprocess_text)



# # Feature extractor that can vectorize text features using TF-IDF
# tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# # Three classifiers
# dt_classifier = DecisionTreeClassifier(criterion='entropy')
# nb_classifier = MultinomialNB()
# lr_classifier = LogisticRegression(max_iter=1000)

# # Create a voting ensemble classifier containing three classifiers
# voting_classifier = VotingClassifier(estimators=[
#     ('dt', dt_classifier), 
#     ('nb', nb_classifier),
#     ('lr', lr_classifier)
# ], voting='soft')

# # Defining Pipelines
# pipeline = Pipeline([
#     ('tfidf', tfidf_vectorizer),
#     ('voting', voting_classifier)
# ])

# accuracy_list = []
# precision_list = []
# recall_list = []
# f1_list = []

# # Run 10 model training and evaluations
# for i in range(10):
#     # Prepare training and testing data
#     X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['recommend_index '], test_size=0.3, random_state=i)
#     # Training the model
#     pipeline.fit(X_train, y_train)
#     # prediction
#     y_pred = pipeline.predict(X_test)
#     # Training the model
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     

#     # save
#     accuracy_list.append(accuracy)
#     precision_list.append(precision)
#     recall_list.append(recall)
#     f1_list.append(f1)
#     cm_list.append(cm)
#     r_squared_list.append(r_squared)
#     mse_list.append(mse)
#     rmse_list.append(rmse)
#     rse_list.append(rse)

#     print(f'Run {i+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')

# # bult DataFrame
# result_df = pd.DataFrame({
#     "Run": range(1, 11),
#     "Accuracy": accuracy_list,
#     "Precision": precision_list,
#     "Recall": recall_list,
#     "F1": f1_list
# })

# # save
# # excel_file = "Results_3.xlsx"
# # result_df.to_excel(excel_file, index=False)

# # print(f"Metrics results saved to {excel_file}")
# total_time = time.time() - start_time
# print(total_time/10)