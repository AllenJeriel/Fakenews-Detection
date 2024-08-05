import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report, accuracy_score, auc , roc_curve,precision_recall_curve
import warnings
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pickle
warnings.filterwarnings('ignore')
import scipy.stats as stats
# importing libraries
from nltk.tokenize import word_tokenize,sent_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,LancasterStemmer
from contractions import fix
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas
from tqdm import tqdm
tqdm.pandas()
from wordcloud import WordCloud
from gensim.models import Word2Vec,doc2vec
import nltk
df = pd.read_csv("fake_news.csv")
df.dropna(subset="text",inplace=True)
data = df[['text','label']]
x_train , x_test , y_train , y_test = train_test_split(data['text'], data['label'] , test_size=0.2)
## We build some function for data cleaning and find insights from it
def remove_blank(data):
    if pd.isnull(data):  # Check if data is NaN
        return ""  # Return an empty string for missing values
    formated_text = str(data).replace("\\n", " ").replace("\t", " ")
    return formated_text

def expand_text(data):
    fixed=fix(data)
    return fixed

def handle_accented(data):
    fixed_text=unidecode(data)
    return fixed_text


stopword_list=stopwords.words("english")
stopword_list.remove('no')
stopword_list.remove('nor')
stopword_list.remove('not')

def clean_text(data):
    tokens=word_tokenize(data)
    text=[i.lower() for i in tokens if (i.lower() not in punctuation) and (i.lower() not in stopword_list) and (len(i)>2) and (i.isalpha())]
    return text

def lemmatization(data):
    lemma=WordNetLemmatizer()
    final_text=[]
    for i in data:
        lemma_word=lemma.lemmatize(i)
        final_text.append(lemma_word)
    return " ".join(final_text)
clean_train=x_train.apply(remove_blank)
clean_test=x_test.apply(remove_blank)

#clean_train=clean_train.apply(expand_text)
#clean_test=clean_test.apply(expand_text)

clean_train=clean_train.apply(handle_accented)
clean_test=clean_test.apply(handle_accented)

clean_train=clean_train.apply(clean_text)
clean_test=clean_test.apply(clean_text)

clean_train=clean_train.apply(lemmatization)
clean_test=clean_test.apply(lemmatization)
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(clean_train) 
tfidf_test=tfidf_vectorizer.transform(clean_test)
import pickle

# Assuming tfidf_vectorizer is your TfidfVectorizer object
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)
PAC=PassiveAggressiveClassifier(max_iter=1000)
PAC.fit(tfidf_train,y_train)
y_pred=PAC.predict(tfidf_test)
MultNB = MultinomialNB()
MultNB.fit(tfidf_train,y_train)
MultNB_pred=MultNB.predict(tfidf_test)
rfc=RandomForestClassifier(n_estimators= 10, random_state= 7)
rfc.fit(tfidf_train,y_train)
rfc_pred = rfc.predict(tfidf_test)
from sklearn.svm import SVC
# Create an instance of the Support Vector Machine classifier
svm_classifier = SVC(kernel='linear')  # You can choose different kernels (linear, rbf, etc.)

# Fit the SVM model to the training data
svm_classifier.fit(tfidf_train,y_train)

# Make predictions on the testing data
svm_pred = svm_classifier.predict(tfidf_test)

# Evaluate the accuracy of the SVM classifier
accuracy = accuracy_score(y_test, svm_pred)
print(f"Accuracy: {accuracy}")

from sklearn.linear_model import LogisticRegression
# Create an instance of the Logistic Regression classifier
logreg_classifier = LogisticRegression()

# Fit the Logistic Regression model to the training data
logreg_classifier.fit(tfidf_train,y_train)

# Make predictions on the testing data
logreg_pred = logreg_classifier.predict(tfidf_test)

# Evaluate the accuracy of the Logistic Regression classifier
accuracy = accuracy_score(y_test, logreg_pred)
print(f"Accuracy: {accuracy}")
try:
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(tfidf_test, y_test)
    print(f'Ensemble Learning Accuracy: {round(result*100,2)}%')
except:    
    Ensemb = VotingClassifier( estimators = [('PAC',PAC),('MultNB',MultNB),('rfc',rfc),('svm',svm_classifier),('lr',logreg_classifier)], voting = 'hard')
    Ensemb.fit(tfidf_train,y_train)
    Ensemb_pred=Ensemb.predict(tfidf_test)
    filename = 'finalized_model.sav'
    pickle.dump(Ensemb, open(filename, 'wb'))
    score4=accuracy_score(y_test,Ensemb_pred)
    print(f'Ensemble Learning Accuracy: {round(score4*100,2)}%')