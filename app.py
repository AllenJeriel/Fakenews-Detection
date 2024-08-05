from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load the trained model
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

@app.route('/')
def index():
    return render_template('index1.html')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        publisher = request.form['publisher']
        title = request.form['title']
        text = request.form['text']
        full_text = publisher + ' ' + title + ' ' + text
        text_vector = tfidf_vectorizer.transform([full_text])
        prediction = loaded_model.predict(text_vector)[0]
        if prediction >= 0.5:
            result = "This is Real News"
        else:
            result = "This is Fake News"
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
