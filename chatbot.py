import json
import sqlite3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template
import numpy as np

# Ensure necessary NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

def create_database():
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS responses (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      question TEXT,
                      answer TEXT)''')
    cursor.execute("INSERT OR IGNORE INTO responses (id, question, answer) VALUES (1, 'What are the admission requirements?', 'The admission requirements at Muthiah Polytechnic College include a minimum qualification of SSLC/10th pass for Diploma courses and 12th pass for lateral entry students. Admissions are based on merit and counseling.')")
    cursor.execute("INSERT OR IGNORE INTO responses (id, question, answer) VALUES (2, 'What courses are available?', 'Muthiah Polytechnic College offers Diploma courses in Mechanical Engineering, Civil Engineering, Electrical & Electronics Engineering, Electronics & Communication Engineering, and Computer Science Engineering.')")
    cursor.execute("INSERT OR IGNORE INTO responses (id, question, answer) VALUES (3, 'What is the tuition fee?', 'The tuition fee for Diploma courses at Muthiah Polytechnic College varies based on government and management quota. Please visit the official website or contact the administration for exact details.')")
    cursor.execute("INSERT OR IGNORE INTO responses (id, question, answer) VALUES (4, 'How can I apply for a scholarship?', 'Scholarships are available for eligible students, including SC/ST, BC/MBC, and merit-based scholarships. Applications can be submitted through the Tamil Nadu government scholarship portal or via the college administration.')")
    cursor.execute("INSERT OR IGNORE INTO responses (id, question, answer) VALUES (5, 'What are the hostel facilities?', 'Muthiah Polytechnic College provides hostel facilities for both boys and girls with Wi-Fi, study rooms, a mess facility, and security.')")
    cursor.execute("INSERT OR IGNORE INTO responses (id, question, answer) VALUES (6, 'Do you provide accommodation?', 'Yes, the college provides hostel accommodation for both boys and girls with essential facilities like Wi-Fi, study rooms, and mess.')")
    cursor.execute("INSERT OR IGNORE INTO responses (id, question, answer) VALUES (7, 'Is hostel accommodation available?', 'Yes, we have hostel accommodations available with all necessary facilities.')")
    cursor.execute("INSERT OR IGNORE INTO responses (id, question, answer) VALUES (8, 'Where is the college located?', 'Muthiah Polytechnic College is located in Annamalai Nagar, Chidambaram, Tamil Nadu 608002.')")
    conn.commit()
    conn.close()

create_database()

def preprocess_text(text):
    try:
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        return ' '.join(words)
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        return preprocess_text(text)

def get_answer(user_query):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM responses")
    data = cursor.fetchall()
    conn.close()

    questions = [row[0] for row in data]
    answers = {row[0]: row[1] for row in data}

    processed_questions = [preprocess_text(q) for q in questions]
    processed_query = preprocess_text(user_query)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(processed_questions + [processed_query])
    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1])
    most_similar_index = np.argmax(similarity_scores)

    if similarity_scores[0, most_similar_index] > 0.2:  # Threshold for matching
        return answers[questions[most_similar_index]]
    else:
        return "Sorry, I don't have an answer for that. You can report this to the admin."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.form['message']
    response = get_answer(user_message)
    return json.dumps({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
