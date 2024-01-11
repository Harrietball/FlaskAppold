from flask import Flask, render_template, request
from gensim.models import Word2Vec

app = Flask(__name__)

# Load the trained Word2Vec model
model = Word2Vec.load("word2vec.model")

@app.route("/", methods=["GET", "POST"])
def index():
    least_similar_words = []

    if request.method == "POST":
        word = request.form["word"]
        
        if word:
            try:
                least_similar_words = model.wv.most_similar(negative=[word], topn=1)
            except KeyError:
                least_similar_words = ["Word not found in the model."]
    

    return render_template("index.html", least_similar_words=least_similar_words)

if __name__ == "__main__":
    app.run(debug=True)




              


