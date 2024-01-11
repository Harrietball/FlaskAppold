import wikipediaapi
import gensim
from gensim.models import Word2Vec


user = 'MyProject/1.0 (hb3229pgt@students.nulondon.ac.uk)'

wiki_wiki = wikipediaapi.Wikipedia(user, 'en')

def get_wikipages_content(page_titles):
    all_text = ""

    for title in page_titles:
        page = wiki_wiki.page(title)
        if page.exists():
            page_content = page.text
            all_text += page_content 
    return all_text

page_titles = ['Artificial intelligence', 'Modernism', 'Immunology', 'Nordic countries', 'Badminton', 'Indigenous Australians', 'Ecology', 'Solar energy', 'William Shakespeare', '2023 in science', 'List of solved missing person cases: pre-2000', 'Tartan', 'Rafael Nadal', 'List of films considered the worst', 'Timeline of 1960s counterculture', 'Presidency of Rodrigo Duterte', 'History of Palestine', 'Electric car use by country', 'History of Mexican Americans', 'Municipal history of Quebec', 'Opinion polling for the next United Kingdom general election', 'List of fake news websites', 'Criminal proceedings in the January 6 United States Capitol attack', 'Sixth Labour Government of New Zealand', 'Philippines', '2018 in paleomammalogy','Parthenon']

content_combined = get_wikipages_content(page_titles) 

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize



def nltk_preprocess(content_combined):
    words = word_tokenize(content_combined)

    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]

    return words



sentences = sent_tokenize(content_combined)
processed_sentences = [nltk_preprocess(sentence) for sentence in sentences]


model = Word2Vec(sentences=processed_sentences, vector_size=90, window=5, min_count=2, epochs=5)
word_vectors = model.wv

reference_pairs = [("full", "empty"), ("large", "small"), ("wide", "narrow"), ("man", "woman"), ("far", "near"), ("early", "late"), ("low", "high")]
target_word = "paper"

mean_reference_vector = sum(model.wv[word] for pair in reference_pairs for word in pair) / len(reference_pairs)

my_new_vector = model.wv[target_word] - mean_reference_vector
opposite_words = model.wv.similar_by_vector(my_new_vector)

model.save("word2vec.model")



