from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from gensim.models import Word2Vec
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from scipy import spatial
import spacy
import ru_core_news_sm
nltk.download('punkt')
nlp = spacy.load("ru_core_news_sm")
nlp = ru_core_news_sm.load()


app = FastAPI()

@app.post("/tf_idf/")
def text_similarity_tfidf(query: str):
    data = pd.read_pickle(os.path.join('vacancies_data.pkl'))
    temp_data = data.copy()
    tfidf_vectorizer = TfidfVectorizer()

    descriptions_list = temp_data['preprocessed_description']

    similarity = []
    for i in descriptions_list:
        tfidf_matrix_description = tfidf_vectorizer.fit_transform([i])
        tfidf_matrix_query = tfidf_vectorizer.transform([query])
        similarity.append(cosine_similarity(tfidf_matrix_description, tfidf_matrix_query)[0][0])

    temp_data['similarity'] = similarity
    temp_data = temp_data.sort_values(by='similarity', ascending=False)
    vac_list = []
    for i in temp_data.head(10).iterrows():
        vacancy = {
            'name': i[1].get('name'),
            'description': i[1].get('description'),
            'scores': i[1].get('similarity')
        }
        vac_list.append(vacancy)
    del temp_data
    return {"data": vac_list}



@app.post("/spacy/")
def text_similarity(query: str):
  data = pd.read_pickle(os.path.join('vacancies_data.pkl'))
  query = nlp(query)
  query_vec = query.vector
  temp_data = data.copy()

  descriptions_list = temp_data['preprocessed_description']

  description_vector = []
  for i in descriptions_list:
    description_vector.append(nlp(i).vector)

  similarity = []
  temp_data['vectors'] = description_vector
  for x in description_vector:
    similarity.append(cos_sim(x, query_vec))
  temp_data['similarity'] = similarity
  temp_data = temp_data.sort_values(by = 'similarity', ascending = False)
  vac_list = []
  for i in temp_data.head(10).iterrows():
    vacancy = {
      'name': i[1].get('name'),
      'description': i[1].get('description'),
      'scores': i[1].get('similarity')
    }
    vac_list.append(vacancy)
  del temp_data
  return {"data": vac_list}

def cos_sim(vector1, vector2):
  cos_sim = 1 - spatial.distance.cosine(vector1, vector2)
  return cos_sim


@app.post("/word2vec/")
def Word2Vec_model(text: str):
    data = pd.read_pickle(os.path.join('vacancies_data.pkl'))
    temp_data = data.copy()
    sentences = temp_data['preprocessed_description'].tolist()
    all_words = [nltk.word_tokenize(sent) for sent in sentences]
    word2vec = Word2Vec(all_words, min_count=2)
    sentences_similarity = np.zeros(len(sentences))
    text_words = [w for w in text.split() if w in word2vec.wv]
    for idx, sentence in enumerate(sentences):
        sentence_words = [w for w in sentence.split() if w in word2vec.wv]
        sim = word2vec.wv.n_similarity(text_words, sentence_words)
        sentences_similarity[idx] = sim
    temp_data['similarity'] = sentences_similarity
    temp_data = temp_data.sort_values(by='similarity', ascending=False)
    vac_list = []
    for i in temp_data.head(10).iterrows():
        vac = {
            'name': i[1].get('name'),
            'description': i[1].get('description'),
            'scores': i[1].get('similarity')
        }
        vac_list.append(vac)
    del temp_data
    return {"data": vac_list}