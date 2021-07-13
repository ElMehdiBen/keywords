# frontend/main.py

import json, re
import streamlit as st
from elasticsearch import Elasticsearch
from keybert import KeyBERT
# you can use RFC-1738 to specify the url
elastic = Elasticsearch(['http://206.81.23.137:9200'])

def pre_process(text):
    # XYZ
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

with st.form(key='my_form'):
    keyword = st.text_input(label='Enter your keyword')

    submit = st.form_submit_button(label='Submit')

if submit:
    st.write(f'Checking Internal DB for : {keyword}')
    body = {
        "query": {
            "match": {
            "website_content": keyword
            }
        }
    }
    res = elastic.search(index="open_web", body=body)
    st.write(f"Found {res['hits']['total']['value']} documents")
    general_content = ""
    for hit in res['hits']['hits']:
        general_content += hit["_source"]["website_content"]
    pre_text = pre_process(general_content)
    stopwords = get_stop_words("./stopwords.txt")
    # keywords = tfidf(pre_text, stopwords)
    model = KeyBERT('distiluse-base-multilingual-cased-v1')
    keywords = model.extract_keywords(pre_text, keyphrase_ngram_range=(2, 2), stop_words=stopwords, top_n=10, use_mmr=True, diversity=0.3)
    st.markdown(keywords)
