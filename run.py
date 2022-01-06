# importing necessary libraries and functions
import re
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, session
from gensim.models import Word2Vec,FastText,KeyedVectors
#import webbrowser

app = Flask(__name__) #Initialize the flask App

app.secret_key = 'BAD_SECRET_KEY'

def fileread(filepath):
    with open(filepath) as f:
        data = f.read()
        dictionary = json.loads(data)
    return dictionary

unigrams = fileread('Dictionary/ugramsdict.txt')
bigrams = fileread('Dictionary/bgramsdict.txt')
trigrams = fileread('Dictionary/tgramsdict.txt')
fourgrams = fileread('Dictionary/fgramsdict.txt')

#read example file
fileName0 = 'examples/examples.txt'
fileName1 = 'examples/corpus_examples.txt'

def readFile(filename):
    fileObj = open(filename, "r") #opens the file in read mode
    data = fileObj.read().splitlines() #puts the file into an array
    fileObj.close()
    return data

data0 = readFile(fileName0)
data1 = readFile(fileName1)
#load model
path0 ='models/word2vec_model/word2vec.model'
#path1 =r'models\fasttext_model\fasttext.model'
#path2=r'models\glove_model\results_glove.word2vec.txt'
#path3=r'models\bigram_word2vec_model\bigrams_word2vec.model'



@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/Obsessions')
def Obsessions():

    return render_template('Obsessions.html')

@app.route('/Compulsions')
def Compulsions():
    
    return render_template('Compulsions.html') 

@app.route('/Related Vocabulary')
def Vocabulary():
    subtype = request.args.get('type') 
    # print(subtype)
    subtype = subtype.lower()
    session['sub_type'] = request.args.get('type')
    unigram = len(set(unigrams[subtype]))
    pharase = len(set(bigrams[subtype]))+len(set(trigrams[subtype]))+len(set(trigrams[subtype]))
    
    return render_template('Related Vocabulary.html',subtype=subtype,unigram = unigram,pharase=pharase) 

@app.route('/Words')

def Words():
    
    subtype = session['sub_type']
    subtype = subtype.lower() 
    vocabtype = request.args.get('radio')
    unigram = sorted(set(unigrams[subtype]))
    return render_template('Words.html',unigram=unigram,vocabtype=vocabtype,subtype=subtype,enumerate=enumerate) 
       

@app.route('/Pharase')
def Pharases():
    subtype = session['sub_type'] 
    subtype = subtype.lower()
    vocabtype = request.args.get('radio') 
    # print(subtype,vocabtype)
    bigram = sorted(set(bigrams[subtype]))
    trigram = sorted(set(trigrams[subtype]))
    fourgram = sorted(set(fourgrams[subtype]))
    return render_template('Pharase.html',bigram=bigram,trigram=trigram,fourgram=fourgram,subtype=subtype,vocabtype=vocabtype,enumerate=enumerate)  

@app.route('/Similar Words')
def Predict():
    query = request.args.get('gram') 
    phrase =request.args.get('ngrams')
    return render_template('Similar Words.html',phrase=phrase,query=query)  

@app.route('/Similarity')
def Smilarity():
    query = request.args.get('query','') 
    phrase =request.args.get('phrase')
    model =request.args.get('model')
 

    def query_examples(query,phrase,data):
        if phrase == 'unigram':
            query_example = [sentence.lower() for sentence in data if re.search(r'\b' + query + r'\b', sentence.lower())]
        else:
            q = query.replace("_"," ")
            query_example = [sentence.lower() for sentence in data if re.search(r'\b' + q + r'\b', sentence.lower())]
        return query_example

    def similar_words(models,q):
        similarity = []
        words= []
        if q in models.wv.vocab: 
            for similar_word, similar in models.wv.most_similar(q, topn=50):
                words.append(similar_word)
                similarity.append(similar)
        else:
            pass

        return similarity,words

    def example(word,data,phrase):
       

        loc = {}
        dct = {}
        for i in word:
            dct[i] = []
            loc[i] = []

        for sentence in data:
            for w in word:

                if phrase == 'unigram':
                    if re.search(r'\b' + w + r'\b', sentence):
                        iter = re.search(r"\b" + w + r"\b", sentence).start()
                        dct[w].append(sentence)
                        loc[w].append(iter)
                else:
                    w1 = w.replace('_',' ') #change this code

                    if re.search(r'\b' + w1 + r'\b', sentence):
                        iter = re.search(r"\b" + w1 + r"\b", sentence).start()
                        dct[w].append(sentence)
                        loc[w].append(iter)

        values = list(dct.values())
        keys = list(dct.keys())
        postion = list(loc.values())
        length = [len(i) for i in keys] 

        return values,keys,postion,length


    if model  == 'Word2vec' and phrase == 'unigram':
        w2v = Word2Vec.load(path0)
        similarity,words = similar_words(w2v,query)
        values,keys,postion,length = example(words,data0,phrase)
        query_example =  query_examples(query,phrase,data0)
        


    elif model =='Fasttext' and phrase == 'unigram':
        ft = FastText.load(path1)
        similarity,words = similar_words(ft,query)
        values,keys,postion,length = example(words,data1,phrase)
        query_example =  query_examples(query,phrase,data0)
        


    elif model =='Glove' and phrase == 'unigram':
        glove = KeyedVectors.load_word2vec_format(path2)
        similarity,words = similar_words(glove,query)
        values,keys,postion,length = example(words,data1,phrase)
        query_example =  query_examples(query,phrase,data0)

    elif model  == 'Word2vec' and phrase == 'bigram':
        bw2v = Word2Vec.load(path3)
        similarity,words = similar_words(bw2v,query)
        values,keys,postion,length = example(words,data1,phrase)
        query_example =  query_examples(query,phrase,data0)
    
    elif model  == 'Fasttext' and phrase == 'bigram':
        pass

    elif model  == 'Glove' and phrase == 'bigram':
        pass
        
    elif model  == 'Word2vec' and phrase == 'trigram':
        pass

    elif model  == 'Fasttext' and phrase == 'trigram':
        pass

    elif model  == 'Glove' and phrase == 'trigram':
        pass

    elif model  == 'Word2vec' and phrase == 'fourgram':
        pass

    elif model  == 'Fasttext' and phrase == 'fourgram':
        pass

    elif model  == 'Glove' and phrase == 'fourgram':
        pass
    
    


    return render_template('Similarity.html',phrase=phrase,query=query,model=model,query_example = query_example,
    similarity=similarity,
    words=words,
    zip=zip,
    values=values,
    keys=keys,
    position=postion,
    length=length)  





if __name__ == '__main__':
    app.run(debug=True)
