import subprocess
subprocess.call(['pip', 'install', 'glove-python-binary'])
from gensim.models import Word2Vec
from gensim.models import FastText
import json
import os.path
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.test.utils import datapath,get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from glove import Corpus, Glove
import regex as re
import string
from spacy.lang.en import English
spacy_en = English(disable=['parser', 'ner'])

from spacy.lang.en.stop_words import STOP_WORDS as stopwords_en
stopwords_en = list(stopwords_en)

mapping_dict = {"en":[stopwords_en, spacy_en]} 

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class Functions(object):
    def __init__(self):
        pass
    
    def clean_text(self, text):
      try:
        text = str(text)
        text = text.lower().strip()
        # text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        # text = re.sub(r'\@\w+|\#','', text)
        # text = re.sub('\(.*?\)','',text)
        text = re.sub(r"\s+", " ", text)
        
      except Exception as e:
        print("\n Error in clean_text : ", e,"\n",traceback.format_exc())
      return text
      import pdb;pdb.set_trace()
      

    def lemmatized(self,text):
      lemmatizer = WordNetLemmatizer()
      wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
      pos_tagged_text = nltk.pos_tag(text.split())

      lemma_words = " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

      return lemma_words

    def get_lemma(self, text, lang, remove_stopwords=True):
            if remove_stopwords: return " ".join([tok.lemma_.lower().strip() for tok in mapping_dict[lang][1](text) if tok.lemma_ != '-PRON-' and len(tok.lemma_) > 1 and tok.lemma_ not in mapping_dict[lang][0]])
            else: return " ".join([tok.lemma_.lower().strip() for tok in mapping_dict[lang][1](text) if tok.lemma_ != '-PRON-'])
        
    def cleaning_pipeline(self, sentences, lang):
      df = pd.DataFrame(sentences, columns=["sentences"])
      df["sentences"] = df["sentences"].apply(self.clean_text)
      df = df[df["sentences"] != '']
      df = df[df["sentences"] != None]
      df = df.dropna()
      return df['sentences']
      import pdb;pdb.set_trace()

    def creat_corpus_unigram(self,sentences_list):
      """ 
      Take list of sentences and create unigram of each sentence.
    
      sentences_list: list of corpus sentences after preprocessing

      retrun list of unigrams for each sentence
      """
      unigrams =  [sent.split() for sent in sentences_list]
      return unigrams
    
    def creat_corpus_bigram(self,sentences_list):
      """ 
      Take list of sentences and create bigrams of each sentence.
    
      sentences_list: list of corpus sentences after preprocessing

      retrun list of bigrams for each sentence
      """
      bigrams = [ ([ i[0]+"_"+i[1] for i in zip(j.split()[:-1], j.split()[1:]) ]) for j in sentences_list ]
      return bigrams

    def creat_corpus_trigram(self,sentences_list):
      """ 
      Take list of sentences and create trigrams of each sentence.
    
      sentences_list: list of corpus sentences after preprocessing

      retrun list of trigrams for each sentence
      """
      trigrams = [ ([ i[0]+"_"+i[1]+"_"+i[2] for i in zip(j.split()[:-1], j.split()[1:],j.split()[2:] )]) for j in sentences_list ]
      return trigrams

    def creat_corpus_fourgram(self,sentences_list):
      """ 
      Take list of sentences and create fourgrams of each sentence.
    
      sentences_list: list of corpus sentences after preprocessing

      retrun list of fourgrams for each sentence
      """
      fourgrams = [ ([ i[0]+"_"+i[1]+"_"+i[2]+"_"+i[3] for i in zip(j.split()[:-1], j.split()[1:],j.split()[2:],j.split()[3:]) ]) for j in sentences_list ]
      return fourgrams

    def readFile(self,fileName):
      """ 
      Read file of dictionary
    
      filename: path of dictionary file.

      retrun list of dictionary
      """
      fileObj = open(fileName, "r") #opens the file in read mode
      words = fileObj.read().splitlines() #puts the file into an array
      fileObj.close()
      return words
    
    def creat_ngram_dict(self,filename):
      """ 
      Read file of ngrams dictinary data to create ngrams from unigram
    
      filename: path of dictionary file.

      retrun dataframe 
      """  
      ngrams = pd.read_csv(filename)
      ngrams['1-grams-refine'] = ngrams['1-grams-refine'].apply(lambda x: x.replace('[','').replace(']','').replace("'","").replace("‘",'').replace("’",'').split(','))
      ngrams['1-grams-refine'] = ngrams['1-grams-refine'].apply(lambda x: [i.strip().lower() for i in x])
      ngrams['2-grams-refine'] = ngrams['1-grams-refine'].apply(lambda x: [(i[0]+"_"+i[1]) for i in zip(x[:-1], x[1:])] )
      ngrams['3-grams-refine'] = ngrams['1-grams-refine'].apply(lambda x: [(i[0]+"_"+i[1]+"_"+i[2]) for i in zip(x[:-1], x[1:],x[2:])] )
      ngrams['4-grams-refine'] = ngrams['1-grams-refine'].apply(lambda x: [(i[0]+"_"+i[1]+"_"+i[2]+"_"+i[3]) for i in zip(x[:-1], x[1:],x[2:],x[3:])] )
      ngrams_df = ngrams.iloc[:,[0,3,7,8,9]]
      # ngrams.head(1)
      return ngrams_df

    def creat_dict_ngrams_list(self,df):
      """ 
      create list of unique ngrams from dataframe columns
    
      df: dataframe name of ngrams of dictionary

      retrun list of unique ngrams from dictionary
      """
      
      ugram = set([ gram  for line in df['1-grams-refine'] for gram in line])
      bgram = set([gram for line in df['2-grams-refine'] for gram in line])
      tgram = set([gram for line in df['3-grams-refine'] for gram in line])
      fgram = set([gram for line in df['4-grams-refine'] for gram in line])
      return ugram,bgram,tgram,fgram
        
    def POS_tag(self,ngrams_dict):
      """ 
    
      ngrams_dict: dataframe column of ngrams dictionary

      retrun ngrams with POS tags
      """  
      
      tagged = nltk.pos_tag(ngrams_dict)

      return tagged
      
    def words_in_corpus(self,grams,dictionary):
      """ 
    
      grams: corpus ngrams (unigrm/bigram/trigram/fourgram)
      dictionary: list of ngrams (unigrm/bigram/trigram/fourgram)

      retrun In vocabulary words , Out of vocabulary words and corpus vocabulary
      """ 
      
      vocabs = {}                #creating dictionary with keys as word  and values as count of word in corpus
      for i in grams:            #loop through each sentence 
        for j in i:              #loop through each word in sentence
          if j not in vocabs:    #if word not in dictionary
            vocabs[j]  = 1       #add word and set count to 1
          else:                  #if word not found in vocab
            vocabs[j] += 1       #increment the count of that word

      iv = []                    #list for words found in corpus
      oov  =  []                 #list for words not found in corpus
      for tok in dictionary:     #loop through the list of dictionary words
        if tok in vocabs:        #if word found in corpus vocabulary
          # print(tok,vocabs[tok])
          iv.append(tok)         #add it to the iv
        else:                    #if not found in corpus vocabulary
          oov.append(tok)        #add it to oov
      return iv,oov,vocabs


    def ngrams_dict(self,mydict,mainclass,ngrams):
      """ 
    
      mydict: dictionary name 
      mainclass: dictionary main classes
      ngrams: list of ngrams column (unigrm/bigram/trigram/fourgram) from ngrams dataframe 

      retrun dictionary with key as main class and value as ngrmas belong to each key
      """ 

      for i,j in zip(mainclass,ngrams): 
        for gram in j:
          if i in mydict:
            mydict[i].append(gram)
            
          else:
            mydict[i] = [gram]
      return mydict

    def save_dict_json(self,filename,dictionary):
      """ 
    
      filename: file path where it will save the json file
      dictionary: dictionary of ngrams

      """ 
      with open(filename, "w") as outfile:
        json.dump(dictionary, outfile, indent=2)

    def load_json_file(self,filename):
      """ 
    
      filename: path of the file where it is saved

      retrun ngrams dictionary 
      """ 
      with open(filename, "r") as fp:
        data = json.load(fp)  
      return data

    def score_evaluation(self,filename,model,models):
      """ 
    
      filename: path for the file where spearman score will be saved
      model: model name for which it will be evaluated
      models: string for model name to be saved

      retrun 
      """     
      similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))

      pearsoncorrelation, pearsonpvalue = similarities[0][0],similarities[0][1]
      spearmancorrelation, spearmanpvalue = similarities[1][0],similarities[1][1]
      oov_ratio = similarities[2]
      simdic = { 
                'pearson' : [pearsoncorrelation,pearsonpvalue],
                'spearman' : [spearmancorrelation, spearmanpvalue],
                'oov_ratio': [oov_ratio]
                
          
      }
      sim = pd.DataFrame(simdic.items(),columns=['Method', 'Correlation'])
      sim['Correlation'], sim['pvalue'] = sim['Correlation'].str
      sim["Model"] = models



      if os.path.exists(filename):
        score = pd.read_csv(filename)
        score = score.append(sim).reset_index(drop=True)  
        score.to_csv(filename,index=False)
      else:
        sim.to_csv(filename,index=False)


    def model_output(self,model,gramsdict,path,models):
      """ 
    
      model: model name for which the we find similar terms and score 
      gramsdict: ngrams list for which the model is trained
      path: path of the file where it will be save

      """ 
      mydict = { }
      for word in gramsdict:
        if word in model.wv.vocab:
          mostSimWord = model.wv.most_similar(word,topn=100)#len(model.wv.vocab))
          mydict[word] = mostSimWord
        else:
          continue



      df = pd.DataFrame(mydict.items(), columns=['Word', 'Similar word'])
      df = df.explode('Similar word').reset_index(drop=True)
      df['Most_smiliar'], df['similarity_scores'] = df['Similar word'].str
      df = df.drop('Similar word', 1)


      w2c = dict()
      for item in model.wv.vocab:
          w2c[item]=model.wv.vocab[item].count


      df2 = pd.DataFrame(w2c.items(), columns=['Most_smiliar','count'])
      df3 = pd.merge(df,df2,on='Most_smiliar',how='left')

      # vector = dict()
      # for item in w2v_model.wv.vocab:
      #     vector[item]=w2v_model.wv[item]

      # df4 = pd.DataFrame(vector.items(), columns=['Most_smiliar','Vector'])
      # df5 = pd.merge(df3,df4,on='Most_smiliar',how='left')


      df3.to_csv(path+models+'.csv',index=False)
      
      
    def word2vec_model(self,ngrams,window,sizes,epochs,min_count):
      """ 
    
      ngrams: corpus ngrams (unigrm/bigram/trigram/fourgram)

      retrun trained word2vec model
      """ 
      w2v_model = Word2Vec(min_count=min_count, window=window,size=sizes,sg=0 )
      w2v_model.build_vocab(ngrams)
      w2v_model.train(ngrams, total_examples=len(ngrams), epochs=epochs)
      
      return w2v_model
      
      

    def glove_models(self,ngrams,window,sizes,epochs):
      """ 
    
      ngrams: corpus ngrams (unigrm/bigram/trigram/fourgram)
      

      retrun trained glove model
      """ 
      # creating a corpus object
      corpus = Corpus() 
      #training the corpus to generate the co occurence matrix which is used in GloVe
      corpus.fit(ngrams, window=window)

      #creating a Glove object which will use the matrix created in the above lines to create embeddings
      #We can set the learning rate as it uses Gradient Descent and number of components
      glove = Glove(no_components=sizes, learning_rate=0.05,)

      #fit over the corpus matrix 
      glove.fit(corpus.matrix, epochs=epochs, no_threads=4, verbose=False)

      # finally we add the vocabulary to the model
      glove.add_dictionary(corpus.dictionary)

      return glove
      
    def glove_results_text(self,glove,filename,dim):
      """ 
      Wrtie glove model text file

      glove: trained glove model
      filename: file path where it will be save.
      dim: dimensionality of model with which it is trained 

      
      """ 
      with open(filename, "w") as f:
          for word in glove.dictionary:
              f.write(word)
              f.write(" ")
              for i in range(0, dim):
                  f.write(str(glove.word_vectors[glove.dictionary[word]][i]))
                  f.write(" ")
              f.write("\n")

    def glove2word2vec_format(self,input_filepath,output_filepath):
      """ 
      Convert glove model to word2vec format

      input_filepath: path of the text file where glove model saved 
      output_filepath: path of the text file where glove model will be save in word2vec format

      retrun trained glove model in word2vec keyed vectors format
      """ 

      glove_input_file = datapath(input_filepath)

      word2vec_output_file = get_tmpfile(output_filepath) 
      glove2word2vec(glove_input_file, word2vec_output_file)

      
      glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file)

      return glove_model
      
    def fasttext_model(self,ngrams,window,sizes,epochs,min_count):
      """ 
      
      
      ngrams: corpus ngrams (unigrams/bigrams/trigrams/fourgrams)

      retrun trained fasttext model
      """ 

      fasttext_model = FastText( window=window, min_count=min_count,sg=1,size=sizes)  # instantiate
      fasttext_model.build_vocab(sentences=ngrams)
      fasttext_model.train(sentences=ngrams, total_examples=len(ngrams), epochs=epochs)  # train

      return fasttext_model
      
    def line_graph(self,filename,title,filepath):
      """ 
      Create line graph for spearman score on different parameters.
      
      filename: path of the file where spearman score file is saved. 
      filepath: path of the .png file where graph will be save.
      title: graph title

      retrun trained glove model in word2vec keyed vectors format
      """ 
      scores = pd.read_excel(filename)
      fig, ax= plt.subplots(figsize=(8,5))


      for key, grp in scores.groupby(['Dimensions']):
          ax = grp.plot(ax=ax, kind='line', x='Window', y='Correlation',  label=key)

      plt.legend(loc='best')
      plt.ylabel("Spearman Correlation")
      plt.title(title)
      if (filepath):
        plt.savefig(filepath, format='png', dpi=150, bbox_inches='tight')
      plt.show()
      
    
    def similarity_word_clusters(self,keys,model):
      """ 
      Create list of similarity score, similar words and embeddings for word2vec and fasttext 
      keys: ngrams dictionary list
      model: trained model name 

      retrun similarity score, similar words and embeddings
      """ 
      similarity_clusters = []
      word_clusters = []
      embedding_clusters = []
            

      for word in keys:
          similarity = []
          words = []
          embeddings = []
          for similar_word, similar in model.most_similar(word, topn=50):
                words.append(similar_word)
                similarity.append(similar)
                embeddings.append(model[similar_word])
          embedding_clusters.append(embeddings)
          similarity_clusters.append(similarity)
          word_clusters.append(words)

      return similarity_clusters,word_clusters,embedding_clusters



    def glove_similarity_word_clusters(self,keys,model):
      """ 
      Create list of similarity score, similar words and embeddings for glove 
      keys: ngrams dictionary list
      model: trained model name

      retrun similarity score, similar words and embeddings
      """ 
      similarity_clusters = []
      word_clusters = []
      embedding_clusters = []
            

      for word in keys:
          similarity = []
          words = []
          embeddings = []
          for similar_word, similar in model.most_similar(word,number=50):
                words.append(similar_word)
                similarity.append(similar)
                embeddings.append(model.word_vectors[model.dictionary[similar_word]])
          embedding_clusters.append(embeddings)
          similarity_clusters.append(similarity)
          word_clusters.append(words)

      return similarity_clusters,word_clusters,embedding_clusters

    #
    def words_in_glove_model(self,model,dictionary):
      """ 
      for glove to check dictionary words present in model vocabulary 
      dictionary: ngrams dictionary list
      model: trained model name

      retrun list of dictionary words that are present in model
      """
      keys = []
      for word in dictionary:
            if word in model.dictionary:
              keys.append(word)
      return keys

     
    def words_in_model(self,model,dictionary):
      """ 
      for word2vec and fasttext to check dictionary words present in model vocabulary 
      dictionary: ngrams dictionary list
      model: trained model name

      retrun list of dictionary words that are present in model
      """
      keys = []
      for word in dictionary:
            if word in model.wv.vocab:
              keys.append(word)
      return keys

 
    def scatter_plot_similar_words(self,title, labels, similarity_clusters, word_clusters, a, filename=None):
      """ 
      Create multiple subpolts
      Scatter plot for similarity score and similar words and saved it into .png format
      title: graph title
      labels: list of dictionary words that are present in model
      similarity_clusters: list of similarity scores
      word_clusters: list of similar words.
      filename: path of file where graph will be save
     
      """    
      fig, ax= plt.subplots((int(len(labels)/2)),2,figsize=(25, 15))
      # plt.figure(4,2,figsize=(16, 9))
      fig.subplots_adjust(hspace = 3, wspace=.1)
      
      colors = cm.rainbow(np.linspace(0, 1, len(labels)))
      for label, similarity, words, color , axs in zip(labels, similarity_clusters, word_clusters, colors,ax.flatten()):
          x = words
          y = similarity
          # plt.plot(x, y,marker="o", linestyle="",   label=label,c=color)
          axs.scatter(x, y, c=color.reshape(1,-1), alpha=a, label=label, )
          # for i, word in enumerate(words):
          #     plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
          #                  textcoords='offset points', ha='right', va='bottom', size=8)
          axs.tick_params(axis='x', rotation=90)
          axs.legend(loc='upper right')
          # axs.set_yticks([0.0,0.2,0.4,0.6,0.8]) 
      # plt.xticks(rotation=90, ha="right")
      # plt.legend(loc='upper right')
      # plt.title(title)
      # plt.grid(True)
      fig.suptitle(title)
      fig.text(0.5, 0.01, 'Most Similar words', ha='center')
      fig.text(0.04, 0.5, 'Cosine Similarity', va='center', rotation='vertical')
      
      if filename:
          plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
      # plt.show()

   
    def plot_similar_words(self,title, labels, similarity_clusters, word_clusters, a, filename=None):
      """ 
      Create one plot
      Scatter plot for similarity score and similar words and saved it into .png format
      title: graph title
      labels: list of dictionary words that are present in model
      similarity_clusters: list of similarity scores
      word_clusters: list of similar words.
      filename: path of file where graph will be save
     
      """   

      plt.figure(figsize=(16, 9))
      colors = cm.rainbow(np.linspace(0, 1, len(labels)))
      for label, similarity, words, color  in zip(labels, similarity_clusters, word_clusters, colors,):
        x = words
        y = similarity
        plt.scatter(x, y, c=color.reshape(1,-1), alpha=a, label=label, )
      plt.xticks(rotation=90, ha="right")
      plt.legend(loc='upper right')
      plt.title(title)
      plt.xlabel('Most Similar words')
      plt.ylabel('Cosine Similarity')
        
      if filename:
              plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
      plt.show()

    def tsne(self,embedding_cluster):
      """ 
      Convert the embedding using Tsne for 2D plot 

      embedding_cluster: list of embedding vector
      
      retrun embeddinf list on for 2D plot
      """ 
      embedding_cluster = np.array(embedding_cluster)
      n, m, k = embedding_cluster.shape
      tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
      embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_cluster.reshape(n * m, k))).reshape(n, m, 2)

      return embeddings_en_2d

    def tsne_plot_similar_words(self,title, labels, embedding_clusters, word_clusters, a, filename=None):
      """ 
      Create embedding plot
      Scatter plot for similarity score and similar words and saved it into .png format
      title: graph title
      labels: list of dictionary words that are present in model
      embeddigs_clusters: list of similarity scores
      word_clusters: list of similar words.
      filename: path of file where graph will be save
     
      """  
      plt.figure(figsize=(16, 9))
      colors = cm.rainbow(np.linspace(0, 1, len(labels)))
      for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
          x = embeddings[:, 0]
          y = embeddings[:, 1]
          plt.scatter(x, y, c=color.reshape(1,-1), alpha=a, label=label)
          for i, word in enumerate(words):
              plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                            textcoords='offset points', ha='right', va='bottom', size=8)
      plt.legend(loc='lower right')
      plt.title(title)
      plt.grid(True)
      if filename:
          plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
      plt.show()
     


