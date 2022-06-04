from sklearn.metrics import precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import *
import pandas as pd
import numpy as np
import unidecode

class TagGenerator:
    def __init__(self, 
            only_nouns=True, 
            to_ascii=True, 
            stemmer=None, 
            vectorization="tf-idf", 
            generate_bigrams = True,
            semantic_field_size=40, 
            encoding="utf-8"):

        self.only_nouns = only_nouns
        self.to_ascii = to_ascii
        self.stemmer = stemmer
        self.vectorization = vectorization
        self.generate_bigrams = generate_bigrams
        self.semantic_field_size = semantic_field_size
        self.encoding = encoding
        self.vectorizer = None
        self.doc_terms_matrix = None
        self.stem_dict = {}

    def get_stemmer(self):
        if self.stemmer is None:
            return None
        elif self.stemmer == "porter":
            return porter.PorterStemmer()
        elif self.stemmer == "snowball":
            return snowball.SnowballStemmer('english')
        elif self.stemmer == "wordnet":
            return wordnet.WordNetLemmatizer()
        else:
            raise Exception("Invalid stemmer: '" + str(self.stemmer) + "'. Valid values are 'porter', 'snowball', 'wordnet' or None.")

    def stem_word(self, stemmer, word):
        if self.stemmer is None:
            return word
        elif self.stemmer == "wordnet": # it is a lemmatizer, not a stemmer
            return stemmer.lemmatize(word)
        else:
            return stemmer.stem(word)

    def get_nouns(self, words):
        nouns = []
        print(pos_tag(words))
        for word, tag in pos_tag(words):
            if tag.startswith("N"):
                nouns.append(word)
        return nouns

    def preprocess(self, document):
        stemmer = self.get_stemmer()
        text = document.lower()

        # converts non-ascii characters to ascii characters
        text = unidecode.unidecode(text) if self.to_ascii else text

        # tokenizes the text, considering words with only two or more characters
        words = RegexpTokenizer(r'[a-zA-Z]{2,}').tokenize(text)

        # removes stopwords
        words = [word for word in words if word not in stopwords.words('english')]

        # considers only nouns and discards everything else
        words = self.get_nouns(words) if self.only_nouns else words

        # stems words according to the stemmer chosen (if any)
        stemmed_words = [self.stem_word(stemmer, word) for word in words]

        # inverted list to help keeping track of which words gave origin to which stems
        self.generate_stem_dict(stemmed_words, words)

        return stemmed_words, words

    def generate_stem_dict(self, stemmed_words, words):
        #inverted_list = {}

        for i in range(len(stemmed_words)):
            if stemmed_words[i] not in self.stem_dict:
                self.stem_dict[stemmed_words[i]] = [words[i]]
            else:
                if words[i] not in self.stem_dict[stemmed_words[i]]:
                    self.stem_dict[stemmed_words[i]].append(words[i])

        #return inverted_list



    def generate(self, documents):
        preprocessed_documents = []

        for document in documents:
            stemmed_words, words = self.preprocess(document)
            preprocessed_documents.append(" ".join(stemmed_words))

        print("len documents", len(documents), "len preprocessed_documents", len(preprocessed_documents))

        if self.vectorization == "tf":
            self.vectorizer = CountVectorizer()
        elif self.vectorization == "tf-idf":
            self.vectorizer = TfidfVectorizer()
        else:
            raise Exception("Invalid stemmer: '" + str(self.vectorization) + "'. Valid values are 'tf' and 'tf-idf'.")

        self.doc_terms_matrix = pd.DataFrame(
            self.vectorizer.fit_transform(preprocessed_documents).toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )
        
        print(self.doc_terms_matrix.shape)
        print(self.doc_terms_matrix)
        print(self.vectorizer.get_feature_names_out())
        print(self.vectorizer.vocabulary_)
        print("preprocessed_documents", preprocessed_documents)

        #print(pd.DataFrame(self.doc_terms_matrix.iloc[1,:]).T.sort_values(by=1, axis=1, ascending=False))#.sort_values(axis=1))
        self.generate_tags()



    def generate_tags(self):
        document_abstract_tags = []

        # obtaining document abstract tags
        for index, row in self.doc_terms_matrix.iterrows():                         # doc_terms matrix is documents (lines) x terms (columns)
            #print(row, type(row))
            print("index", index)
            #abstract_tags = pd.DataFrame(self.doc_terms_matrix.iloc[index,:]).T.sort_values(by=index, axis=1, ascending=False)
            abstract_tags = pd.DataFrame(row).T.sort_values(by=index, axis=1, ascending=False)
            print(abstract_tags)
            num_non_zero = int(abstract_tags[abstract_tags.columns].gt(0).sum(axis=1))
            #print(num_non_zero)
            abstract_tags = abstract_tags.columns[:min(num_non_zero, self.semantic_field_size)].to_list()
            print(abstract_tags)

            document_abstract_tags.append(abstract_tags)


        U, S, VH = np.linalg.svd(self.doc_terms_matrix, full_matrices=False)

        print(U.shape, S.shape, VH.shape)

        print("U\n", U)
        print("S\n", S)
        print(np.diag(S))
        print("VH\n", VH)

        S_reduced = np.zeros((len(S), len(S)), dtype=float)
        S_tmp = np.diag(S[:self.semantic_field_size]) # if semantic_field_size > len(S), the slicing will consider up to len(S), without raising an error
        S_reduced[:len(S_tmp), :len(S_tmp)] = S_tmp
        print(S.shape, S_tmp.shape, S_reduced.shape, S_reduced)


        reconstructed_matrix =pd.DataFrame(U @ S_reduced @ VH, columns=self.doc_terms_matrix.columns)
        print(self.doc_terms_matrix.shape, reconstructed_matrix.shape)

        print(reconstructed_matrix)

        term_correlation = reconstructed_matrix.corr()

        print(term_correlation)

        


            #break

        
        
    #def 

#stemmed, words, inverted_list = TagGenerator().preprocess("It was a warm morning, with no clouds in the sky, when a thunder struck Guilherme's head. How was that possible? That's simple. It was just Thor saying hello. And, yes, Thor is simply a troll and is always on some cloud, waiting for an opportunity to perform some pranks. He is a trolly prankster.")
#print(stemmed)
#print(words)
#print(inverted_list)
TagGenerator().generate([
    "It was a warm morning, with no clouds in the sky, when a thunder struck Guilherme's head. How was that possible? That's simple. It was just Thor saying hello. And, yes, Thor is simply a troll and is always on some cloud, waiting for an opportunity to perform some pranks. He is a trolly prankster.",
    "Guilherme is a post grad student at Federal University of Rio de Janeiro (UFRJ). He currently lives in Praça Seca, Rio de Janeiro, and is 30 years old.",
    " Earth. The world we live in. It is our home, and the home of Guilherme Caeiro de Mattos, an post grad student who lives in country called Brazil. Specifically in a city called Rio de Janeiro, that is hot as hell.",
    "It is a saying commonly told among practitioners of martial arts. It says \"健全なる魂は健全なる精神と健全なる肉体に宿る\"."
])