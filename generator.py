from sklearn.metrics import precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import *
import pandas as pd
import numpy as np
import unidecode
import itertools

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
            raise Exception("Invalid stemmer: \"%s\". Valid values are 'porter', 'snowball', 'wordnet' or None." % (str(self.stemmer)))

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

    def get_synonyms(self, word):
        pass

    def preprocess(self, document):
        stemmer = self.get_stemmer()
        text = document.lower()

        # converts non-ascii characters to ascii characters
        text = unidecode.unidecode(text).lower() if self.to_ascii else text

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

    def reverse_stem(self, stemmed_word):
        print("reverse_stem", stemmed_word)
        if stemmed_word in self.stem_dict:
            # returns the first word that led to that
            # stem, since it is not possible to map 
            # that stem to the exact word if multiple
            # words led to it.
            return self.stem_dict[stemmed_word][0] 
        else:
            raise Exception("Attemp to reverse unknown stem \"%s\". Did this stem really originate from a word present in the original set of documents?" % (stemmed_word))

    def unstem(self, documents):
        print("unstem", documents)
        unstemmed_words = []
        for document_words in documents:
             unstemmed_words.append([self.reverse_stem(stemmed_word) for stemmed_word in document_words])
        return unstemmed_words

    def preprocess_documents(self, documents):
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
            raise Exception("Invalid vectorization: \"%s\". Valid values are 'tf' and 'tf-idf'." % (str(self.vectorization)))

        self.doc_terms_matrix = pd.DataFrame(
            self.vectorizer.fit_transform(preprocessed_documents).toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )
        
        print(self.doc_terms_matrix.shape)
        print(self.doc_terms_matrix)
        print(self.vectorizer.get_feature_names_out())
        print(self.vectorizer.vocabulary_)
        print("preprocessed_documents", preprocessed_documents)


    def generate(self, documents, root=None):
        self.preprocess_documents(documents)
        document_abstract_tags = self.get_document_abstract_tags()
        set_summary_tags = self.get_set_summary_tags(root)

        document_abstract_tags = self.unstem(document_abstract_tags)
        set_summary_tags = self.unstem([set_summary_tags])[0]

        return document_abstract_tags, set_summary_tags


    def get_document_abstract_tags(self):
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

        return document_abstract_tags

    def method_1(self, root):
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


        reconstructed_matrix = pd.DataFrame(U @ S_reduced @ VH, columns=self.doc_terms_matrix.columns)
        print(self.doc_terms_matrix.shape, reconstructed_matrix.shape)

        print(reconstructed_matrix)

        term_correlation = reconstructed_matrix.corr()
        print(term_correlation)

        current_term = self.stem_word(self.get_stemmer(), root.strip().lower())

        print(current_term)

        print("stem_dict", self.stem_dict)

        candidates = term_correlation[current_term][term_correlation[current_term] > 0]
        
        if len(candidates) == 0:
            print("No candidates")
            return [root]

        candidates = pd.DataFrame(candidates).sort_values(by=current_term, axis=0, ascending=False)

        set_summary_tags = candidates[current_term].iloc[0:min(len(candidates), self.semantic_field_size)].index.to_list()

        print(candidates, set_summary_tags)


        #set_summary_tags = []

        #for i in range(self.semantic_field_size - 1):
        #    set_summary_tags.append(current_term)

        #    candidates = term_correlation[current_term][term_correlation[current_term] > 0]

        #    if len(candidates) == 0:
        #        print("No more candidates")
        #        break

        #    candidates = pd.DataFrame(candidates).sort_values(by=current_term, axis=0, ascending=False)
        #    print(candidates, set_summary_tags)

        #    next_term = None

        #    for index, row in candidates.iterrows():
        #        if index not in set_summary_tags:
        #            next_term = index
        #            break

        #    if (next_term != current_term) and (next_term is not None):
        #        current_term = next_term
        #    else:
        #        break

        return set_summary_tags

    def method_2(self):
        set_summary_tags = []

    def get_set_summary_tags(self, root=None):
        set_summary_tags = []

        if (root is not None) and (root.strip() != ""):
            if root not in list(itertools.chain.from_iterable(self.stem_dict.values())):
                print(root, self.stem_dict.values())
                print("Root term \"%s\" not found. Generating set summary tags based on method 2." % str(root))
                return self.get_set_summary_tags()

            return self.method_1(root)
        else:
            return self.method_2()


#stemmed, words, inverted_list = TagGenerator().preprocess("It was a warm morning, with no clouds in the sky, when a thunder struck Guilherme's head. How was that possible? That's simple. It was just Thor saying hello. And, yes, Thor is simply a troll and is always on some cloud, waiting for an opportunity to perform some pranks. He is a trolly prankster.")
#print(stemmed)
#print(words)
#print(inverted_list)
TagGenerator().generate([
    "It was a warm morning, with no clouds in the sky, when a thunder struck Guilherme's head. How was that possible? That's simple. It was just Thor saying hello. And, yes, Thor is simply a troll and is always on some cloud, waiting for an opportunity to perform some pranks. He is a trolly prankster.",
    "Guilherme is a post grad student at Federal University of Rio de Janeiro (UFRJ). He currently lives in Praça Seca, Rio de Janeiro, and is 30 years old.",
    " Earth. The world we live in. It is our home, and the home of Guilherme Caeiro de Mattos, an post grad student who lives in country called Brazil. Specifically in a city called Rio de Janeiro, that is hot as hell.",
    "It is a saying commonly told among practitioners of martial arts. It says \"健全なる魂は健全なる精神と健全なる肉体に宿る\"."
], "rio")