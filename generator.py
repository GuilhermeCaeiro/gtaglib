from sklearn.metrics import precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import *
from wordcloud import WordCloud, ImageColorGenerator
import pandas as pd
import numpy as np
import unidecode
import itertools
import os

class TagGenerator:
    def __init__(self, 
            only_nouns=True, 
            to_ascii=True, 
            stemmer=None, 
            use_tfidf=True, 
            generate_bigrams = False,
            semantic_field_size=40, 
            encoding="utf-8"): # unused

        self.only_nouns = only_nouns
        self.to_ascii = to_ascii
        self.stemmer = stemmer
        self.use_tfidf = use_tfidf
        self.generate_bigrams = generate_bigrams
        self.semantic_field_size = semantic_field_size
        self.encoding = encoding
        self.vectorizer = None
        self.doc_terms_matrix = None
        self.tf_matrix = None
        self.tfidf_matrix = None
        self.stemmer_object = None
        self.stem_dict = {}
        self.occurrence_count = {}

    def instantiate_stemmer(self):
        if self.stemmer is None:
            self.stemmer_object = None
        elif self.stemmer == "porter":
            self.stemmer_object = porter.PorterStemmer()
        elif self.stemmer == "snowball":
            self.stemmer_object = snowball.SnowballStemmer('english')
        elif self.stemmer == "wordnet":
            self.stemmer_object = wordnet.WordNetLemmatizer()
        else:
            raise Exception("Invalid stemmer: \"%s\". Valid values are 'porter', 'snowball', 'wordnet' or None." % (str(self.stemmer)))

    def stem_word(self, word):
        if self.stemmer is None:
            return word
        elif self.stemmer == "wordnet": # it is a lemmatizer, not a stemmer
            return self.stemmer_object.lemmatize(word)
        else:
            return self.stemmer_object.stem(word)

    def get_nouns(self, words):
        nouns = []
        #print(pos_tag(words))
        for word, tag in pos_tag(words):
            if tag.startswith("N"):
                nouns.append(word)
        return nouns

    def get_synonyms(self, word):
        pass

    def add_occurence_count(self, words):
        words = set(words) # counts each word only once per document
        for word in words:
            if word not in self.occurrence_count:
                self.occurrence_count[word] = 1
            else:
                self.occurrence_count[word] += 1

    def preprocess(self, document):
        self.instantiate_stemmer()
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
        stemmed_words = [self.stem_word(word) for word in words]

        # inverted list to help keeping track of which words gave origin to which stems
        self.generate_stem_dict(stemmed_words, words)

        self.add_occurence_count(stemmed_words) 

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
        #print("reverse_stem", stemmed_word)
        splitted_word = stemmed_word.split(" ")

        if stemmed_word in self.stem_dict:
            # returns the first word that led to that
            # stem, since it is not possible to map 
            # a stem to the exact word if multiple
            # words led to it.
            return self.stem_dict[stemmed_word][0]
        elif len(splitted_word) == 2:
            if (splitted_word[0] in self.stem_dict) and (splitted_word[1] in self.stem_dict):
                return self.stem_dict[splitted_word[0]][0] + " " + self.stem_dict[splitted_word[1]][0]
            else:
                raise Exception("Attemp to reverse unknown stem \"%s\". Did this stem really originate from a word present in the original set of documents?" % (stemmed_word))
        else:
            raise Exception("Attemp to reverse unknown stem \"%s\". Did this stem really originate from a word present in the original set of documents?" % (stemmed_word))

    def unstem(self, documents):
        #print("unstem", documents)
        unstemmed_words = []
        for document_words in documents:
             unstemmed_words.append([self.reverse_stem(stemmed_word) for stemmed_word in document_words])
        return unstemmed_words

    def preprocess_documents(self, documents):
        preprocessed_documents = []

        for document in documents:
            stemmed_words, words = self.preprocess(document)
            preprocessed_documents.append(" ".join(stemmed_words))

        #print("len documents", len(documents), "len preprocessed_documents", len(preprocessed_documents))
        ngram_range = (1, 2) if self.generate_bigrams else (1, 1) 
        self.vectorizer = CountVectorizer(ngram_range=ngram_range)
        self.tf_matrix = pd.DataFrame(
            self.vectorizer.fit_transform(preprocessed_documents).toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )

        if self.use_tfidf:
            self.tfidf_matrix = pd.DataFrame(
                TfidfTransformer().fit_transform(self.tf_matrix).toarray(),
                columns=self.vectorizer.get_feature_names_out()
            )
            self.doc_terms_matrix = self.tfidf_matrix
        else:
            self.doc_terms_matrix = self.tf_matrix

        
        #print(self.tf_matrix.shape)
        #print(self.tf_matrix)
        #print(self.tfidf_matrix)
        #print(self.vectorizer.get_feature_names_out())
        #print(self.vectorizer.vocabulary_)
        #print("preprocessed_documents", preprocessed_documents)



    def get_document_abstract_tags(self):
        document_abstract_tags = []

        # obtaining document abstract tags
        for index, row in self.doc_terms_matrix.iterrows():                         # doc_terms matrix is documents (lines) x terms (columns)
            #print(row, type(row))
            #print("index", index)
            #abstract_tags = pd.DataFrame(self.doc_terms_matrix.iloc[index,:]).T.sort_values(by=index, axis=1, ascending=False)
            abstract_tags = pd.DataFrame(row).T.sort_values(by=index, axis=1, ascending=False)
            #print(abstract_tags)
            num_non_zero = int(abstract_tags[abstract_tags.columns].gt(0).sum(axis=1))
            #print(num_non_zero)
            abstract_tags = abstract_tags.columns[:min(num_non_zero, self.semantic_field_size)].to_list()
            #print(abstract_tags)

            document_abstract_tags.append(abstract_tags)

        return document_abstract_tags

    def get_set_summary_tags_method_1(self, root):
        U, S, VH = np.linalg.svd(self.tf_matrix, full_matrices=False)

        #print(U.shape, S.shape, VH.shape)

        #print("U\n", U)
        #print("S\n", S)
        #print(np.diag(S))
        #print("VH\n", VH)

        S_reduced = np.zeros((len(S), len(S)), dtype=float)
        S_tmp = np.diag(S[:self.semantic_field_size]) # if semantic_field_size > len(S), the slicing will consider up to len(S), without raising an error
        S_reduced[:len(S_tmp), :len(S_tmp)] = S_tmp
        #print(S.shape, S_tmp.shape, S_reduced.shape, S_reduced)


        reconstructed_matrix = pd.DataFrame(U @ S_reduced @ VH, columns=self.tf_matrix.columns) # "@" multiplies np arrays as matrices 
        #print(self.tf_matrix.shape, reconstructed_matrix.shape)

        #print(reconstructed_matrix)

        term_correlation = reconstructed_matrix.corr()
        #print(term_correlation)

        current_term = self.stem_word(root.strip().lower())

        #print(current_term)

        #print("stem_dict", self.stem_dict)

        candidates = term_correlation[current_term][term_correlation[current_term] > 0]
        
        if len(candidates) == 0:
            print("No candidates")
            return [root]

        candidates = pd.DataFrame(candidates).sort_values(by=current_term, axis=0, ascending=False)

        set_summary_tags = candidates[current_term].iloc[0:min(len(candidates), self.semantic_field_size)].index.to_list()

        #print(candidates, set_summary_tags)


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

    def get_set_summary_tags_method_2(self):
        #print("get_set_summary_tags_method_2", self.occurrence_count)
        set_summary_tags = sorted(self.occurrence_count, key=self.occurrence_count.get, reverse=True)
        return set_summary_tags[:min(self.semantic_field_size, len(set_summary_tags))]

    def get_set_summary_tags(self, method, root):
        set_summary_tags = []

        if method == 1:
            if root not in list(itertools.chain.from_iterable(self.stem_dict.values())):
                #print(root, self.stem_dict.values())
                raise Exception("Root term \"%s\" not found." % str(root))

            return self.get_set_summary_tags_method_1(root)
        elif method == 2:
            return self.get_set_summary_tags_method_2()
        else:
            raise Exception("\"method\" must be 1 or 2.")

    def get_differential_tags(self, method, document_abstract_tags, set_summary_tags):
        if method == 1:
            return document_abstract_tags
        elif method == 2:
            document_differential_tags = []

            for document_abstract_tag_list in document_abstract_tags:
                document_differential_tags.append([term for term in document_abstract_tag_list if term not in set_summary_tags])

            return document_differential_tags
        else:
            raise Exception("\"method\" must be 1 or 2.")


    def generate(self, documents, method=2, root=None):
        self.preprocess_documents(documents)
        document_abstract_tags = self.get_document_abstract_tags()
        #print("generate", method, root)
        set_summary_tags = self.get_set_summary_tags(method, root)
        document_differential_tags = self.get_differential_tags(method, document_abstract_tags, set_summary_tags)

        document_abstract_tags = self.unstem(document_abstract_tags)
        set_summary_tags = self.unstem([set_summary_tags])[0]
        document_differential_tags = self.unstem(document_differential_tags)

        return document_abstract_tags, set_summary_tags, document_differential_tags

    def retrieve_word_frequency(self, document_tags, document_number):
        document_word_frequency = {}
        max_frequency = 0
        document_line = self.tf_matrix.iloc[document_number]

        #print("retrieve_word_frequency", document_tags, document_number)

        if len(document_tags) == 0:
            return {}

        for tag in document_tags:
            frequency = document_line[self.stem_word(tag)]
            document_word_frequency[tag] = frequency

        max_frequency = max(document_word_frequency.values())

        # normalization
        for tag in document_word_frequency:
            document_word_frequency[tag] = document_word_frequency[tag]/max_frequency

        return document_word_frequency

    def retrieve_word_frequency_per_document(self, tag_lists):
        #print("retrieve_word_frequency_per_document", tag_lists, len(tag_lists))
        results = []

        for i in range(len(tag_lists)):
            document_tags = tag_lists[i]
            tag_frequencty_list = self.retrieve_word_frequency(document_tags, i)
            results.append(tag_frequencty_list)

        return results

    def retrieve_set_word_frequency(self, tags):
        results = {}
        max_frequency = 0

        #print("retrieve_set_word_frequency", tags)

        if len(tags) == 0:
            return {}

        for tag in tags:
            frequency = self.tf_matrix[self.stem_word(tag)].sum()
            results[tag] = frequency

        max_frequency = max(results.values())

        for tag in results:
            results[tag] = results[tag]/max_frequency

        return results

    def create_tag_cloud_image(self, data, base_name, outputdir):
        for i in range(len(data)):
            if len(data[i]) == 0:
                continue
                
            word_cloud = WordCloud(
                #font_path="fonts/Georgia.ttf", 
                width=500, 
                height=500, 
                max_words=self.semantic_field_size, 
                #stopwords=stop
            )
            word_cloud.fit_words(data[i])
            word_cloud.to_file(os.path.join(outputdir, base_name) + "_" + str(i) + ".png")



    def generate_tag_cloud(self, documents, method=2, root=None, generate_atc=False, outputdir=""):
        document_abstract_tags, set_summary_tags, document_differential_tags = self.generate(documents, method, root)

        if generate_atc:
            #print("document_abstract_tags")
            document_abstract_tags = self.retrieve_word_frequency_per_document(document_abstract_tags)
            self.create_tag_cloud_image(document_abstract_tags, "document_abstract_tags", outputdir)
            #print("generate_tag_cloud document_abstract_tags", document_abstract_tags)

        #print("set_summary_tags")
        set_summary_tags = self.retrieve_set_word_frequency(set_summary_tags)
        #print("document_differential_tags")
        document_differential_tags = self.retrieve_word_frequency_per_document(document_differential_tags)
        #print("generate_tag_cloud set_summary_tags", set_summary_tags)
        #print("generate_tag_cloud document_differential_tags", document_differential_tags)

        self.create_tag_cloud_image([set_summary_tags], "set_summary_tags", outputdir)
        self.create_tag_cloud_image(document_differential_tags, "document_differential_tags", outputdir)




        


#stemmed, words, inverted_list = TagGenerator().preprocess("It was a warm morning, with no clouds in the sky, when a thunder struck Guilherme's head. How was that possible? That's simple. It was just Thor saying hello. And, yes, Thor is simply a troll and is always on some cloud, waiting for an opportunity to perform some pranks. He is a trolly prankster.")
#print(stemmed)
#print(words)
#print(inverted_list)
a, b, c = TagGenerator(semantic_field_size=40).generate([
    "It was a warm morning, with no clouds in the sky, when a thunder struck Guilherme's head. How was that possible? That's simple. It was just Thor saying hello. And, yes, Thor is simply a troll and is always on some cloud, waiting for an opportunity to perform some pranks. He is a trolly prankster.",
    "Guilherme is a post grad student at Federal University of Rio de Janeiro (UFRJ). He currently lives in Praça Seca, Rio de Janeiro, and is 30 years old.",
    " Earth. The world we live in. It is our home, and the home of Guilherme Caeiro de Mattos, an post grad student who lives in country called Brazil. Specifically in a city called Rio de Janeiro, that is hot as hell.",
    "It is a saying commonly told among practitioners of martial arts. It says \"健全なる魂は健全なる精神と健全なる肉体に宿る\"."
], 2, "rio")


print(a)
print(b)
print(c)

TagGenerator(semantic_field_size=40).generate_tag_cloud([
    "It was a warm morning, with no clouds in the sky, when a thunder struck Guilherme's head. How was that possible? That's simple. It was just Thor saying hello. And, yes, Thor is simply a troll and is always on some cloud, waiting for an opportunity to perform some pranks. He is a trolly prankster.",
    "Guilherme is a post grad student at Federal University of Rio de Janeiro (UFRJ). He currently lives in Praça Seca, Rio de Janeiro, and is 30 years old.",
    " Earth. The world we live in. It is our home, and the home of Guilherme Caeiro de Mattos, an post grad student who lives in country called Brazil. Specifically in a city called Rio de Janeiro, that is hot as hell.",
    "It is a saying commonly told among practitioners of martial arts. It says \"健全なる魂は健全なる精神と健全なる肉体に宿る\"."
], 2, "rio")