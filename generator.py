from sklearn.metrics import precision_score, recall_score
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import *
import unidecode

class TagGenerator:
    def __init__(self, only_nouns=True, to_ascii=True, stemmer="porter", vectorizer="tf-idf", generate_bigrams = True, encoding="utf-8"):
        self.only_nouns = only_nouns
        self.to_ascii = to_ascii
        self.stemmer = stemmer
        self.vectorizer = vectorizer
        self.generate_bigrams = generate_bigrams
        self.encoding = encoding

    def get_stemmer(self):
        if self.stemmer.lower() is None:
            return None
        elif self.stemmer.lower() == "porter":
            return porter.PorterStemmer()
        elif self.stemmer.lower() == "snowball":
            return snowball.SnowballStemmer('english')
        elif self.stemmer.lower() == "wordnet":
            return wordnet.WordNetLemmatizer()
        else:
            raise Exception("Invalid stemmer: " + self.stemmer)

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
        inverted_list = self.generate_inverted_list(stemmed_words, words)

        return stemmed_words, words, inverted_list

    def generate_inverted_list(self, stemmed_words, words):
        inverted_list = {}

        for i in range(len(stemmed_words)):
            if stemmed_words[i] not in inverted_list:
                inverted_list[stemmed_words[i]] = [words[i]]
            else:
                if words[i] not in inverted_list[stemmed_words[i]]:
                    inverted_list[stemmed_words[i]].append(words[i])

        return inverted_list



    def generate(self, documents):
        preprocessed_documents = []

        for document in documents:
            preprocessed_documents.append(self.preprocess(document))
        
    #def 

stemmed, words, inverted_list = TagGenerator().preprocess("It was a warm morning, with no clouds in the sky, when a thunder struck Guilherme's head. How was that possible? That's simple. It was just Thor saying hello. And, yes, Thor is simply a troll and is always on some cloud, waiting for an opportunity to perform some pranks. He is a trolly prankster.")
print(stemmed)
print(words)
print(inverted_list)