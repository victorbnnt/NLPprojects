import re
import string
import unicodedata
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize


# Removing these words from stopword lists
negative_words = ['aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
                  'doesn', "doesn't", 'don', "don't", 'hadn', "hadn't", 'hasn',
                  "hasn't", 'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't",
                  'mustn', "mustn't", 'needn', "needn't", 'no', 'nor', 'not', 'shouldn',
                  "shouldn't", 'wasn', "wasn't", 'weren', "weren't", "won't", 'wouldn',
                  "wouldn't", "are", "could", "did", "does", "do", "have", "has", "is",
                  "might", "must", "need", "should", "would", "will"]

# Adding these words in stopword lists
included_words = ["rt"]


class text_cleaner():

    def __init__(self,
                 df=None,
                 label=None,
                 more_punc=["’"],
                 stopwords_locale="english",
                 stopwords=nltk_stopwords,
                 excluded_sw=[],
                 included_sw=included_words,
                 lemmatizer=WordNetLemmatizer(),
                 lem_param=['a', 'r', 'n', 'v']):
        """
            text preprocessing
            add ’ to defaut punctuation list
            default stopword list is the one from nltk library
            default stopword locale is english
            default punctuation list is the one from string library
            default lemmatizer is WordNetLemmatizer
        """
        self.df = df
        self.label = label
        self.base_punc = string.punctuation
        self.more_punc = more_punc
        try:
            self.stopwords = stopwords.words(stopwords_locale)
        except:
            self.stopwords = stopwords
        self.stopwords_locale = stopwords_locale
        self.excluded_sw = excluded_sw
        self.included_sw = included_sw
        self.lemmatizer = lemmatizer
        self.lem_param = lem_param

    def lowerize(self):
        """ text lowercase
            removes \n
            removes \t
            removes \r """
        self.df[self.label] = self.df[self.label].str.lower()
        self.df[self.label] = self.df[self.label].apply(lambda x: x.replace("\n", " "))
        self.df[self.label] = self.df[self.label].apply(lambda x: x.replace("\r", " "))
        self.df[self.label] = self.df[self.label].apply(lambda x: x.replace("\t", " "))

    def remove_emails(self):
        """ This function removes email adresses
            inputs:
             - text """
        self.df[self.label] = self.df[self.label].apply(lambda x: re.sub(r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])""", " ", x))

    def remove_mentions(self):
        """ This function removes mentions (Twitter - starting with @) from texts
            inputs:
             - text """
        self.df[self.label] = self.df[self.label].apply(lambda x: re.sub(r"@([a-zA-Z0-9_.-]{1,100})", " ", x))

    def remove_hashtags(self):
        """ This function removes hashtags
            inputs:
             - text """
        self.df[self.label] = self.df[self.label].apply(lambda x: re.sub(r"#\w+", " ", x))

    def remove_hyperlinks(self):
        """ This function removes hyperlinks from texts
            inputs:
             - text """
        self.df[self.label] = self.df[self.label].apply(lambda x: re.sub(r"http\S+", " ", x))

    def remove_html_tags(self):
        """ This function removes html tags from texts
            inputs:
             - text """
        self.df[self.label] = self.df[self.label].apply(lambda x: re.sub(r"<.*?>", " ", x))

    def remove_numbers(self):
        """ This function removes numbers from a text
            inputs:
             - text """
        self.df[self.label] = self.df[self.label].apply(lambda x: re.sub(r"\d+", " ", x))

    def encode_unknown(self):
        """ This function encodes special caracters """
        self.df[self.label] = self.df[self.label].apply(lambda x: unicodedata.normalize("NFD", x).encode('ascii', 'ignore').decode("utf-8"))

    def clean_punctuation_no_accent(self):
        """ This function removes punctuation and accented characters from texts in a dataframe
            To be appplied to languages that have no accents, ex: english
        """
        self.df[self.label] = self.df[self.label].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))

    def clean_punctuation(self):
        """ This function removes punctuation from texts in a dataframe """
        self.df[self.label] = self.df[self.label].apply(lambda x: self.remove_punctuation(x))

    def remove_stop_words(self, text):
        """ This function removes stop words from a text
            inputs:
             - stopword list
             - text """

        # update stopword list
        sw_update = set.union(set(self.stopwords), set(self.included_sw)) - set(self.excluded_sw)

        # prepare new text
        text_splitted = text.split(" ")
        text_new = list()

        # loop
        for word in text_splitted:
            if word not in sw_update:
                text_new.append(word)
        return " ".join(text_new)

    def clean_stopwords(self):
        """ This function removes stopwords """
        self.df[self.label] = self.df[self.label].apply(lambda x: self.remove_stop_words(x))

    def more_cleaning(self):
        """ This function
         1) removes remaining one-letter words and two letters words
         2) replaces multiple spaces by one single space
         3) drop empty lines """
        self.df[self.label] = self.df[self.label].apply(lambda x: re.sub(r'\b\w{1,2}\b', " ", x))
        self.df[self.label] = self.df[self.label].apply(lambda x: re.sub(r"[ \t]{2,}", " ", x))
        self.df[self.label] = self.df[self.label].apply(lambda x: x if len(x) != 1 else '')
        self.df[self.label] = self.df[self.label].apply(lambda x: np.nan if x == '' else x)
        self.df = self.df.dropna(subset=[self.label], axis=0).reset_index(drop=True).copy()

    def lemmatize_one_text(self, text):
        """ This function lemmatizes words in text (it changes word to most close root word)
            inputs:
             - lemmatizer
             - text """

        # initialize lemmatizer
        lemmatizer = self.lemmatizer

        # tags
        lem_tags = self.lem_param

        # prepare new text
        text_splitted = text.split(" ")
        text_new = list()

        # change bool
        changed = ''

        # loop
        for word in text_splitted:
            changed = ''
            for tag in lem_tags:
                if lemmatizer.lemmatize(word, tag) != word:
                    changed = tag
            if changed == '':
                text_new.append(word)
            else:
                text_new.append(lemmatizer.lemmatize(word, changed))

        return " ".join(text_new)

    def lemmatize(self):
        """ This function lemmatizes texts """
        self.df[self.label] = self.df[self.label].apply(lambda x: self.lemmatize_one_text(x))

    def vocabulary_richness(self, text):
        """ This function returns vocabulary richness of a text
            inputs:
             - text """
        tokens = word_tokenize(text)
        total_length = len(tokens)
        uniques = set(tokens)
        unique_length = len(uniques)
        return unique_length / total_length

    def add_vocabulary_richness(self):
        """ This function adds a vocabulary richness column to the data frame """
        self.df["vocabulary_richness"] = self.df[self.label].apply(lambda x: self.vocabulary_richness(x))

    def clean_texts(self):
        self.lowerize()
        self.remove_emails()
        self.remove_mentions()
        self.remove_hyperlinks()
        self.remove_hashtags()
        self.remove_html_tags()
        self.remove_numbers()
        self.encode_unknown()
        self.clean_punctuation_no_accent()
        self.clean_stopwords()
        self.more_cleaning()
        self.lemmatize()
        self.add_vocabulary_richness()
        return self.df.copy()

