import nltk
import spacy
import unicodedata
from contractions import CONTRACTION_MAP
import re
from nltk.corpus import wordnet
import collections
#from textblob import Word
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup

tokenizer = ToktokTokenizer()
wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
wnl = nltk.stem.wordnet.WordNetLemmatizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_md')
#nlp_vec = spacy.load('en_vectors_web_lg', parse=True, tag=True, entity=True)

def simple_tokenize(text, use_stop_word=False):
    if use_stop_word:
        doc_tokens = [token.strip().lower() for token in wtk.tokenize(text) if token.lower() not in stopword_list]
    else:
        doc_tokens = [token.strip() for token in wtk.tokenize(text)]
    return doc_tokens

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    if bool(soup.find()):
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    else:
        stripped_text = text
    return stripped_text


#def correct_spellings_textblob(tokens):
#	return [Word(token).correct() for token in tokens]  


def simple_porter_stemming(text):
    ps = nltk.stem.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def lemmatize_text_wnl(text):
    text = [word.strip() for word in wtk.tokenize(text)]
    text = ' '.join([wnl.lemmatize(word) if (not word.isnumeric()) & (word not in stopword_list) else word for word in text])  # 0307 edition. prevent lemmatizing stopwords 
    return text

def lemmatize_text_spacy(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
            
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):

   contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                     flags=re.IGNORECASE|re.DOTALL)
   def expand_match(contraction):
       match = contraction.group(0)
       first_char = match[0]
       expanded_contraction = contraction_mapping.get(match)\
                               if contraction_mapping.get(match)\
                               else contraction_mapping.get(match.lower())
       expanded_contraction = first_char+expanded_contraction[1:]
       return expanded_contraction

   expanded_text = contractions_pattern.sub(expand_match, text)
   expanded_text = re.sub("'", "", expanded_text)
   return expanded_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]|\[|\]' if not remove_digits else r'[^a-zA-Z\s]|\[|\]'
    text = re.sub(pattern, '', text)
    return text


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_tokens = [token for token in filtered_tokens if len(token) > 1] 
    filtered_text = ' '.join(filtered_tokens)   
    return filtered_text


def normalize_corpus(corpus, html_stripping=False, accented_char_removal=True, text_lower_case=True, 
                     text_stemming=False, text_lemmatization_spacy=False, text_lemmatization_wnl=True,
                     special_char_removal=True, remove_digits=False, contraction_expansion=False,
                     stopword_removal=True, split=False, by_doc=False, stopwords=stopword_list):
    
    normalized_corpus = []

    # normalize each document in the corpus
    for doc in corpus:

        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)

        # remove extra newlines
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))

        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)

       # expand contractions
        if contraction_expansion:
           doc = expand_contractions(doc)

        # lemmatize text with spacy
        if text_lemmatization_spacy and not text_lemmatization_wnl:
            doc = lemmatize_text_spacy(doc)

        # lemmatize text with nltk
        if text_lemmatization_wnl and not text_lemmatization_spacy:
            doc = lemmatize_text_wnl(doc)

        # stem text
        if text_stemming and not text_lemmatization_wnl and not text_lemmatization_spacy:
        	doc = simple_porter_stemming(doc)

        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

         # lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case, stopwords=stopwords)

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()
        
        if split:
            doc = doc.split()

        if by_doc and not split:
            normalized_corpus.append([doc])
        else:
            normalized_corpus.append(doc)
            
        
    return normalized_corpus
