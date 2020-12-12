# -*- coding: utf-8 -*-
# +
"""
PREPROCESSING MODULE

Author: Maryam Rahafrooz
Created June 29 2020

This is a module for cleaning and preprocessing of the collected tweets

-Data cleaning:
1) Removing mentions and URL links from the tweets,
2) Removing hashtags from the end of the tweets (keep the hashtags that are used at the middle of the tweet texts),
3) Normalizing the tweets by lower casing their characters and expanding the contractions,
4) Removing all punctuations, numbers, and emojis,
5) Filtering out the tweets with non-ASCII characters (tweets with non-ASCII characters other than emojis and 
special characters).

-Text preprocessing:
1) Tokenizing,
2) PoS tagging,
3) Lemmatizing using PoS tags,
4) Removing end punctuations, stopwords, and one and two-letter tokens,

For stopword removal, we combine nltk and sklearn stopword lists (378 stopwords). 
The nltk stopword list has 179 words, and the sklearn stopword list includes 318.

"""
import pandas as pd
import numpy as np
import re
from contractions import CONTRACTION_MAP  #file contractions.py is added to the repository
import swifter
import spacy
import gensim
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
from tabulate import tabulate
import csv
from sklearn.feature_extraction import stop_words
from autocorrect import spell


def isEnglish(text):
    """ 
    detect tweets with non-ASCII characters   
    """
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def all_hashtags_remover(string):
    string = re.sub(r"#(\w+)", '', string, flags=re.MULTILINE)
    return string


def end_hashtags_remover(string):
    string = re.sub(r'http\S+', '', string)  # remove URLs, remove urls first then detect the hashtags that are at the end of the text.
    #in this way we filter out the hashtags that occure before and after the last url at the end of the tweet texts
    string = re.sub("(?:\\s*#\\w+)+\\s*$", '', string)
    return string


def hashtag_decision(df, col_name):  # to  decide about hashtags in texts of the tweets
    """
    text : original text of tweets
    text_h_in (all hashtags) = text keep all the hashtags inside texts just remove # sign
    text_h_out (no hashtags) : remove all hashtags from the texts
    text_neh (no end hashtag) : remove end hashtags only. if tweet ends with a url remove it after deleting the url)
    """
    if col_name == 'text_h_in':
        df['text_h_in'] = df['text']
#         df.rename(columns={'text':'text_h_in'}, inplace=True)# text _h_in is same as the original text
    elif col_name == 'text_h_out':
        df['text_h_out'] = df['text'].apply(lambda x: all_hashtags_remover(x))  # to create text with all hashtags excluded
    elif col_name == 'text_h_ne':
        df['text_h_ne'] = df['text'].apply(lambda x: end_hashtags_remover(x))  # to create text with no end hashtags
        # to list the hashtags that are not end hashtags:
        df['hashtags_ne'] = df['text_h_ne'].apply( lambda x: re.findall(r"#(\w+)", x))
#     # reorder the columns
#     cols = list(df.columns)
#     a, b = cols.index('hashtags'), cols.index('text_neh')
#     cols[b], cols[a] = cols[a], cols[b]
#     df = df[cols].copy()

# create hashtags_list (hashtags column as list)
    df['hashtags_list'] = df['hashtags'].str.split(';')
    df['hashtags_list'] = df['hashtags_list'].apply(lambda d: d if isinstance(d, list) else [])  #to replace NaN with empty list
    # number of hashtags: h_humber , h_ne_number
    df['h_number'] = df['hashtags_list'].apply(len)

    df['mentions_list'] = df['mentions'].str.split(';')
    df['mentions_list'] = df['mentions_list'].apply(lambda d: d if isinstance(d, list) else [])  #to replace NaN with empty list
    # number of hashtags: h_humber , h_ne_number
    df['men_number'] = df['mentions_list'].apply(len)

    if col_name == 'text_h_ne':df['h_ne_number'] = df['hashtags_ne'].apply(len)


#     if (col_name=='text_h_in'or col_name=='text_h_out'):
#print('This dataset has a total of', df['h_number'].sum(),'(repetetive) hashtags.')
#     else:
#print('This dataset has a total of', df['h_number'].sum(),'(repetetive) hashtags out of which ',df['h_ne_number'].sum(),'hashtags are non-end hashtags.',)
    return df


def text_preprocessing(string):
    # normalize the text: ’ and en and em dash are not ASCII characters
    string = string.replace("’", "'")  #replace ’
    string = string.replace("\u2013", " ")  #replace en dash with white space
    string = string.replace("\u2014", " ")  #replace em dash with white space
    string = re.sub(r'http\S+', '', string)  # remove URLs,
    string = re.sub(r'([@][\w_-]+)', '', string)  # remove mentions
    string = string.lower()  #lowercasing
    #expand contractions
    contractions_re = re.compile('(%s)' % '|'.join(CONTRACTION_MAP.keys()))

    def replace(match):
        return CONTRACTION_MAP[match.group(0)]

    string = contractions_re.sub(replace, string)

    def deEmojify(string):
        """remove emojis
        input: tweet text
        output: tweet text without emojis
        """
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\U0001F600-\U0001F92F"
            u"\U0001F190-\U0001F1FF"
            u"\U0001F926-\U0001FA9F"
            u"\u200d"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\u3030"
            u"\ufe0f"
            "]+",
            flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

    string = deEmojify(string)  #remove emojies

    string = re.sub('[' + '"$%&()*+\-/:<=>[\\]^_`\'”“’;,{|}!?.\\~•@ #…‘‼™»«® ⃣ ' + ']+',' ', string)  # strip punctuations
    #note: you can exclude end-punctuations such as !?. and keep them inside the text for POS tagging (we didn't exclud them)
    #in re to consider - need to add \- also same is for ' we add \'
    string = re.sub('[0-9]+', '',string)  #remove digit numbers (separate digits or inside the words)

    # for the punctuations that still remain in the text,remove consecutive punctuation marks and replace them with the first one
    #e.g. u.s.! -> u.s! or there!!! -> there!
    r = re.compile(r'([.,/#!?$%^&*;:{}=_`~()-])[.,/#!?$%^&*;:{}=_`~()-]+')
    string = r.sub(r'\1', string)

    #     def misspells(string):
    #         string= ' '.join ([spell(w) for w in string.split()])
    #         return string
    #     string = misspells(string)

    string = re.sub('\s+', ' ',string)  #remove tab and new line and double spaces
    return string


def tokenize(string):
    """ 
    Tokenize, PoS tagging, lemmatize, and remove stop words and  puncs and single letter words 
    
    note: nltk stop word list has 179 words, sklearn stop word list has 318
    in this module we combine the two sets of stop words (378 words)
    for example, "please" and "somehow" will be removed at this step. 
    """
    punctuation = u",.?!()-_\"\'\\\n\r\t;:+=*<>@#§^$%&|/`”“’{|}~•[]'"
    stop_words_eng = set(stopwords.words('english'))
    stop_words_sckitlearn = set(stop_words.ENGLISH_STOP_WORDS)
    lemmatizer = WordNetLemmatizer()
    tag_dict = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}

    def extract_wnpostag_from_postag(tag):
        '''
        POS tag list:
        CC coordinating conjunction
        CD cardinal digit
        DT determiner
        EX existential there (like: "there is" ... think of it like "there exists")
        FW foreign word
        IN preposition/subordinating conjunction
        JJ adjective 'big'
        JJR adjective, comparative 'bigger'
        JJS adjective, superlative 'biggest'
        LS list marker 1)
        MD modal could, will
        NN noun, singular 'desk'
        NNS noun plural 'desks'
        NNP proper noun, singular 'Harrison'
        NNPS proper noun, plural 'Americans'
        PDT predeterminer 'all the kids'
        POS possessive ending parent's
        PRP personal pronoun I, he, she
        PRP$ possessive pronoun my, his, hers
        RB adverb very, silently,
        RBR adverb, comparative better
        RBS adverb, superlative best
        RP particle give up
        TO to go 'to' the store.
        UH interjection errrrrrrrm
        VB verb, base form take
        VBD verb, past tense took
        VBG verb, gerund/present participle taking
        VBN verb, past participle taken
        VBP verb, sing. present, non-3d take
        VBZ verb, 3rd person sing. present takes
        WDT wh-determiner which
        WP wh-pronoun who, what
        WP$ possessive wh-pronoun whose
        WRB wh-abverb where, when
        '''
        #take the first letter of the tag
        #the second parameter is an "optional" in case of missing key in the dictionary
        return tag_dict.get(tag[0].upper(), None)

    def lemmatize_tupla_word_postag(tupla):
        #giving a tupla of the form (wordString, posTagString) like ('guitar', 'NN'), return the lemmatized word
        tag = extract_wnpostag_from_postag(tupla[1])
        return lemmatizer.lemmatize(tupla[0], tag) if tag is not None else tupla[0]  
        # lematize and if tag is=None then it returns the word itself

    def bag_of_words_nc(sentence,stop_words=None):  # if have new stp words, add them here
        if stop_words is None:
            # stop_words = stop_words_eng #only nltk stopwords
            stop_words = stop_words_eng.union(stop_words_sckitlearn)  # combines the stop-word set of nltk and sklearn
        original_words = word_tokenize(sentence)
        tagged_words = nltk.pos_tag(original_words)  #returns a list of tuples: (word, tagString) like ('And', 'CC')
        original_words = None
        lemmatized_words = [lemmatize_tupla_word_postag(ow) for ow in tagged_words]
        tagged_words = None
        cleaned_words = [w for w in lemmatized_words if (w not in punctuation) 
                         and (len(w) > 2) and (w not in stop_words) ]
        #also removes single letter words.e.g. len(w)>3 will remove words with 1,2,3 letters
        lemmatized_words = None
        return cleaned_words

    return bag_of_words_nc(string)


def spacy_tokenizer(preprocessed_data):  #, tmpfolder, tmppath, subfolder
    '''
    input: preprocessed  texts
    output: remove stop-words and tokenize (bigrams) for specified postags
    reference:https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

    '''
    # Convert preprocessed texts to list
    data = preprocessed_data.preprocessed.values.tolist()

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    #tokenize:
    def sent_to_words(sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(
        data_words, min_count=5,
        threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    # print(trigram_mod[bigram_mod[data_words[0]]])

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    stop_words = stopwords.words('english')
    stop_words.extend(['re'])

    def remove_stopwords(texts):
        return [[
            word for word in simple_preprocess(str(doc))
            if word not in stop_words
        ] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    # def make_trigrams(texts):
    #     return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts,
                      allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV','PROPN']):  #PROPN: proper noun
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([
                token.lemma_ for token in doc
                if token.pos_ in allowed_postags and len(token.lemma_) > 2])  
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(
        data_words_bigrams,
        allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN'])  

    #     preprocessed_data['spacy_lemmatized']=data_lemmatized

    #     #save data_lemmatized
    #     preprocessed_data.to_csv(os.path.join(tmpfolder, tmppath,subfolder,'spacy_lemmatized.csv'), index=False )

#     print('the spacy_lemmatized texts are successfully created!')
    return data_lemmatized


def preprocess(min_word):
    """ 
    preprocess and tokenize the tweet texts in specified dataset and column and remove tweets 
    that include non-ASCII characters with isEnglish function because en_tweets fuction by itself
    cannot capture all non english tweets.
    
    input: dataset and column to consider
    output: preprocessed and tokenized tweets for that dataset and column
    
    note: we first remove special punctuations and emojis and then filter out the tweets based on
    having non-ASCII characters otherwise special characters and emojis are non-ASCII and will 
    lead to removing those tweets
    
    note: if want to use the preprocessed tweets themselves (rather than the tokenized texts),
    use output of this function
    """
    dataset = 'data files/data2.csv'  #options are 'data1' and 'data2'
    col_name = 'text_h_in'  #options are 'text_h_in ,'text_h_out' , 'text_h_ne'

    # read the specified dataset and save it in 'df'
    df = pd.read_csv(dataset)

    # to decide about hashtags in texts of the tweets based on the specified column in 'col_name'
    df = hashtag_decision(df, col_name)

    #preprocess the column specified by 'col_name' and create the following two columns
    df['preprocessed'] = df[col_name].swifter.apply(lambda x: text_preprocessing(x))  # preprocessed text

    #remove duplicate preprocessed tweets and keep the first case
    df = df.drop_duplicates(subset=['preprocessed', 'hashtags'], keep="first")

    #remove preprocessed tweets with less than 'min_word' words
    df = df[df['preprocessed'].apply(lambda x: len(x.split())) >= min_word]

    df['tokenized'] = df['preprocessed'].swifter.apply(lambda x: tokenize(x))  # preprocessed text is tokenized

    df['spacy_lemmatized'] = spacy_tokenizer(df)

    #remove a tweet if there is any non-ASCII character in the preprocessed text of that tweet
    df['EnLetter'] = df['preprocessed'].apply(lambda x: isEnglish(x))
    df = df[(df.EnLetter == True)]  #filter out tweets with non-ASCII characters in text

    df['EnLetter'] = df['hashtags'].astype('str').apply(lambda x: isEnglish(x))
    df = df[(df.EnLetter == True)]  #filter out tweets with a hashtag with non-ASCII characters
    df = df.drop('EnLetter', axis=1)  # drop the column

    df.to_csv('data files/df.csv',index=False)  #save df (is preprocessed form of the specified dataset)
    print('The dataset of preprocessed and tokenized tweets for dataset is successfully created! This dataset has',
          format(df.shape[0], ','), 'tweets.\n')
