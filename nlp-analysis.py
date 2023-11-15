import pandas as pd
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

nlp_spacy = spacy.load("en_core_web_sm")

# Load your DataFrame
nlp_analysis_df = pd.read_csv('news_article_text.csv')  # Replace with your file path or DataFrame source

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# English stop words list
stop_words = set(stopwords.words('english'))


# Preprocessing function
def preprocess_text(text):
    # Tokenize and convert to lower case
    tokens = nlp_spacy(text.lower())

    # Remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    stripped = [token.text for token in tokens]

    # Remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # Filter out stop words
    words = [w for w in words if not w in stop_words]

    # Lemmatize
    lemmatized = [lemmatizer.lemmatize(w) for w in words]

    return ' '.join(lemmatized)


# Apply preprocessing to each row in the 'Article Body' column
nlp_analysis_df['ArticleBodyCleaned'] = nlp_analysis_df['Article Body'].dropna().apply(preprocess_text)

# Define a list of words you want to remove
words_of_interest = ['payment', 'bank', 'account', 'new', 'market', 'provider', 'service', 'million',
                     'busi']  # Replace with your words


def remove_corpus_words_of_interest(text):
    # Check if the text is a string
    if not isinstance(text, str):
        return text  # Return as is if not a string (e.g., NaN or None)

    # Split text into words
    words = text.split()

    # Remove words in the predefined list
    words = [word for word in words if word.lower() not in words_of_interest]

    # Join the words back into a string
    return ' '.join(words)


# Apply the function to each row in the specific column
nlp_analysis_df['ArticleBodyCleaned'] = nlp_analysis_df['ArticleBodyCleaned'].apply(remove_corpus_words_of_interest)

"""Text blob analysis polarity"""
from textblob import TextBlob

# Function to get sentiment polarity
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply sentiment analysis to the cleaned text
nlp_analysis_df['SentimentPolarity'] = nlp_analysis_df['ArticleBodyCleaned'].dropna().apply(get_sentiment)

"""Vader Analysis The VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool is particularly 
good at handling sentiments expressed in social media contexts."""
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()


# Function to get VADER sentiment polarity scores
def get_vader_sentiment(text):
    return sia.polarity_scores(text)


# Apply VADER sentiment analysis to the cleaned text
nlp_analysis_df['VaderSentiment'] = nlp_analysis_df['ArticleBodyCleaned'].dropna().apply(
    lambda text: get_vader_sentiment(text)['compound'])

nlp_analysis_df.to_csv('nlp_analysis_with_vader_sentiment.csv', index=False)

"""Named Entity Recognition using spaCy
spaCy is a powerful and efficient library for NLP that provides capabilities for various tasks including NER."""

import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Function to extract entities
def extract_entities(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities


# Apply NER to the cleaned text
nlp_analysis_df['NamedEntities'] = nlp_analysis_df['ArticleBodyCleaned'].dropna().apply(extract_entities)

"""Topic Modeling using Gensim Topic modeling is a type of statistical model for discovering abstract topics that 
occur in a collection of documents. Gensim is a popular library for unsupervised topic modeling.
"""
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.parsing.preprocessing import preprocess_string

# Assuming `ArticleBodyCleaned` is the column after cleaning and preprocessing
# Drop the NaN values or replace them with an empty string
nlp_analysis_df.dropna(subset=['ArticleBodyCleaned'], inplace=True)

# Tokenize the cleaned text
nlp_analysis_df['Tokens'] = nlp_analysis_df['ArticleBodyCleaned'].dropna().apply(preprocess_string)

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(nlp_analysis_df['Tokens'].tolist())

# Convert dictionary into a bag-of-words
corpus = [dictionary.doc2bow(text) for text in nlp_analysis_df['Tokens'].tolist()]

# Apply LDA model
lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=10)


# Get the topics for each document
def get_lda_topics(text, dictionary=dictionary, lda_model=lda_model):
    bow = dictionary.doc2bow(preprocess_string(text))
    topics = lda_model.get_document_topics(bow)
    return topics


"""One approach is to modify the get_lda_topics function to return a formatted list or string of topic descriptions, 
based on the highest probability topics for each document. You can set a threshold for including topics (e.g., 
only include topics with a probability higher than a certain value) or simply list the top N topics."""

#
nlp_analysis_df['Topics'] = nlp_analysis_df['ArticleBodyCleaned'].dropna().apply(get_lda_topics)

"""The output [(2, 0.99489766)] from Gensim's LDA model is a list of tuples, where each tuple represents a topic and 
its corresponding probability for the given document. In this case, you have one tuple in the list:

The first element of the tuple (2) is the topic number or id. This id is an index for the array of topics that the 
model has learned. In an LDA model, each topic is a distribution over words in the vocabulary.

The second element (0.99489766) is the probability assigned to this topic for the given document. This value tells 
you how strongly the model believes that the document belongs to that topic. Here, it's saying that the given 
financial news article is almost certainly (with a probability of about 99.49%) about the topic labeled as '2' by the 
model.

To interpret this in the context of your LDA model, you would look at the set of words that are most strongly associated with Topic 2. This will give you an idea of what this topic represents (e.g., it might be related to stock market trends, mergers and acquisitions, etc.). In Gensim, you can use the show_topic method of the LDA model to get the top words associated with a topic:

# Assuming `lda_model` is your trained LdaModel object
print(lda_model.show_topic(2, topn=10))
This would print out the top 10 words that are most representative of Topic 2. The words and their corresponding weights will give you an insight into what the topic is about and, consequently, what aspect of financial news your document is mostly associated with."""


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    topic_num_keywords_probs = []

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the dominant topic, percentage contribution, and keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                topic_num_keywords_probs.append((int(topic_num), round(prop_topic, 4), topic_keywords))
            else:
                break
    return topic_num_keywords_probs


formatted_topics = format_topics_sentences(lda_model, corpus, nlp_analysis_df['ArticleBodyCleaned'])

# Add this data to your DataFrame
nlp_analysis_df['Dominant_Topic'] = [topic[0] for topic in formatted_topics]
nlp_analysis_df['Topic_Perc_Contrib'] = [topic[1] for topic in formatted_topics]
nlp_analysis_df['Keywords'] = [topic[2] for topic in formatted_topics]


# Function to alphabetize keywords
def alphabetize_keywords(keywords_str):
    keywords_list = keywords_str.split(", ")
    keywords_list.sort()
    return ", ".join(keywords_list)


# Apply the function to each row in the 'Keywords' column
nlp_analysis_df['Keywords_Alphabetized'] = nlp_analysis_df['Keywords'].apply(alphabetize_keywords)
nlp_analysis_df.to_csv('article_text_with_nlp_analysis.csv', index=False)

# ignore
nlp_analysis_df.to_csv('article_text_with_nlp_analysis_text.csv', index=False)
