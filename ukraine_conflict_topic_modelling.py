#Project: Framing the Ukraine conflict on Twitter - a hashtag-based analysis of the social media discourse around the Ukraine conflict (43397 tweets) from 03/07/2022 to 12/07/2022

#load libraries
import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import html
import re
import spacy
from tqdm import tqdm
from wordcloud import WordCloud 
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary


#import tweets on Ukraine conflict
#data source: https://www.kaggle.com/datasets/aneeshtickoo/russia-ukraine-conflict

df = pd.read_csv('ukraine_conflict.csv')

#for simplicity focus on tweets in English (36581 tweets)
df = df[df['language'] == 'en'] 

#parse hashtags from strings to lists
def parse_hashtags(x):
    try:
        return [tag.lower() for tag in ast.literal_eval(x) if isinstance(tag, str)]
    except:
        return []

#apply parsing function to hashtags column
df['hashtags_list'] = df['hashtags'].apply(parse_hashtags)


#count hashtags
hashtags_counter = Counter()

for hashtags in df['hashtags_list']:
    hashtags_counter.update(hashtags)


#convert counter to df 
hashtag_df = pd.DataFrame(hashtags_counter.items(), columns=['hashtag', 'count'])
hashtag_df = hashtag_df.sort_values(by='count', ascending=False).reset_index(drop=True)


#visualize top 50
plt.figure(figsize=(12, 6))
sns.barplot(data=hashtag_df.head(50), x='count', y='hashtag', palette='coolwarm')
plt.title("Top 20 Hashtags in Ukraine Conflict Tweets (July 3–12, 2022)")
plt.xlabel("Frequency")
plt.ylabel("Hashtag")
plt.tight_layout()
plt.show()

#create a list with top 100 hashtags
all_hashtags = hashtag_df['hashtag'].tolist()


#save top 100 hashtags to a text file   
with open('all_hashtags.txt', 'w') as f:
    for hashtag in all_hashtags:
        f.write(f"{hashtag}\n") 

#used chatgpt to classify hashtags intoin clear pro-Ukraine and pro-Russia categories (if any)
#saved the lists in pro_russia.txt and pro_ukraine.txt

#load pro-Ukraine hashtags
with open('pro_ukraine.txt', 'r') as f:
    pro_ukraine = [line.strip().lower() for line in f.readlines()]

#load pro-Russia hashtags
with open('pro_russia.txt', 'r') as f:
    pro_russia = [line.strip().lower() for line in f.readlines()]


#function to classify hashtags based on pro-Ukraine and pro-Russia lists
def classify_hashtags(tags):
    if not tags or not isinstance(tags, list):
        return 'Neutral'
    
    score = 0
    for tag in tags:
        if tag in pro_ukraine:
            score += 1
        elif tag in pro_russia:
            score -= 1

    if score > 0:
        return 'Pro-Ukraine'
    elif score < 0:
        return 'Pro-Russia'
    else:
        return 'Neutral'    

#apply classification function to hashtags_list column
df['group'] = df['hashtags_list'].apply(classify_hashtags)

#count classified hashtags and plot results
classification_counts = df['group'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=classification_counts.index, y=classification_counts.values, palette='Set2')
plt.title("Hashtag Classification in Ukraine Conflict Tweets (July 3–12, 2022)")
plt.xlabel("Classification")
plt.ylabel("Count")
plt.tight_layout()
plt.show()  

df.groupby('group').size()

#Neutral>Pro-Ukraine>Pro-Russia 
#Neutral hashtags likely include mixed cases

nlp = spacy.load("en_core_web_sm")  # load spacy model

#load stop words from spacy
stopwords = nlp.Defaults.stop_words
#print(stopwords)

#add custom stop words
custom_stopwords = ['ukraine', 'russia', 'war', 'conflict', 'putin', 'zelensky', 'russian', 'ukrainian']
for word in custom_stopwords:
    for variant in (word.lower(), word.capitalize()):
        nlp.Defaults.stop_words.add(variant)
        nlp.vocab[variant].is_stop = True


#remove hashtags from tweets to avoid bias in word frequency 
def remove_hashtags(tweet):
    return re.sub(r'#\w+', '', tweet)

#function to get most common words in tweets
def get_common_words(tweets, n=50):
    counter = Counter()
    
    def clean_texts(tweets):
        for t in tweets:
            unescaped = html.unescape(t)  # Convert &amp; to &
            yield remove_hashtags(unescaped)
    
    clean_tweets = clean_texts(tweets)  # pass tweets directly here
    
    for doc in tqdm(nlp.pipe(clean_tweets, batch_size=1000, n_process=4), total=len(tweets), desc="Processing tweets"):
        tokens = [
            token.lemma_.lower() 
            for token in doc 
            if token.is_alpha and not token.is_stop and len(token) > 1
        ]
        counter.update(tokens)
    
    return counter.most_common(n)


pro_ukraine_tweets = df[df['group'] == 'Pro-Ukraine']['tweet']
pro_russia_tweets = df[df['group'] == 'Pro-Russia']['tweet']
neutral_tweets = df[df['group'] == 'Neutral']['tweet']

# apply to each group
pro_ukraine_words = get_common_words(pro_ukraine_tweets, n=50)
pro_russia_words = get_common_words(pro_russia_tweets, n=50)
neutral_words = get_common_words(neutral_tweets, n=50)


#word clouds for each group
def plot_wordcloud(words, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()  

plot_wordcloud(pro_ukraine_words, "Pro-Ukraine Words")
plot_wordcloud(pro_russia_words, "Pro-Russia Words")
plot_wordcloud(neutral_words, "Neutral Words")  

#a simple word cloud inspection shows that:
#pro-Ukraine tweets focus more on direct war costs: "military", "destroy", "force", "people"
#the few pro-Russia tweets focus on personal attacks towards the opponent and no mention of direct war consequences: "scammer", "unscrupulous", "take", "advantage"
#the neutral tweets are indeed more neutral (i.e., "news", "world", "live", "say") and focus on indirect costs of war: "gas", "inflation", "price"
#would be interesting to see if this changes depending on the location of the user(close to the war vs further), but location data is availabe for very few tweets

#let's now use topic modeling to get a more fine-grained understanding of the topics discussed in each group
#topic modeling with BERT
def create_topic_model():
    return BERTopic(embedding_model="all-MiniLM-L6-v2", verbose=True, calculate_probabilities=True)

#assign topic models to each group
pro_russia_model = create_topic_model()
pro_ukraine_model = create_topic_model()
neutral_model = create_topic_model()


#transform pro_ukraine_tweets in a list
pro_ukraine_tweets = df[df['group'] == 'Pro-Ukraine']['tweet'].tolist()
pro_russia_tweets = df[df['group'] == 'Pro-Russia']['tweet'].tolist()
neutral_tweets = df[df['group'] == 'Neutral']['tweet'].tolist()

#light preprocessing function (BERT is robust to noise, so we keep it simple)
def preprocess(tweet):
    tweet = html.unescape(tweet)
    tweet = remove_hashtags(tweet)
    
    # remove urls and mentions
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"[^\w\s,]", "", tweet)  # Remove punctuation, emojis, and special characters
    
    # lowercase, strip, normalize spaces
    tweet = tweet.lower().strip()
    tweet = re.sub(r"\s+", " ", tweet)
    return tweet if tweet else None


#apply preprocessing 

cleaned_pro_ukraine_tweets = [tweet for tweet in pro_ukraine_tweets if tweet is not None]
cleaned_pro_russia_tweets = [tweet for tweet in pro_russia_tweets if tweet is not None]
cleaned_neutral_tweets = [tweet for tweet in neutral_tweets if tweet is not None]

#fit topic models
pro_ukraine_topics, pro_ukraine_probs = pro_ukraine_model.fit_transform(cleaned_pro_ukraine_tweets)
pro_russia_topics, pro_russia_probs = pro_russia_model.fit_transform(cleaned_pro_russia_tweets)
neutral_topics, neutral_probs = neutral_model.fit_transform(cleaned_neutral_tweets)

#show first 10 topics for pro-Ukraine tweets
pro_ukraine_model.get_topic_info().head(10) #topics range from "war in ukraine" to "famine caused by Russia" to specific attacks on those days to "money aid"
pro_ukraine_model.get_topic(2)  
pro_ukraine_model.get_representative_docs(2)[:10]  #show representative documents for topic 2 = famine caused by Russia

#show first 10 topics for pro-Russia tweets
pro_russia_model.get_topic_info().head(10) #topics range from "war in ukraine" to "famine caused by Russia" to specific attacks on those days to "money aid"
pro_russia_model.get_representative_docs() #show representative documents for topic 0 = personal attacks on Zelensky
#based mostly on a scam tweet

#show first 10 topics for neutral tweets
neutral_model.get_topic_info().head(10) #topics range from "oil prices" to "inflation" to "food production" to "russia's propaganda" to "war being considered a money-laundry scheme" + other non-ukraine relevant topics

#show representative documents for topic 4 = money laundering scheme
neutral_model.get_representative_docs(4)[:10]  #show representative documents for topic 4

#visualize topics
pro_ukraine_model.visualize_topics()
neutral_model.visualize_topics() #shows more heterogeneous topics, including non-Ukraine related topics

#evaluate topic coherence

def evaluate_coherence(model, texts):
    # Create a dictionary and corpus for the model
    tokenized_texts = [text.split() for text in texts if isinstance(text, str) and text.strip() != '']
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    # Compute coherence score
    coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    def get_topic_words(topic_model, top_n=10):
        topic_words = []
        topics = topic_model.get_topics()
        for topic_id in topics:
            if topic_id == -1:  # skip outliers/noise topic
                continue
            words = [word for word, weight in topics[topic_id][:top_n]]
            topic_words.append(words)
        return topic_words
    topic_words = get_topic_words(model, top_n=10)

    # Compute coherence score
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    return coherence_score  

#evaluate coherence for each group
pro_ukraine_coherence = evaluate_coherence(pro_ukraine_model, cleaned_pro_ukraine_tweets)
pro_russia_coherence = evaluate_coherence(pro_russia_model, cleaned_pro_russia_tweets)
neutral_coherence = evaluate_coherence(neutral_model, cleaned_neutral_tweets)
print(f"Pro-Ukraine Coherence: {pro_ukraine_coherence}")
print(f"Pro-Russia Coherence: {pro_russia_coherence}")
print(f"Neutral Coherence: {neutral_coherence}")    


