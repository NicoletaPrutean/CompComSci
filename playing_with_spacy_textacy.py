import spacy
from spacy import displacy
import textacy
from textacy import text_stats 
from textacy import preprocessing
from textacy import extract
from functools import partial



nlp = spacy.load("en_core_web_sm")  # Load the English model (small version)

doc = nlp("They're linguists!") #creates a list of tokens
text = "They're linguists!"
len(doc)
len(text)
doc[0]

#tokenization
tokens = [token.text for token in doc]  # Extracting tokens from the doc
print(tokens)

#lemmatization
lemmas = [token.lemma_ for token in doc]  # Extracting lemmas from the doc
print(lemmas)

#part of speech tagging: POS tagging
pos_tags = [(token.text, token.pos_) for token in doc]  # Extracting part of speech tags (e.g., pronoun, noun, verb, punct, auxiliary)
print(pos_tags)

#named entity recognition: NER
doc = "Noam Chomsky was born in the US in 1928."
doc = nlp(doc)
print(doc.ents)

for token in doc.ents:
    print(token.text, token.label_)  # Extracting named entities and their labels

displacy.render(doc, style="ent", jupyter=True)  # Visualizing named entities in Jupyter Notebook

#dependency parsing (stanza might be better for this)
#analyses relationships between words in a sentence

doc = nlp("Chomsky wrote the syntactic structures in 1957.")
displacy.render(doc, style="dep", jupyter=True)  # Visualizing dependency parsing in Jupyter Notebook

for token in doc:
    print(token.dep_, token.text)


#play with textacy
#flesch reading ease = how easy it is to read a text
#analysing Biden's speech

with open("sou.txt", "r") as f:
    text = f.read()
    text = text.replace("\n", "")  # Replace newlines with spaces for better processing

nlp = spacy.load("en_core_web_sm") #load model
doc = nlp(text) #tokenize

#nr sentences
text_stats.basics.n_sents(doc)

#nr words
text_stats.basics.n_words(doc)

#nr unique words
text_stats.basics.n_unique_words(doc)

#lexical density = ratio of unique words to total words
text_stats.diversity.ttr(doc) #25%

#dependecy relations
text_stats.counts.dep(doc)

#coarsed-grained part of speech tags
text_stats.counts.pos(doc)

#fine-grained part of speech tags
text_stats.counts.tag(doc)

#morphological features (smallest meaningful units of language e.g., teach-er)
text_stats.counts.morph(doc)

#flesch reading ease
text_stats.readability.flesch_reading_ease(doc)  # Higher score means easier

txt = "this! is: www.example.com #$"

preproc = preprocessing.make_pipeline(
    preprocessing.remove.html_tags,
    preprocessing.replace.urls,
    preprocessing.normalize.unicode,
    partial(preprocessing.remove.punctuation, only=['$', '//', '!', '"']),
    preprocessing.replace.emojis,
    preprocessing.normalize.whitespace,)

clean = preproc(txt)
print(clean)  # Cleaned text

#extract some info from doc

#extract 3-grams with some features

en3 = extract.basics.ngrams(doc, n=3, min_freq=2, filter_stops=False, filter_punct=True, filter_nums=False, include_pos={"PRON", "VERB"})
list(en3)  # List of 3-grams with specified features

#keyword extraction
extract.keyterms.textrank(doc, topn=10)  # Extracting top 10 key terms using TextRank

#find tokens that match a specific pattern
em = extract.matches.token_matches(doc, [{"POS": "DET", "OP": "?"}, {"POS": "ADJ", "OP": "+"}, {"POS": "NOUN", "OP": "+"}]) # (e.g., optional determiner, an adjective, a noun) 
print(list(em))  # List of tokens matching the specified pattern


#keyword in context
ek = extract.kwic.keyword_in_context(doc, keyword="people", window_width=25) #25 characters on each side of the keyword
for i in list(ek):
    print(i)  


