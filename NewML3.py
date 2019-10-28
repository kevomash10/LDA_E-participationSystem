#imports
import nltk;import re
from time import time
import os,codecs,gensim,spacy,logging, warnings
import numpy as np
import pandas as pd
from pprint import pprint
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
warnings.filterwarnings("ignore",category=DeprecationWarning)


#loading dataset
# print()
# print("#" * 80)
# print()
# print("load dataset...")
# from sklearn.datasets import fetch_20newsgroups
# categories = ['sci.space', 'talk.politics.guns', 'sci.med','talk.religion.misc', 'alt.atheism' ,'rec.sport.hockey' ,'rec.sport.baseball' ,'soc.religion.christian',
#  'talk.politics.mideast', 'talk.politics.misc']
# newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,remove=('headers', 'footers', 'quotes'),shuffle=True, random_state=1)
# # newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,remove=('headers', 'footers', 'quotes'),shuffle=True, random_state=1)

# data= newsgroups_train.data
# data_target = newsgroups_train.target

t0 = time()
df = pd.read_json('newsgroups.json')
print("print topic all names...")
print(df.target_names.unique())
data = df.content.values.tolist()
print(df.head())
print(len(df))
print("done in %0.3fs." % (time() - t0))
print()
print("##" * 80)


from nltk.corpus import stopwords
	# stop_words = stopwords.words('english')
stopwords_verbs = ['about','always','say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can','this','will','with','from','your','what','been','over','would','which']
stopwords_other = ['one', 'mr', 'bbc', 'shall','image','use', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something','from', 'subject', 're', 'edu', 'use','i','am','please','cannot','would','do','not','make','say','know','come','be','use','mb']
# my_stopwords = stopwords.words('english') + stopwords_verbs + stopwords_other

noise_list = ["women","woman","people","bring","bother","belong","slmr","faqs","talk","morris","clinton","bill","a","is","be","about","above","after","again","against","ain","about","above","across","after","again","against","all","almost","alone","along","already","also","alt","ough","always","among","an","and","another","any","anybody","anyone","anything","anywhere","are","area","areas","around","as","ask","asked","asking","asks","at","away","back","backed","backing","backs","be","became","because","become","becomes","been","before","began","behind","being","beings","best","better","between","big","both","but","by","came","can","cannot","case","cases","certain","certainly","clear","clearly","come","could","did","differ","different","differently","do","does","done","down","downed","downing","downs","during","each","early","either","end","ended","ending","ends","enough","even","evenly","ever","every","everybody","everyone","everything","everywhere","face","faces","fact","facts","far","felt","few","find","finds","first","for","four","from","full","fully","further","furthered","furthering","furthers","gave","general","generally","get","gets","give","given","gives","go","going","good","goods","got","great","greater","greatest","group","grouped","grouping","groups","had","has","have","having","he","her","here","herself","high","higher","highest","him","himself","his","how","however","if","important","in","interest","interested","interesting","interests","into","is","it","its","itself","just","keep","keeps","kind","knew","know","known","knows","large","largely","last","later","latest","least","less","let","lets","like","likely","long","longer","longest","made","make","making","man","many","may","me","member","members","men","might","more","most","mostly","mr","mrs","much","must","my","myself","necessary","need","needed","needing","needs","never","new","new","newer","newest","next","no","nobody","non","noone","not","nothing","now","nowhere","number","numbers","of","off","often","old","older","oldest","on","once","one","only","open","opened","opening","opens","or","order","ordered","ordering","orders","other","others","our","out","over","part","parted","parting","parts","per","perhaps","place","places","point","pointed","pointing","points","possible","present","presented","presenting","presents","problem","problems","put","puts","quite","rather","really","right","right","room","rooms","said","same","saw","say","says","second","seconds","see","seem","seemed","seeming","seems","sees","several","shall","she","should","show","showed","showing","shows","side","sides","since","small","smaller","smallest","so","some","somebody","someone","something","somewhere","state","states","still","such","sure","take","taken","than","that","the","their","them","then","there","therefore","these","they","thing","things","think","thinks","this","those","though","thought","thoughts","three","through","thus","to","today","together","too","took","toward","turn","turned","turning","turns","two","under","until","up","upon","us","use","used","uses","very","want","wanted","wanting","wants","was","way","ways","we","well","wells","went","were","what","when","where","whether","which","while","who","whole","whose","why","will","with","within","without","work","worked","working","works","would","year","years","yet","you","young","younger","youngest","your","yours","all","am","an","and","any","are","arent","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"] 
noise_list = stopwords.words('english') + stopwords_verbs + stopwords_other + noise_list
# def remove_noise(input_text):
#     words = input_text.split()
#     noise_free_words = [word for word in words if word not in noise_list] 
#     noise_free_text = " ".join(noise_free_words) 
#     return noise_free_text

#preprocess
# def pre_process(texts):

# 	print(">>remove Emails...")
# 	data = [re.sub('\S*@\S*\s?', '', sent) for sent in texts]

# 	print(">>remove new line characters..")
# 	data = [re.sub('\s+', ' ', sent) for sent in data]

# 	print(">>remove single quotes...")
# 	data = [re.sub("\'", "", sent) for sent in data]


# 	# data = list(sent_to_words(data))

# 	# data = ''.join(data)

# 	print(">>remove stopwords...")
# 	# data= remove_noise(data)
# 	data=remove_stopwords(data)

# 	print(">>lemmatization...")
# 	data_lemmatized = lemmatization(data, allowed_postags=['NOUN','VERB'])
# 	# data_lemmatized=lemmatizer(data)
# 	print(data_lemmatized)
# 	print()

# 	data_lemmatized = list(sent_to_words(data_lemmatized))
	

# 	# print(">>tokenize..")
# 	# word_tokenize(str(data_words))


# 	return data_lemmatized

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in noise_list] for doc in texts]


nlp = spacy.load('en', disable=['parser', 'ner'])
def lemmatization(texts, allowed_postags=['NOUN', 'VERB']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# from nltk.stem.wordnet import WordNetLemmatizer
# lemm=WordNetLemmatizer()
# def lemmatizer(input_text):
#     words = input_text
#     lem_words = [lemm.lemmatize(word,"v") for word in words] 
#     # lem_text = " ".join(lem_words) 
#     return lem_words

def pre_process(texts):

	print(">>remove Emails...")
	# Remove Emails
	data = [re.sub('\S*@\S*\s?', '', sent) for sent in texts]

	print(">>remove new line characters...")
	# Remove new line characters
	data = [re.sub('\s+', ' ', sent) for sent in data]

	print(">>remove single quotes...")
	# Remove distracting single quotes

	data = [re.sub("\'", "", sent) for sent in data]

	print(">>remove short words..")
	data = [re.sub(r'\b\w{1,3}\b','', sent) for sent in data]
	
	data_words = list(sent_to_words(data))	

	print(">>lemmatization...")
	nlp = spacy.load('en', disable=['parser', 'ner'])

	# Do lemmatization keeping only noun, adj, vb, adv
	data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN','VERB'])
	# print(data_lemmatized)
	# data_lemmatized= lemmatizer(data_words)
	print()	

	print(">>remove stopwords...")

	# data_words = ''.join(str(v) for v in data_lemmatized)
	# data_words=remove_noise(data_words)
	
	# # Remove Stop Words
	data_words = remove_stopwords(data_lemmatized)
	# print(data_words)
	return data_words

#call the preprocess function
print()
print("Doing pre-processing..")
t0 = time()	
# data_lemmatized=list()
data_lemmatized=pre_process(data)
print("done in %0.3fs." % (time() - t0))
print()
print("#" * 80)


print("create dictionary...")
print()
t0 = time()
# Create Dictionary

# data_lemmatized= [d.split() for d in data_lemmatized]
# data_lemmatized= [data_lemmatized.split()]
# data_lemmatized=' '.join((str(v) for v in data_lemmatized))
id2word= gensim.corpora.Dictionary(data_lemmatized)
print(id2word)
# id2word = corpora.Dictionary(data_lemmatized)
id2word.filter_extremes(no_below=3, no_above=0.6, keep_n=10000)
id2word.compactify()
print()
print("done in %0.3fs." % (time() - t0))
print("#" * 80)


print("document Term Frequency...")
print()
t0 = time()
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# print(corpus)
print("done in %0.3fs." % (time() - t0))
print("#" * 80)

#save the corpus and dictionary
import joblib
print("saving corpus and dictionary..")

joblib.dump(corpus, 'Newcorpus_corpus.pkl')

joblib.dump(id2word,'Newid2word.gensim')

print("saved successfully..")

print("done in %0.3fs." % (time() - t0))
print()
print("#" * 80)


#build and save model
print("build model...")
print()
t0 = time()
# Build LDA model
num_topics=20
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=20,
                                           alpha=0.02,
                                           eval_every = 10,
                                           eta=0.02,
                                           per_word_topics=True)


joblib.dump(lda_model,'Newmodel.gensim')
print("saved")
print("done in %0.3fs." % (time() - t0))
print()


# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
print()
print("done in %0.3fs." % (time() - t0))
print("=" * 80)


print("coherence and perplexity..")
print()
t0 = time()
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
print()
print("done in %0.3fs." % (time() - t0))
print("=" * 80)


print("USING TEST DATA.")
test_doc = 'Great structures are build to remember an event happened in the history.'.split()
# test_doc = lemmatization(test_doc, allowed_postags=['NOUN','VERB'])
# test_doc =' '.join((str(v) for v in test_doc))
print(test_doc)
# test_doc=list()
bow_test_doc = id2word.doc2bow(test_doc)
print(lda_model.get_document_topics(bow_test_doc))
# print(lda_model.get_term_topics(bow_test_doc))
print("done in %0.3fs." % (time() - t0))
print()
