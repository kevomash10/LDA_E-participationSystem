import joblib
from time import time
from nltk.stem.wordnet import WordNetLemmatizer
from profanity_check import predict, predict_prob
import spacy
import matplotlib.pyplot as plt
import pandas as pd

loaded_corpus = joblib.load('Newcorpus_corpus.pkl')
loaded_id2word = joblib.load('Newid2word.gensim')
loaded_model = joblib.load('Newmodel.gensim')

print("USING TEST DATA.")  
t0 = time()
test_doc = input("Enter text to do topic model distribution: ").split()

#test_doc = 'Nasa will launch a space shuttle in the near future in fucking history '.split()
print(test_doc)
print()

noise_list = ["a","is","be","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"] 
def remove_noise(input_text):
    words = input_text
    noise_free_words = [word for word in words if word not in noise_list] 
    # noise_free_text = " ".join(noise_free_words) 
    return noise_free_words
lemm=WordNetLemmatizer()
def lemmatizer(input_text):
    words = input_text
    lem_words = [lemm.lemmatize(word,"v") for word in words] 
    # lem_text = " ".join(lem_words) 
    return lem_words

def offensive(input_text):
    words = input_text
    offensive_words = [predict([word])for word in words] 
    for i in range(len(offensive_words)):
    	if offensive_words[i] ==[1]:
    		print()
    		print('this sentence contain some offensive word/s')
    	# print(offensive_words[i])
    # print(offensive_words[i] for i in [len(offensive_words)])
    # offensive_text = " , ".join(str([word])for word in offensive_words) 
    return offensive_words


test_doc=remove_noise(test_doc)
print(test_doc)
print()
# test_doc=offensive(test_doc)
# print(test_doc)
test_doc=lemmatizer(test_doc)
print(test_doc)
print()

# lda_model[corpus[0]]
bow_test_doc = loaded_id2word.doc2bow(test_doc)
print(bow_test_doc)
print()

print(loaded_model.get_document_topics(bow_test_doc))
print()

test_doc=offensive(test_doc)
print(test_doc)
print()
# print(loaded_model.get_term_topics(bow_test_doc))
# # loaded_model.update(bow_test_doc)
# print("done in %0.3fs." % (time() - t0))
# print()


# print("USING TEST DATA.")
# test_doc = 'Great structures are build to remember an event happened in the history.'.split()
# # test_doc = lemmatization(test_doc, allowed_postags=['NOUN','VERB'])
# # test_doc =' '.join((str(v) for v in test_doc))
# print(test_doc)
# # test_doc=list()
# bow_test_doc = id2word.doc2bow(test_doc)
# print(lda_model.get_document_topics(bow_test_doc))
# # print(lda_model.get_term_topics(bow_test_doc))
# print("done in %0.3fs." % (time() - t0))
# print()
