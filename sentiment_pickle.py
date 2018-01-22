import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteCLassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifier=classifiers
    
    def classify(self,features):
        votes=[]
        for c in self._classifier:
            v=c.classify(features)
            votes.append(v)
        return mode(set(votes))
    
    def confidence(self,features):
        votes=[]
        for c in self._classifier:
            v=c.classify(features)
            votes.append(v)
        choice_votes=votes.count(mode(votes))
        conf=choice_votes/len(set(votes))
        return conf
short_pos=open("../positive.txt","r").read()
short_neg=open("../negative.txt","r").read()

all_word=[]
documents=[]

#j is adject, r is adverb and v is verb
allowed_word_type=["J"]
for p in short_pos.split('\n'):
    documents.append({p,"pos"})
    words=word_tokenize(p)
    pos=nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_type:
            all_word.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append({p,"neg"})
    words=word_tokenize(p)
    neg=nltk.pos_tag(words)
    for w in neg:
        if w[1][0] in allowed_word_type:
            all_word.append(w[0].lower())
            
save_documents=open("../documents.pickle","wb")
pickle.dump(documents,save_documents)
save_documents.close()

all_word=nltk.FreqDist(all_word)
word_features=list(all_word.keys())[:5000]

save_documents=open("../word_features.pickle","wb")
pickle.dump(word_features,save_documents)
save_documents.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets=[(find_features(rev),category) for (rev,category)in documents]

save_documents=open("../featuresets.pickle","wb")
pickle.dump(featuresets,save_documents)
save_documents.close()


classifier=nltk.NaiveBayesClassifier.train(training_set)
print("Original NaiveB",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

save_documents=open("../classifier.pickle","wb")
pickle.dump(classifier,save_documents)
save_documents.close()

MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_Classifier",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)
save_documents=open("../MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier,save_documents)
save_documents.close()


BernoulliNB_classifier=SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier",(nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)
save_documents=open("../BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier,save_documents)
save_documents.close()

LogisticRegression_classifier=SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier",(nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)
save_documents=open("../LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier,save_documents)
save_documents.close()

LinearSVC_classifier=SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier",(nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)
save_documents=open("../LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier,save_documents)
save_documents.close()

SGDC_classifier=SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("LinearSVC_classifier",(nltk.classify.accuracy(SGDC_classifier,testing_set))*100)
save_documents=open("../SGDC_classifier.pickle","wb")
pickle.dump(SGDC_classifier,save_documents)
save_documents.close()


voted_classifier = VoteCLassifier(
                                classifier,
                                LinearSVC_classifier,
                                MNB_classifier,
                                BernoulliNB_classifier,
                                LogisticRegression_classifier
                                )
print("Voteed_classifier Accuracy",(nltk.classify.accuracy(voted_classifier,testing_set))*100)
def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
