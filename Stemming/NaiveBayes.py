import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

class word_chart(object):
    text = ''
    rates = np.array([1,1,1,1,1])
    def __init__(self, text, rates):
        self.text = text
        self.rates = rates
    def __str__(self):
        res= 'Word: '+self.text+"\n"+"Rates: " + str(self.rates)
        return res

def in_Dictionary(Dictionary, word):
    for id in range(len(Dictionary)):
        if(word==Dictionary[id].text):
            return True,id
    return False,0

def learnDictionary(Dataset,Dictionary=[], Normalisation=True):

    # nbr=0 # Number of comment

    # for comment in Dataset.text:
    for nbr, comment in enumerate(Dataset.text):
        try:
            words = word_tokenize(comment)

            for word in words:
                bool,id = in_Dictionary(Dictionary,word)

                if(not bool):
                    # The words is not in the Dictionary
                    Wds = word_chart(word,np.array([1,1,1,1,1])) # Laplace smoothing
                    Dictionary.append(Wds)
                    id = len(Dictionary)-1

                star = Dataset.stars[nbr]-1
                Dictionary[id].rates[star] += 1
        except:
            print(comment)

        # nbr = nbr + 1
    sums = np.zeros(5)  # total numbers of times words which are in comments of a given rate

    if (Normalisation):
        # Normalisation
        for i in range(len(Dictionary)):
            sums += Dictionary[i].rates
        for i in range(len(Dictionary)):
            Dictionary[i].rates = Dictionary[i].rates/len(Dataset.text)
        # # Computation of the ratios
        # for i in range(len(Dictionary)):
        #     Dictionary[i].rates = (Dictionary[i].rates) / (1 - Dictionary[i].rates)

    return Dictionary

def aPrioriProbablities(Dataset,aPriori):

    nbr = 0  # Number of comment
    total = np.sum(aPriori) # because of the Laplace smoothing above
    for comment in Dataset.text:
        try:
            words = word_tokenize(comment)
            total += len(words)
            star = Dataset.stars[nbr] - 1
            aPriori[star] +=  len(words)
            nbr = nbr + 1
        except:
            print(comment)
    return aPriori/total

def aPrioriProbablities2(Dataset,aPriori):

    nbr = 0  # Number of comment
    total = np.sum(aPriori) # because of the Laplace smoothing above
    for rate in Dataset.stars:
        aPriori[rate-1] += 1
        total += 1

    return aPriori/total

def outputCalculation(Dictionary,test, aPriori,Testset):
    res,i = np.zeros(len(Testset)),0
    for comment in Testset.text:
        try:
            words = word_tokenize(comment)
            P = np.zeros(5)
            for word in words:
                bool,id = in_Dictionary(Dictionary,word)
                _,id2 = in_Dictionary(test,word)
                if (bool):
                    P += np.log(Dictionary[id].rates)*test[id2].rates
            # print(P)
            res[i] = np.argmax(P + np.log(aPriori))+1
            i += 1
        except:
            print(comment)
    return res

if __name__ == '__main__':

    # Training the data
    Dataset = pd.read_csv("train_df_stemmed.csv")
    Dictionary = learnDictionary(Dataset)

    # Most probable word for each star
    n=10
    m,name=np.zeros(n),np.reshape([' ']*5*n,(n,5)).tolist()
    M = np.ones(n)
    for j in range(n):
        for mot in Dictionary:
            for i in range (5):
                # print('word: ' + str(mot.rates[i]) + ' loc_max:' + str(m[i])+'overallMAx'+str(M[i]))
                if (mot.rates[i]>m[i] and mot.rates[i]<M[i] and not(mot.text in name[:][i])):
                    m[i] = mot.rates[i]
                    name[j][i] = mot.text
        M=m
        print(name[:][j])
        # print(m)
        m = np.zeros(5)

    # print(Dictionary[0])

    # Output computation
    aPriori = np.ones(5)
    aPriori = aPrioriProbablities2(Dataset,aPriori)
    print(aPriori)
    Testset = pd.read_csv("test_df_stemmed.csv")
    test = learnDictionary(Testset, Dictionary=[], Normalisation=False)
    LR = outputCalculation(Dictionary, test, aPriori,Testset)
    S = [s for s in Testset.stars]
    print('ratio of good predictions: ' + str(len(np.where(abs(LR-S)==0)[0])/len(S)))