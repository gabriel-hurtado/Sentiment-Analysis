import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

class word_chart(object):
    text = ''
    rates = [1,1,1,1,1]
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

    nbr=0 # Number of comment

    for comment in Dataset.text:
        try:
            words = word_tokenize(comment)

            for word in words:
                bool,id = in_Dictionary(Dictionary,word)

                if(not bool):
                    # The words is not in the Dictionary
                    Wds = word_chart(word,[1,1,1,1,1]) # Laplace smoothing
                    Dictionary.append(Wds)
                    id = len(Dictionary)-1

                star=Dataset.stars[nbr]-1
                Dictionary[id].rates[star] += 1
        except:
            print(comment)
        nbr = nbr + 1
    sums = np.zeros(5)  # total numbers of times words which are in comments of a given rate

    if (Normalisation):
        # Normalisation
        for i in range(len(Dictionary)):
            sums += Dictionary[i].rates
        for i in range(len(Dictionary)):
            Dictionary[i].rates = Dictionary[i].rates/sums
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

def outputCalculation(Dictionary,test, aPriori,Testset):
    res,i = np.zeros(len(Testset)),0
    for comment in Testset.text:
        words = word_tokenize(comment)
        P = np.zeros(5)
        for word in words:
            bool,id = in_Dictionary(Dictionary,word)
            _,id2 = in_Dictionary(test,word)
            if (bool):
                P += np.log(Dictionary[id].rates)*test[id2].rates+np.log(aPriori)
        # print(P)
        res[i] = np.argmax(P)+1
        i += 1
    return res

if __name__ == '__main__':

    Dataset = pd.read_csv("train_df_stemmed.csv")
    Dictionary = learnDictionary(Dataset)
    # print(len(Dataset.text))
    # for i in range(5):
    #     print((Dictionary[i]))

    # m,name=np.zeros(5),['','','','','']
    # for mot in Dictionary:
    #     for i in range (5):
    #         if (mot.rates[i]>m[i]):
    #             m[i] = mot.rates[i]
    #             name[i] = mot.text
    # print(name)
    # print(m)
    # print(total)
    
    aPriori = np.ones(5)
    aPriori = aPrioriProbablities(Dataset,aPriori)
    print(aPriori)

    Testset = pd.read_csv("test_df_stemmed.csv")
    test = learnDictionary(Testset, Dictionary=[], Normalisation=False)
    # for i in range(5):
    #     print((test[i]))
    LR = outputCalculation(Dictionary, test, aPriori,Testset)
    S=[s for s in Testset.stars]
    # print(np.linalg.norm(S-LR,2))
    # print(LR)
    # print(S)