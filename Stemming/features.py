import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pickle as pk
################################################# To install nltk #####################################################

# Si t'as anaconda: conda install nltk (sur le terminal)
# Sinon: pip install nltk (sur le terminal)
# nltk.download() (sur python)

#######################################################################################################################

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

comun=open('comunWords.txt','r')
tab = []
for w in comun.readlines():
    tab.append(w[:-1])

def features(filename,n=-1):

    review = pd.read_csv(filename + ".csv")

    ps=PorterStemmer()
    reviews=[]
    for comment in review.text[:n]:
        try:
            words=word_tokenize(comment)
        except:
            print(comment)
        wds=''
        for w in words:
            wstmed=ps.stem(w)
            if not(wstmed in tab):
                # wds.append(wstmed)
                wds=wds+' '+wstmed
        reviews.append(wds)

    for i in range(len(reviews)):
        review.text[i]=reviews[i]
    review = pd.DataFrame({'text': review.text[:n], 'stars': review.stars[:n]})
    review.to_csv(filename+"_stemmed.csv", index=False)


if __name__ == '__main__':

    nbr=200
    features("train_df",n=3*nbr)
    print("train_done")
    features("test_df",n=nbr)
    print("test_done")

