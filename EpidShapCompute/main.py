from nltk.util import ngrams
from collections import defaultdict
import shap
import json
import pickle
import os
import itertools
import re
import time
import sys
import threading


if __name__ == "__main__":

    ngramm = 3
    how_many_shap_files = 139
    corpora = 'Shapely/' # place your computed shap files address here
    shaps_dict, ff_dict = {}, {}

    done = False
    def animate():
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if done:
                break
            sys.stdout.write('\rloading ' + c)
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\rDone!     ')

    t = threading.Thread(target=animate)
    t.daemon=True

    def ngram_impact(gram_limit,shap_values):
      features = [list(t) for t in [x for x in shap_values[:,:,"relevant"].data]] # all words in one batch of test set
      features = list(itertools.chain(*features)) 

      impact=[list(t) for t in [x for x in shap_values[:,:,"relevant"].values]] # the impact values of these words on the batch document they occur in
      impact = list(itertools.chain(*impact))

      f_list,i_list=[],[]
      i=1
      for i in range(i,gram_limit+1):
        if i==1:
          f_list += features
          i_list += impact
        else:
          ngram_features = [' '.join(e) for e in ngrams(features, i)]
          ngram_impact = [sum(list(e)) for e in ngrams(impact, i)]
          f_list += ngram_features
          i_list += ngram_impact

      return f_list,i_list

    

    for subdir, dirs, files in os.walk(corpora):
      #print(dirs)
      x=1
      for file in files:

        #print(data_file)
        path = os.path.join(subdir, file)
        print(path)
        #print("Itr: ",x,"\n")
        
        with open(path,"rb") as shap_file:
          shaps_dict[x] = pickle.load(shap_file)
        
        x+=1

    feat_dict, imp_dict = defaultdict(list), defaultdict(list)
    for key,shaps in shaps_dict.items():
      print("\t*")
      features,impact=ngram_impact(ngramm,shaps)
      feat_dict[key] = [re.sub(' +', ' ', f.strip()) for f in features]
      imp_dict[key] = impact
    

    features = feat_dict[1]
    impact = imp_dict[1]

    for i in range(2,how_many_shap_files+1):
      features += feat_dict[i]
      impact += imp_dict[i]

    print("\nLen of features is",len(features),"\n")

    t.start()
    for feat in features:
      #print(".")
      
      index_list =[idx for idx, el in enumerate(features) if el == feat]
      ss=0
      for x in index_list:
        #print("\t_")
        ss=ss+impact[x]
      ff_dict[feat]=ss
    time.sleep(10)
    done = True
    print("DONE Processing")

    with open("features_dict.txt","w") as f_file:
      f_file.write(json.dumps(ff_dict))

