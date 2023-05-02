"""
@author:"Edmond Menya"
@created:"05-05-2023"

"""

import os
import re
import csv
import itertools
import json
import sys
import torch
import numpy as np
import sacremoses as ms
from optparse import OptionParser
from transformers import BertForTokenClassification, AdamW, AutoTokenizer, AutoModelForMaskedLM, AutoModel, ElectraForPreTraining, ElectraTokenizerFast
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences


if __name__ == "__main__":
    
    from data import dataGenerator
    import config as cfg
    from models import modelTrainTest as mtt
    from data.profile import Profile
    
    if not sys.argv:
        modelChoice=sys.argv[0]
        
    else:
        optparser = OptionParser()
        optparser.add_option('-a', '--modelChoice',
                             dest='modelChoice',
                             help='select model',
                             default=cfg.MODEL_CHOICE,
                             type='string')
        (options, args) = optparser.parse_args()
    
    modelChoice=options.modelChoice
    rel_data=cfg.rel_url
    irr_data=cfg.irr_url
    max_len=cfg.MAX_LEN
    batch_size=cfg.BATCH_SIZE
    num_folds=cfg.NUM_FOLDS
    epochs=cfg.EPOCHS
    corpus_type=cfg.CORPUS_TYPE
    
    #print(sys.argv[0])
    
    # Get GPU device name
    #device_name = tf.test.gpu_device_name()

    """if device_name == '/device:GPU:0':
      print('Found GPU at: {}'.format(device_name))
    else:
      raise SystemError('GPU device not found')"""
      
    
    # tell Pytorch to use the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    print('We are running ', modelChoice)
    
    data = dataGenerator.data_generator(rel_data,irr_data,corpus_type)
    print(data.head())
    data[corpus_type] = dataGenerator.text_cleaner(data[corpus_type])
    data['Length'] = data[corpus_type].apply(lambda row: min(len(row.split(" ")), len(row)) if isinstance(row, str) else None)
    print(data.head())

    if corpus_type == 'title':
        data = data[data['Length']>1]#ignore any title with less than 1 word
        data = data[data['Length']<20]#ignore any title with more than 20 words
    elif corpus_type == 'text':
        data = data[data['Length']>50]#ignore any corpus with less than 50 words
        data = data[data['Length']<1000]#ignore any corpus with more than 1000 words
    
    
    if modelChoice == 'epidbioelectra':
        tokenizer = ElectraTokenizerFast.from_pretrained("kamalkraj/bioelectra-base-discriminator-pubmed")

    elif modelChoice == 'scibert':
        tokenizer =AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')

    elif modelChoice == 'clinicalbert':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    elif modelChoice == 'pubmedbert':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

    elif modelChoice == 'epidbiobert':
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        
    elif modelChoice == 'transformerxl':
        tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
    
    label_encoder=LabelEncoder()
    label_encoder.fit(data['Label'])
        
    X=list(data[corpus_type])
    y=list(label_encoder.transform(data['Label']))
        
    tok_texts_and_labels = [dataGenerator.tok_with_labels(sent, labs, tokenizer) for sent, labs in zip(X, y)]
        
    tok_texts = [tok_label_pair[0] for tok_label_pair in tok_texts_and_labels]
    labels = np.array([tok_label_pair[1] for tok_label_pair in tok_texts_and_labels])
        
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tok_texts],maxlen=max_len, dtype="long", value=0.0,truncating="post", padding="post")
                          
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
        
    print('Inputs shape is ',input_ids.shape)
        
    acc_per_fold,f_score_per_fold,prec_per_fold,rec_per_fold, org_labels_per_fold, preds_per_fold = mtt.k_fold_cross_val(num_folds,epochs,batch_size,max_len,input_ids,labels,attention_masks,device,modelChoice,tokenizer,list(label_encoder.classes_))
        
    acc_list=[]
    for i in acc_per_fold:
        acc_list.append(np.round(i.tolist(),4))
          
    print(acc_list)
    #Write results to file
    wr_text=str('------------------Model: '+modelChoice+' --------------------\n\n')
    #wr_text+=str('Accuracy per fold: '+str(acc_per_fold)+'\n')
    wr_text+=str('F_score per fold: '+str(f_score_per_fold)+'\n')
    wr_text+=str('Precision per fold: '+str(prec_per_fold)+'\n')
    wr_text+=str('Recall per fold: '+str(rec_per_fold)+'\n')
    wr_text+=str('\n\n\nOriginal per fold: '+str(org_labels_per_fold)+'\n')
    wr_text+=str('\n\n\nPredicted probs per fold: '+str(preds_per_fold)+'\n')

    wr_text+=str('Accuracy List: '+str(acc_list)+'\n')
    #wr_text += str(acc_list)

    f_name = str('Shap_'+corpus_type+'_'+modelChoice+'_res.txt')
    Profile.write_file(wr_text, f_name)

