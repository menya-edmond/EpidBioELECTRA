"""
Configurations for epid Models
Change Variables and Hyperparams from this file
"""

#chose model to train and test, options are['epidbioelectra','epidbiobert','scibert','clinicalbert','pubmedbert','transformerxl']
MODEL_CHOICE='epidbioelectra'


#set hyperparameters
MAX_LEN=128 # 8 | 16 | 128 | 256 | 512
BATCH_SIZE=32 # 8 | 16 | 32 | 64
NUM_FOLDS=5
EPOCHS=5 #50
CORPUS_TYPE='text' # text | title


#set path to relevant and irrelevant corpus folders
#rel_url='data/padi_web/relevant/relevant_articles.csv'
#irr_url='data/padi_web/irrelevant/irrelevant_articles.csv'

rel_url='data/padi_web_large/relevant/relevant_articles.csv'
irr_url='data/padi_web_large/irrelevant/irrelevant_articles.csv'
