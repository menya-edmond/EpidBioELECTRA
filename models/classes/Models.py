"""
Module contains models necessary to perform epidemiological corpus tagging and returning their label as either relevant or irrelevant

"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForMaskedLM, AutoModel, ElectraForPreTraining, ElectraTokenizerFast, TransfoXLModel

__author__ = "Edmond Menya"
__email__ = "edmondmenya@gmail.com"

class EpidBioELECTRA(nn.Module):
    """
    EpidBioELECTRA Model addopted from pretrained BioELECTRA on huggingface library built on pytorch classes
    """

    def __init__(self,device):
        """
        Constructor for the deep attention model, layers are adopted from pytorch and weights pretrained on BioELECTRA LM
        :param no_target_classes: no of corpus target classes defines models last layer architecture
        """
        super().__init__()
        self.dv = device
        self.bert = ElectraForPreTraining.from_pretrained("kamalkraj/bioelectra-base-discriminator-pubmed")
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm=nn.BatchNorm1d(1,affine=False)
        self.output = nn.Linear(self.bert.config.hidden_size, no_target_classes)
        self.output = nn.Linear(self.bert.config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        """
        Main network architecture as defined in pytorch library.
        :param input_ids: token position ids for input corpus words as tokenized by model pretrained tokenizer
        :param attention_mask: attention mask that corresponds with input_id token positions
        :return: probability distribution over predicted corpus classes, max value of the predicted probability distribution
        """
        if type(input_ids) is np.ndarray: #added to accomodate computation of shap values which come in as tensors
            input_ids=torch.tensor(input_ids).to(self.dv)
            attention_mask=torch.tensor(attention_mask).to(self.dv)
            
            if input_ids.shape != attention_mask.shape:
                #print("Shape was altered from",attention_mask.shape)
                attention_mask = attention_mask[:list(input_ids.shape)[0]]
                #print(" to ",attention_mask.shape,"\n")
            
        encoded_layer = self.bert(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True)
        encl = encoded_layer[-1][-1]
        max_pooled, _  = torch.max(encl, 1)
        out = self.output(max_pooled)
        if len(list(out.shape)) >= 3:
            out=torch.squeeze(out)

        out = self.softmax(out)
            
        norm=torch.squeeze(self.batchnorm(torch.unsqueeze(max_pooled,1)),0)
        dropped = self.dropout(norm)
        out = self.output(dropped)
        norm=torch.squeeze(torch.unsqueeze(max_pooled,1),0) #removed norm
        out = self.output(norm) #removed drop out

        """else:
            print("Input shape is ",input_ids.shape," att shape is ",attention_mask.shape,"\n")
            print(input_ids,"\n\n")
            print(attention_mask,"\n\n")
            #pass

        #return out, out.argmax(-1)"""
        
        return out
