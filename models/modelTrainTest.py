"""
Module contains methods to train and test epidemiological corpus taggers found in models.py

"""

import torch
import shap
import pickle
import json
import numpy as np
import torch.nn as nn
from models.classes import Models
from collections import defaultdict, OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score,precision_score,recall_score
from transformers import AdamW, get_linear_schedule_with_warmup

__author__ = "Edmond Menya"
__email__ = "edmondmenya@gmail.com"

def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler,max_grad_norm): #here
    """
    Trains model for every epoch
    :param model: object of epid model being run
    :param data_loader: object of train dataset
    :param loss_fn: loss function to be used in training
    :param optimizer: optimizer algorithm to be used in training
    :param device: GPU device being used to run model
    :param scheduler: scheduler value to reduce learning rate as training progresses
    :param max_grad_norm: grad norm value for gradient clipping
    :return: computed train accuracy, computed train loss
    """
    model = model.train()
    losses = []
    correct_predictions = 0.
    batch_no = 1
    for step,batch in enumerate(data_loader):
        """
        Iterates over the dataset for every batch taking them into the model for training iterations
        """

        batch = tuple(t.to(device) for t in batch) #here
        #batch = tuple(t for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        outputs = model(b_input_ids,b_input_mask)
        #outputs = torch.tensor(model(b_input_ids,b_input_mask)).to(device)
        y_hat=outputs.argmax(-1)
        #_,preds = torch.max(outputs,dim=2)

        outputs = outputs.view(-1,outputs.shape[-1])
        b_labels_shaped = b_labels.view(-1)

        #loss=loss_fn(outputs,b_labels_shaped)
        loss = loss_fn(torch.log(outputs + 1e-20),b_labels_shaped)#modified loss to take in probs instead of logits
        #loss = Variable(loss, requires_grad = True)

        correct_predictions += torch.sum(y_hat == b_labels_shaped)
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        batch_no=batch_no+1

    return correct_predictions/len(data_loader.dataset) , np.mean(losses)

def binarize_y(y_true):
  bin_y=[]
  for y in y_true:
    if y == 0:
      bin_y.append([1,0])
    else:
      bin_y.append([0,1])
  return np.array(bin_y)

def model_eval(model,data_loader,loss_fn,device): #here
    """
    Evaluates the model after training by computing test accuracy and error rates
    :param model: object of epid model being tested
    :param data_loader: object of test dataset
    :param loss_fn: loss function to be used in testing
    :param device: GPU device being used to run model
    :return: test accuracy,test loss,f_score value,precision value,recall value
    """
    model = model.eval()
    
    losses = []
    correct_predictions = 0.

    original_labels, new_labels, pred_probs = [], [], []
    
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
        
            outputs = model(b_input_ids,b_input_mask)
            #outputs = torch.tensor(model(b_input_ids,b_input_mask)).to(device)
            y_hat=outputs.argmax(-1)
            
        
            #probs,_ = torch.max(outputs,dim=1)
            raw_probs = outputs

            outputs = outputs.view(-1,outputs.shape[-1])
            b_labels_shaped = b_labels.view(-1)

            #print("Outs NI ",y_hat,"\n")
            #print("Original NI ",b_labels_shaped,"\n")
            

            #loss = loss_fn(outputs,b_labels_shaped)
            loss = loss_fn(torch.log(outputs + 1e-20),b_labels_shaped)#modified loss to take in probs instead of logits

            correct_predictions += torch.sum(y_hat == b_labels_shaped)
            losses.append(loss.item())

            predicted_indices = y_hat.to('cpu').numpy()
            original_indices = b_labels_shaped.to('cpu').numpy()
            

            for label_idx,pred_idx,prob_val in zip(predicted_indices,original_indices,raw_probs.tolist()):
              new_labels.append(label_idx)
              original_labels.append(pred_idx)
              pred_probs.append(prob_val)

            #print("HAPA: ",predicted_indices,"\n")
            #print("NA HAPA: ",original_indices)

        prec = precision_score(original_labels,new_labels,pos_label=1)
        rec = recall_score(original_labels,new_labels,pos_label=1)
        fscore = f1_score(original_labels,new_labels,pos_label=1)

        print('\nPrecision_Score: ',prec)
        print('\nRecall_Score: ',rec)
        print('\nF_Score: ',fscore)
    return correct_predictions/len(data_loader.dataset) , np.mean(losses), fscore, prec, rec , binarize_y(original_labels), np.array(pred_probs)


def k_fold_cross_val(num_folds,epochs,batch_size,max_len,input_ids,labels,attention_masks,device,model_choice,tokenizer,class_names):
    """
    Trains and tests epid model using cross validation technigue and averages the cross validated models performance
    :param num_folds: value of k for the k-fold corss validator
    :param epochs: number of iterations to run for every k-fold instance
    :param batch_size: the set batch value for dataset segmentations
    :param max_len: the set max token length per corpus
    :param input_ids: token position ids for input corpus words as tokenized by model pretrained tokenizer
    :param labels: labels for corpus in dataset
    :param attention_masks: attention mask that corresponds with input_id token positions
    :param device: GPU device being used to run k-folded models
    :return: accuracy for each fold,f_score value for each fold,precision value for each fold,recall value for each fold
    """
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    split_size=.2
    acc_per_fold, f_score_per_fold, prec_per_fold, rec_per_fold = [],[],[],[]
    org_labels_per_fold, pred_probs_per_fold = {}, {}
    
    for train, test in kfold.split(input_ids, labels):
        print(f"\n ****************************** This is Fold Number {fold_no} ***************************** \n")
        
        no_target_classes=len(np.unique(labels))+1

        tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids[train], labels[train],
                                                                random_state=2018, test_size=split_size)
        tr_masks, val_masks, _, _ = train_test_split(np.array(attention_masks)[train], input_ids[train],
                                                  random_state=2018, test_size=split_size)

        n_attention_masks = [[float(i != 0.0) for i in ii] for ii in tr_inputs]

        tr_masks, test_masks, _, _ = train_test_split(n_attention_masks, tr_inputs,
                                                  random_state=2018, test_size=split_size)

        tr_inputs, test_inputs, tr_tags, test_tags = train_test_split(tr_inputs, tr_tags,
                                                                  random_state=2018, test_size=split_size)

        tr_inputs = torch.tensor(tr_inputs)
        val_inputs = torch.tensor(val_inputs)
        test_inputs = torch.tensor(test_inputs)


        tr_tags = torch.tensor(tr_tags)
        val_tags = torch.tensor(val_tags)
        test_tags = torch.tensor(test_tags)

        tr_masks = torch.tensor(tr_masks)
        val_masks = torch.tensor(val_masks)
        test_masks = torch.tensor(test_masks)

        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)

        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size, drop_last=True)

        test_data = TensorDataset(test_inputs, test_masks, test_tags)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, drop_last=True)
        
        
        if model_choice == 'epidbioelectra':
            model = Models.EpidBioELECTRA(device)
        elif model_choice =='scibert':
            model = Models.SciBERT(device)
        elif model_choice == 'clinicalbert':
            model = Models.ClinicalBERT(device)
        elif model_choice =='pubmedbert':
            model = Models.PubmedBERT(device)
        elif model_choice == 'epidbiobert':
            model = Models.EpidBioBERT(device)
        elif model_choice == 'transformerxl':
            model = Models.TransformerXL(device)
            
        model.to(device)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
              {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
              'weight_decay_rate': 0.01},
              {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
              'weight_decay_rate': 0.0}
          ]

        optimizer = AdamW(
          optimizer_grouped_parameters,
          lr=1e-5,
          eps=1e-8
        )

        max_grad_norm = 1.0

        total_steps = len(train_dataloader.dataset) * epochs
        warmup_steps = int(epochs * len(train_dataloader.dataset) * 0.1 / batch_size) #mine


        scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=0,
          #num_warmup_steps=warmup_steps,
          num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss().to(device) #here

        #%%time here
        history = defaultdict(list)
        best_accuracy = 0
        normalizer = batch_size*max_len
        loss_values = []

        """for epoch in range(epochs):
          
            total_loss = 0
            print(f'======== Epoch {epoch+1}/{epochs} ========')
            train_acc,train_loss = train_epoch(model,train_dataloader,loss_fn,optimizer,device,scheduler,max_grad_norm) #here
            #train_acc,train_loss = train_epoch(model,train_dataloader,loss_fn,optimizer,scheduler,max_grad_norm)
            #train_acc = train_acc/normalizer
            print(f'Train Loss: {train_loss} Train Accuracy: {train_acc}')
            total_loss += train_loss.item()

            avg_train_loss = total_loss / len(train_dataloader.dataset)  
            loss_values.append(avg_train_loss)

            val_acc,val_loss,_,_,_,_,_ = model_eval(model,valid_dataloader,loss_fn,device) #here
            #val_acc,val_loss,_,_,_ = model_eval(model,valid_dataloader,loss_fn)
            #val_acc = val_acc/normalizer
            print(f'Val Loss: {val_loss} Val Accuracy: {val_acc}')

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)"""

        test_acc,test_loss,f_score,prec,rec,original_labels,pred_probs = model_eval(model,test_dataloader,loss_fn,device)
        #test_acc,test_loss,f_score,prec,rec = model_eval(model,test_dataloader,loss_fn)
        #test_acc = test_acc
        print(f'Test Loss for fold {fold_no}: {test_loss} Test Accuracy: {test_acc} F Score: {f_score} Precision: {prec} Recall: {rec}')
        acc_per_fold.append(test_acc)
        f_score_per_fold.append(f_score)
        prec_per_fold.append(prec)
        rec_per_fold.append(rec)
        org_labels_per_fold[fold_no] = original_labels
        pred_probs_per_fold[fold_no] = pred_probs

        

        #our_cv_acc.append(torch.mean(torch.stack(acc_per_fold)))
        #our_cv_f_score.append(np.mean(f_score_per_fold))

        #############Computing Shap Values from trained Model over the Test Set#################

        """masker = shap.maskers.Text(tokenizer, output_type="token_ids")
        #print("HAPA SAWA \n\n")
        
        explainer = shap.Explainer(model, masker ,output_names=class_names)
        #print("HAPA PIA \n\n")

        batch_no = 1

        for step, batch in enumerate(test_dataloader):
            #print("INA RUN \n\n")
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            print("\nThis is batch",batch_no)

            batch_shap_values = explainer(b_input_ids)
            #shap_values = shap_values + str(batch_shap_values) #gets shap values for all batches

            #############Save computed Shap Values###################

            with open("/mnt/edmond/EpidBioELECTRA/models/Shapely/shaps_"+str(batch_no)+".txt","wb") as shap_file:
                pickle.dump(batch_shap_values,shap_file) #store shap value for each batch in its own file
            batch_no+=1"""
        f_name = str('/mnt/edmond/EpidBioELECTRA/results/labels_'+str(fold_no)+'_res.txt')
        f_name2 = str('/mnt/edmond/EpidBioELECTRA/results/probs_'+str(fold_no)+'_res.txt')

        with open(f_name,"w") as my_file:
              json.dump(original_labels.tolist(),my_file)

        with open(f_name2,"w") as my_file2:
              json.dump(pred_probs.tolist(),my_file2)

        fold_no = fold_no + 1

        
      
    return acc_per_fold,f_score_per_fold,prec_per_fold,rec_per_fold, org_labels_per_fold, pred_probs_per_fold

    
