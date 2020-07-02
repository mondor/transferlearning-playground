import argparse
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import torch.nn.functional as F
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, Sentence
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path
import numpy as np
import random
from flair.training_utils import store_embeddings


train_on_gpu = torch.cuda.is_available()

def oversample_df(df, classes):
    classes_count = []
    for c in classes:    
        classes_count.append(len(df.loc[df[c] == 1]))
    
    max_count = max(classes_count)
    resample_ratios = [round(max_count/c) for c in classes_count]
            
    resampled = []
    for i in range(len(resample_ratios)):
        c = classes[i]
        ratio = resample_ratios[i]        
        for r in range(ratio):            
            resampled.append(df.loc[df[c] == 1])
            
    resampled_df = pd.concat(resampled, ignore_index=True)
    resampled_df = resampled_df.sample(frac=1)
    resampled_df = resampled_df.reset_index(drop=True)
    return resampled_df


def undersample_df(df, classes):
    classes_count = []
    for c in classes:    
        classes_count.append(len(df.loc[df[c] == 1]))
    
    min_count = min(classes_count)
    
    resampled = []
    for c in classes:
        resampled.append(df[df[c] == 1][:min_count])
        
    resampled_df = pd.concat(resampled, ignore_index=True)
    resampled_df = resampled_df.sample(frac=1)
    resampled_df = resampled_df.reset_index(drop=True)
    return resampled_df


def get_batches(df, target_names, mode=None, batch_size=16):
    if mode == 'oversample':
        df = oversample_df(df, target_names)
    elif mode == 'undersample':
        df = undersample_df(df, target_names)
        
    df = df.sample(frac=1).reset_index(drop=True)
    for i in range(0, len(df), batch_size):
        x = []
        y = []
        for _, row in df[i:i+batch_size].iterrows():
            image_concept = '' if pd.isna(row['image_concept']) else row['image_concept']
            message = '' if pd.isna(row['message']) else row['message']                        
            
            # shuffle image concepts
            words = image_concept.split()
            random.shuffle(words)
            image_concept = ' '.join(words)
            
            # join message and image_concept together
            txt = ' '.join([message, image_concept])                    
            x.append(Sentence(txt))                        
            y.append([row[t] for t in target_names])
        
        yield x, torch.FloatTensor(y)

        
def train_model(model, epochs, lr, train_df, val_df, target_names, checkpoint_file, early_stopping=5):        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = np.inf
    no_improvement = 0

    if train_on_gpu:
        model = model.cuda()
        
    for epoch in range(epochs):        
        total_train_loss = 0
        total_val_loss = 0
        train_loss = 0
        val_loss = 0
        
        # Train
        model.train()        
        for i, (sentences, labels) in enumerate(get_batches(train_df, target_names, 'oversample')):
            if train_on_gpu:
                labels = labels.cuda()
            
            optimizer.zero_grad()
            
            out = model(sentences)            
                        
            loss = criterion(out, labels)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            store_embeddings(sentences, 'cpu')
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, batch {i}, train loss {loss.item()/labels.size(0)}")
            
            
        train_loss = total_train_loss/len(train_df)

        
        # Eval
        model.eval()
        all_pred = np.array([])
        all_labels = np.array([])
        for sentences, labels in get_batches(val_df, target_names):
            if train_on_gpu:
                labels = labels.cuda()
            
            out = model(sentences)
            loss = criterion(out, labels)
            total_val_loss += loss.item()
            
            store_embeddings(sentences, 'cpu')
            
            pred = torch.round(torch.sigmoid(out))
            pred_np = pred.data.cpu().numpy() if train_on_gpu else pred.data.numpy()
            labels_np = labels.data.cpu().numpy() if train_on_gpu else labels.data.numpy()
            all_pred = np.concatenate([all_pred, pred_np.flatten()])
            all_labels = np.concatenate([all_labels, labels_np.flatten()])

            
        val_loss = total_val_loss / len(val_df)
        f1 = f1_score(all_labels, all_pred, average='weighted')
        acc = accuracy_score(all_labels, all_pred)
        
        print(f"Epoch {epoch}, train loss {train_loss}, val loss {val_loss}, accuracy {acc}, f1 {f1}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            no_improvement = 0
            torch.save(model.state_dict(), checkpoint_file)
            print(f"Save model at Epoch {epoch}, train loss {train_loss}, val loss {val_loss}, accuracy {acc}, f1 {f1}")
        else:
            no_improvement += 1
            print("No improvement.")
            if no_improvement >= early_stopping:
                print(f"Early Stopping at Epoch {epoch}")
                break
                

def get_dfs(data_dir, column_names):    
    csv_file = os.path.join(data_dir, 'preprocess_data.txt')
    df = pd.read_csv(csv_file, names=column_names, engine='python')    
    df = df.loc[df['message'].notnull() & df['image_concept'].notnull()]            
    train_df, validation_df = train_test_split(df, test_size=0.3, random_state=42)    
    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)    
    return train_df, validation_df
    
                    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--early_stopping', type=int, default=5)
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--column_names', type=str, required=True)
    parser.add_argument('--target_names', type=str, required=True)    
    args = parser.parse_args()
    
    
    column_names = args.column_names.split(',')
    target_names = args.target_names.split(',')
    
    train_df, validation_df = get_dfs(args.data_dir, column_names)
    
    # summary
    for target in target_names:
        train_target_count = len(train_df.loc[train_df[target] == 1])
        val_target_count = len(validation_df.loc[validation_df[target] == 1])
        print(f"class: ({target}), train: {train_target_count}, val: {val_target_count}")
        
    document_embeddings = DocumentRNNEmbeddings([
            WordEmbeddings('twitter'),
        ], 
        hidden_size=128,
        reproject_words=True,
        reproject_words_dimension=128
    )
    
    classifier = TextClassifier(
        document_embeddings, 
        label_dictionary=target_names,
        multi_label=True
    )
    
    print(classifier)
    
    checkpoint_file = os.path.join(args.model_dir, 'model.pt')
    lr = 0.001       
    train_model(
        classifier, 
        args.epochs, 
        lr, 
        train_df, 
        validation_df, 
        target_names, 
        checkpoint_file,
        early_stopping=args.early_stopping
    )
    
    print("success!")