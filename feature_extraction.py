import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import tqdm
import glob
import urllib.request
import torchvision
import torch
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from torchvision import transforms
import tqdm
import pickle

def get_filtered_df(df_file, all_images_path):
    df = pd.read_csv(df_file, header=None, names=["id", "image", "published", "disabled"])
    df['available'] = 0
    
    for i, row in df.iterrows():
        file = os.path.join(all_images_path, row['id'] + '.jpg')
        if os.path.isfile(file):
            df.at[i, 'available'] = 1

    df = df.loc[df.query('available == 1').index, :]
    return df.reset_index(drop=True)


def to_small_image(df, all_images_path, resized_images_path):    
    os.makedirs(resized_images_path, exist_ok=True)

    for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        _id = row['id']
        file = os.path.join(all_images_path, _id + '.jpg')
        if os.path.isfile(file):
            img = mpimg.imread(file)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(resized_images_path, _id + ".jpg"), img)

            
def feature_extraction(df, images_path, image_map_file):
    model = torchvision.models.resnet50(pretrained=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    inv_normalize = transforms.Normalize(
       mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
       std=[1/0.229, 1/0.224, 1/0.225]
    )

    transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    feature_extraction_model = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extraction_model.eval()
    
    image_map = {}
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        _id = row['id']
        file = os.path.join(images_path, _id + '.jpg')
        img = mpimg.imread(file)
        img = transformer(img)    
        img_rep = feature_extraction_model(img.unsqueeze(0))
        image_map[_id] = img_rep.squeeze().detach().numpy()
    
    with open(image_map_file, 'wb') as f:
        pickle.dump(image_map, f)
            

def main():
    
    all_images_path = 'data/all_images'
    resized_images_path = 'data/small_images'
    image_map_file = 'image_map_small.pkl'
    
    df = get_filtered_df('data.csv', all_images_path)
    feature_extraction(df, resized_images_path, image_map_file)
    
    
if __name__ == '__main__':
    main()
    
    