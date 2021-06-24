import multiprocessing
import numpy as np 
from tqdm import tqdm

from joblib import Parallel, delayed
import os
import httpx
import argparse
import pandas as pd

def get_image_meta(dataset, image_dir='data/downloaded_posters/'):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        
    # Get image links
    image_links = dataset['Poster'].tolist()
    image_names = dataset['imdbId'].tolist()
    
    return image_links, image_names
    
def download_sequential(image_links, image_names, image_dir='data/downloaded_posters/'):
    for image_link, image_name in tqdm(zip(image_links, image_names), total=len(image_links)):
        if isinstance(image_link, str):
            with httpx.Client() as client:
                results = client.get(image_link)
            
            if results.status_code != 404:
                with open(image_dir + str(image_name) + '.jpg', 'wb') as f:
                    f.write(results.content)
                    
                    
def download_parallel(image_links, image_names, image_dir='data/downloaded_posters/'):
    def download_image(image_link, image_path):
        if isinstance(image_link, str) and not os.path.exists(image_path):
            results = httpx.get(image_link, timeout=None)
            
            if results.status_code != 404:
                with open(image_path, 'wb') as f:
                    f.write(results.content)
        
    num_cores = multiprocessing.cpu_count()
    succes_list = Parallel(n_jobs=num_cores)(delayed(download_image)(image_link, image_dir + str(image_name) + '.jpg') for image_link, image_name in tqdm(zip(image_links, image_names), total=len(image_links)))
                                             
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d','--data', type=str, default='data/MovieGenre.csv'
    )
    
    parser.add_argument(
        '-p', '--parallel', action='store_false'
    )
    args = parser.parse_args()
    
    df = pd.read_csv(args.data)
    image_links, image_names = get_image_meta(df)
    
    if args.parallel:
        download_parallel(image_links, image_names)
    else:
        download_sequential(image_links, image_names)