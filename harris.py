from os.path import join
import numpy as np
import skimage.color
from PIL import Image
from opts import get_opts
import matplotlib.pyplot as plt
import visual_words
from sklearn.cluster import KMeans

## function to compute alpha harris corners
def harris_corners(alpha, img):
    
    W = img.shape[1]
    H = img.shape[0]
    img = skimage.color.rgb2gray(img)
    
    
    Iy, Ix = np.gradient(img)
    
    IyIy  = Iy**2
    IxIx = Ix**2
    IxIy = Iy*Ix
    
    corners = []
    corner_scores = []
    window = 5
    

    for x in range(window//2, W-window//2):
        for y in range(window//2,  H - window//2):
   
            # sum over the window
            sigma_IxIx = IxIx[y - window//2: y +window//2+1, x - window//2: x+window//2+1].sum()
            sigma_IyIy = IyIy[y - window//2: y +window//2+1, x - window//2: x+window//2+1].sum()
            sigma_IxIy  = IxIy[y - window//2: y +window//2+1, x - window//2: x+window//2+1].sum()
            
            detM = (sigma_IxIx * sigma_IyIy) - (sigma_IxIy**2)
            traceM =  sigma_IxIx + sigma_IyIy
            
            # 0.05 was as the average between the values of 0.04-0.06 provided by wikipedia
            R = 1*(detM - 0.05*(traceM*traceM))
            
            corners.append([y,x])
            corner_scores.append(R)
    
    sorted_indices = np.argsort(corner_scores)
    sorted_indices = sorted_indices[-alpha:]

    sorted_corners = [corners[i] for i in sorted_indices]
    sorted_scores = [corner_scores[i] for i in sorted_indices]  
    
    return sorted_corners




def compute_dictionary_harris(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    alpha = opts.alpha
    filter_scales = opts.filter_scales
    K = opts.K
    
    F = len(filter_scales)*4
    
  
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    filter_responses = np.empty((0,3*F), float)
    
    print("finding corners")
    i = 1
    for file in train_files:
        print(i)
        img_path =  join(opts.data_dir,file)
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)/255
        responses = visual_words.extract_filter_responses(opts,img)
        
        corners = harris_corners(alpha, img)
        
        i+=1
        
        
        
        reduced_responses = reduced_harris(corners,responses,F)
        
        filter_responses = np.vstack((filter_responses,reduced_responses))
        
    print(filter_responses.shape)
    kmeans = KMeans(n_clusters=K,n_jobs = n_worker).fit(filter_responses) 
    dictionary = kmeans.cluster_centers_
    print("done kmeans")
    np.save(join(out_dir, 'dictionary.npy'), dictionary)




def reduced_harris(corners, responses, F):

    
    result = np.empty((0,3*F), float)
    
    for i in range(len(corners)):
        current_row = np.empty((1,0),float)
        for j in range(3*F):
            point= responses[corners[i][0],corners[i][1],j]
            current_row = np.append(current_row,point)
            
        
        result =  np.vstack((result,current_row))
        
    return result
   

    


