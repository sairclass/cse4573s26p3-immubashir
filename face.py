'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####

    # Converting input tensor to HWC uint8 format for compatibility with face_recognition.
    img_hwc = _to_hwc_uint8(img)

    # Trying both RGB and BGR channel orders to improve detection robustness.
    candidates = [img_hwc, _flip_channels_hwc(img_hwc)]

    best_locations = []
    best_score = -1.0

    # Selecting the best detection result based on number of faces and total bounding box area.
    for candidate in candidates:
        locations = _detect_face_locations(candidate)

        score = float(len(locations)) * 1e12 + _total_box_area(locations)
        if score > best_score:
            best_score = score
            best_locations = locations
    
    # Converting the best detected face locations to [x, y, w,, h] format within image bounds.
    H, W = img_hwc.shape[0], img_hwc.shape[1]
    detection_results = _locations_to_xywh(best_locations, H, W)

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####

    img_names = sorted(list(imgs.keys()))
    features = []

    # Extracting feature embeddings for each image.
    for img_name in img_names:
        feature = _extract_face_embedding(imgs[img_name])
        features.append(feature)

    # Stacking all embeddings into a feature matrix.
    X = torch.stack(features, dim = 0)

    # Normalizing embeddings to unit length for stable clustering.
    X = _l2_normalize(X)

    # Applying K-means clustering to group similar faces.
    labels = _kmeans_torch(X, K)

    # Assigning images to clusters based on predicted labels.
    for img_name, label in zip(img_names, labels.tolist()):
        cluster_results[int(label)].append(img_name)

    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)