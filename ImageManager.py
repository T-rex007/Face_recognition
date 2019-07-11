"""
Author: Tyrel Cadogan
Email: shaqc777@yahoo.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from matplotlib import patches, patheffects
from tensorflow.keras.models import model_from_json, load_model
from mtcnn.mtcnn import MTCNN
from pathlib import Path as path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import pdb ### Python debuger
import os ### Navigate Through Dirrectory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import cv2

class ImageManager:
    ### Class Constructor
    def __init__(self,  imgpaths ,feature_extractor = None, face_detector = None, 
                 img_rtpath = None, feat_rtpath = None):
        
        self.feature_extractor = feature_extractor
        self.face_detector = face_detector
        self.bb_lst = []
        self.im_sz = feature_extractor.input_shape
        self.img = None
        self.imgpaths = [str(i[0]) for i in imgpaths.values ]
        self.img_rtpath = img_rtpath
        self.feat_rtpaths = feat_rtpath
        idty = []
        for i in self.imgpaths:
            idty.append(i.split("/"))
        df = pd.DataFrame(idty)
        self.c = df[0].values ### The class of each image
        self.classes = list(df[0].unique()) ### different class in dset
        self.min_class_count = min(df[0].value_counts())
        self.classmap = {}### class to index mapping
        for i in self.classes:
            self.classmap[i] = np.where(self.c == i)
        self.sample_feat1 = []
        self.sample_feat2 = []
        self.feat1 = None
        self.feat2 = None
        self.sample_labels = []
        ### Face Detector performance
        self.undetected_faces = []
        self.face_detector_error = None
        
        
    def detect_faces(self, img):
        results = self.face_detector.detect_faces(img)
        for i in results:
            self.bb_lst.append(i["box"])
        return bb_lst
    
    def extract_feature(self, img):
        image_features = []
        for i in range(len(bb_lst)):
            img = crop_face(img, bb_lst[i])
            img = resize(img,(self.im_sz[1],im_sz[2]))
            img = normalize(img).reshape((1, im_sz[1],im_sz[2], im_sz[3]))
            image_features.append(feature_extractor.predict(img))
        return image_features
    def extract_all_faetures(self): ###To be tested
        for i in self.imgpaths:
            if(os.path.isfile(self.feature_rt_path+"/"+i+".npy") ==True):
                continue
            else:
                img = plt.imread(self.imgrt_path+"/" + i)
                bb_lst = self.detect_faces(img)
                for i1 in range(len(bb_lst)):
                    feature = self.extract_feature(model, img, bb_lst[i1])
                    ###check to see if dirrectory e
                    if os.path.isdir(self.feature_rt_path+"/"+i[:7]):
                        np.save(self.feature_rt_path+"/"+i+".npy", feature)
                    else:
                        os.mkdir(self.feature_rt_path+"/"+i[:7])
                        np.save( self.feature_rt_path+"/"+i+".npy", feature)


    def normalize(self, img):
        """
        Normalize the image pixels values between  1 and 0
        """
        return img/255
    
    def crop_face(self,img):
        """
        Crop the area delimited by the the bounding box bb
        """
        im_lst = []
        for i in range(len(bb)):
            x1, y1, width, height = bb[i]
            x1,y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            im_lst.append(img[y1:y2, x1:x2])
        return im_lst
    
    def random_sample(self,label, classidx, k):
        """
        Returns k random indexes of of a particalar class or all other classes 
        than that is indicated by classidx
        Args:
            label: Indicates whether to sample indexes form th indicate class or not.
            classidx: which class to sample
            k: Sample size 
        """
        if (label == 1):
            return random.choices(self.classmap[classes[classidx]][0], k = k)
        elif(label == 0):
            tmp = [list(self.classmap[i][0].reshape(-1)) 
                   for i in self.classes if(i != self.classes[classidx])]
            tmp = list(itertools.chain.from_iterable(tmp))
            return random.choices(tmp, k = k)
        print("OOPS it broke")
        return None
            
    def update_samples(self,sample, label):
        """
        Update sample values(Mutator funtion)
        Args:
            Sample: Tuple containing the feature pairs in form of a list
                    eg.(feature_idx_list1, feature_idx_list2)
            label: Indicates the label of the input feature pair are 
        """
        tmp = [label] * len(sample[0])
        
        self.sample_labels = self.sample_labels + tmp
        self.sample_feat1 = self.sample_feat1 + sample[0]
        self.sample_feat2 = self.sample_feat2 + sample[1]
        
    def balance_random_sample(self, k):
        ### Positive sampling
        for clss_idx in range(len(self.classes)):
            self.update_samples((self.random_sample(1, clss_idx, k),
                                 self.random_sample(1, clss_idx, k)),1)
            self.update_samples((self.random_sample(0, clss_idx, k),
                                 self.random_sample(0, clss_idx, k)),0)
        
    def load_features(self):
        """
        loads in extracted features and stores it to the 
        feat1 and feat2 attribute of the object
        """
        ### Obtaining feature1
        img_sam = np.array(self.imgpaths)[self.sample_feat1]
        img_sam = list(img_sam)
        if(os.path.isfile("test/image_features/"+ img_sam[0][:-4]+".jpg.npy") == True):
            imf = np.load("test/image_features/"+ img_sam[0][:-4]+".jpg.npy")###
        else:
            self.undetected_faces.append("test/image_features/"+ img_sam[0][:-4]+".jpg.npy")
        for i in img_sam:
            if (os.path.isfile("test/image_features/"+ i[:-4]+".jpg.npy") == True):
                imf1 = np.load("test/image_features/"+ i[:-4]+".jpg.npy")###
                imf = np.vstack([imf, imf1])
            else:
                self.undetected_faces.append("test/image_features/"+ i[:-4]+".jpg.npy")
        self.feat1 = imf
        
        ### Loading feat2
        img_sam = np.array(self.imgpaths)[self.sample_feat2]
        img_sam = list(img_sam)
        if(os.path.isfile("test/image_features/"+ img_sam[0][:-4]+".jpg.npy") == True):
            imf = np.load("test/image_features/"+ img_sam[0][:-4]+".jpg.npy")###
        else:
            self.undetected_faces.append("test/image_features/"+ img_sam[0][:-4]+".jpg.npy")
        for i in img_sam:
            if (os.path.isfile("test/image_features/"+ i[:-4]+".jpg.npy") == True):
                imf1 = np.load("test/image_features/"+ i[:-4]+".jpg.npy")###
                imf = np.vstack([imf, imf1])
            else:
                self.undetected_faces.append("test/image_features/"+ i[:-4]+".jpg.npy")
        self.feat2 = imf
    
    def get_features(self):
        return (self.feat1, self.feat2, self.sample_labels)
    
    def compute_face_detector_error(self):
        for i in self.imgpaths:
            if (os.path.isfile("test/image_features/"+ i[:-4]+".jpg.npy")):
                self.detected_faces.append(i)
        