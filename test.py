from joblib import load 
from VisionUtils import *
from Manager import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from imgaug import augmenters as iaa
#import imgaug
def transform(f1, f2, scaler,selector):
    cos_d = np.array(feat_distance_cosine_scalar(f1, f2.T))
    cos_d = cos_d.reshape(-1,1)
    sqr_diff = np.power(np.abs(f1- f2), 2)
    rat = f1/f2
    data = np.hstack([cos_d, sqr_diff, rat])
    data_ = scaled_data = scaler.transform(data)
    #print(scaled_data.shape)
    data_ = selector.transform(scaled_data)
    #print(data_.shape)
    return data_

def extract_feature( feature_extractor, img, bb):
    """
    Extract features given img, bounding box and feature extractor
    """
    insz = feature_extractor.input_shape
    img = crop_face(img, bb)
    img = cv2.resize(img, (160,160), interpolation = cv2.INTER_AREA)
    img = normalize(img).reshape((1, insz[1],insz[2], insz[3]))
    image_feature = feature_extractor.predict(img)
    return image_feature

def main():
    m = MTCNN()
    #tf.enable_eager_execution()
    feature_extractor = load_model("Models/FaceNet/facenet_keras.h5")
    selector = load("demo/PipelineParts/feature_selector.joblib")
    scaler = load("demo/PipelineParts/scaler.joblib")
    gboost = load("demo/PipelineParts/GboostModel.joblib")

    ### Extract Encoding from Faces in database
    DATABASE_PATH = "demo/database/pics/"
    img_list = os.listdir(DATABASE_PATH)
    impath = DATABASE_PATH+ img_list[0]

    feat_dict = {}
    for path in img_list:
        feat_dict.update({path[:-4]: get_feat(m, feature_extractor, DATABASE_PATH+path) })

    print("Current Person in Data Base")
    print(feat_dict.keys())

    #seq = iaa.Sequential([iaa.SigmoidContrast()])
    # VID_PATH = "demo/database/face_detection.avi"

    ### Run Face Verification Algorithm
    VID_PATH = "demo/database/face_detection.avi"
    cap = cv2.VideoCapture(0)
    m = MTCNN()
    font = cv2.FONT_HERSHEY_SIMPLEX
    while(True):
        ### Capture Videos
        ret, frame = cap.read()
        #frame = cv2.resize(frame, (600, 600))
        #frame = increase_brightness(frame)
        #frame = im_aug = seq.augment_images(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #cv2.imshow('vodeos', img)
        bb_lst= detect_faces(frame, m)
        if (len(bb_lst)==0):
            cv2.putText(frame,"No Faces Detected",(50,50), font, 2,(0,0,0),2,cv2.LINE_AA)
        for b in bb_lst:
            ### Draw Rectangle
            frame = cv2.rectangle( frame,(b[0],b[1]),(b[0] + b[2],b[1]+ b[3]),(0,255,0),3)
            ### Extract features from frame
            f2 = extract_feature(feature_extractor, frame, b)
            ###Check to see if detected face is in the data base 
            ### and label each bb in each frame
            idp = []
            p = []
            u = []
            names = []
            probs = []
            for k, v in feat_dict.items():
                pred = gboost.predict_proba(transform(v,f2, scaler, selector ))
                _pred = gboost.predict(transform(v,f2, scaler, selector))
                print('pred_proba: ', pred)
                print('pred: ', _pred)
                if _pred[0] == 1:
                    names.append(k)
                    probs.append(pred[0][1])
                    print('names: ', names)
                    print('pred_proba: ', pred)
                    print('pred: ', _pred)   
                idp.append(k)
                p.append(pred[0][1])
                u.append(pred[0][0])
            name = "Unk" + str(np.max(u))
            if len(names) > 0 and probs[np.argmax(probs)] > 0.98:
                name = names[np.argmax(probs)] +" >0.98" 
            cv2.putText(frame,name,(b[0],b[1]), font, 2,(0,0,0),2,cv2.LINE_AA)
        ### display video captured
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('videos', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if(__name__ == "__main__"):
    main()