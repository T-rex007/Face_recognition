class ImageManager:
    ### Class Constructor
    def __init__(self, imgpaths, feature_extractor, face_detector):
        self.imgpaths = imgpaths
        self.feature_extractor = feature_extractor
        self.face_detector = face_detector
        self.bb_lst = []
        self.im_sz = feature_extractor.input_shape
        self.img = None
        
    def detect_faces(self, path):
        img = plt.imread(path)
        results = self.face_detector.detect_faces(img)
        for i in results:
            self.bb_lst.append(i["box"])
    
    def resize(self):
        tmp1 = tf.image.resize(self.img, (self.im_sz[0], self.im_sz[1]))
        return tmp1.numpy().reshape((im_sz[0], im_sz[1], tmp1.shape[-1]))