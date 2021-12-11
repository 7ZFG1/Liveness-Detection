import cv2
import threading
import numpy as np

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class getFeature:
    def __init__(self):
        self.label = 0

    def gui(self, image):
        r, g, b = self.getRGB(image)
        gray_mean = self.getGray(image)
        luminance = self.getLuminance(r,g,b)
        luminanceR = self.getLuminanceR(r,g,b)
        print(gray_mean, luminance, luminanceR)

        #caption = "{} {} {} {} ".format(1, gray_mean, luminance, luminanceR)
        caption = gray_mean, luminance, luminanceR
        caption = np.reshape(caption, (1, -1))
        return caption
 
    def getLuminance(self,R,G,B):
        luminance = (0.299*R) + (0.587*G) + (0.114*B)
        luminance = np.mean(luminance)
        return luminance

    def getLuminanceR(self,R,G,B):
        luminanceR = (0.2126*R)+(0.7152*G)+(0.0722*B)
        luminanceR = np.mean(luminanceR)
        return luminanceR

    def getGray(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_mean = np.mean(img)
        return gray_mean

    def getRGB(self, img):
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        return r, g, b

class classifiers:
    def __init__(self):
        self.data = pd.read_csv('dataset.csv')
        self.label = self.data.iloc[:,-1:].values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data.iloc[:,:-1],self.label,test_size=0.20,random_state=0)

        self.svc = SVC(kernel = 'poly')
        self.nb = GaussianNB()
        self.mlp = MLPClassifier()
        self.rfc = RandomForestClassifier()
        self.dtc = DecisionTreeClassifier()

        print("[INFO]: Loading...")
        self.svc.fit(self.x_train,self.y_train)
        self.nb.fit(self.x_train,self.y_train)
        self.mlp.fit(self.x_train,self.y_train)
        self.rfc.fit(self.x_train,self.y_train)
        self.dtc.fit(self.x_train,self.y_train)


    def Support_VC(self):
        result_svc = self.svc.predict(self.capture) 
        return result_svc

    def GaussiaNb(self):
        result_nb = self.nb.predict(self.capture)
        return result_nb

    def MLP(self):
        result_mlp = self.mlp.predict(self.capture) 
        return result_mlp

    def randomForest(self):
        result_rfc = self.rfc.predict(self.capture)
        return result_rfc

    def desicionTree(self):
        result_dtc = self.dtc.predict(self.capture)
        return result_dtc
        
    def gui(self, capture):
        self.capture = capture
        svm_acc = self.Support_VC()
        gaussianNB_acc = self.GaussiaNb()
        mlp_acc = self.MLP()
        rfc_acc = self.randomForest()
        dtc_acc = self.desicionTree()
        return svm_acc, gaussianNB_acc, mlp_acc, rfc_acc, dtc_acc

class Demo:
    def __init__(self):
        self.mlclassifiers = classifiers()
        self.ext_feature = getFeature()
        self.img = []
        self.face_img=[]
        self.majority_vote = False
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        threading.Thread(target=self.camera).start()

    def gui(self):
        while True:
            self.detect_face()

    def camera(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened:
            _, self.img = cap.read()
            #self.img = cv2.flip(self.img,1)

    def detect_face(self):
        if self.img != []:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                face_img = self.img[y:y+h,x:x+w]
                self.ml_classf(face_img)
                if self.majority_vote==True:
                    cv2.rectangle(self.img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    #self.img = cv2.putText(self.img, 'ALIVE', (x,y-5), 2, cv2.FONT_HERSHEY_SIMPLEX, (0,255,0), 2, cv2.LINE_AA)
                    cv2.putText(self.img, 'ALIVE', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(self.img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    #self.img = cv2.putText(self.img, 'NOT ALIVE', (x,y-5), 2, cv2.FONT_HERSHEY_SIMPLEX, (0,0,255), 2, cv2.LINE_AA)
                    cv2.putText(self.img, 'NOT ALIVE', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)

            cv2.imshow("frame", self.img)
            cv2.waitKey(10)

    def ml_classf(self, face_img):
        capture = self.ext_feature.gui(face_img)
        svm_acc, gaussianNB_acc, mlp_acc, rfc_acc, dtc_acc = self.mlclassifiers.gui(capture)
        print(svm_acc, gaussianNB_acc, mlp_acc, rfc_acc, dtc_acc)
        summ = svm_acc + gaussianNB_acc + mlp_acc + rfc_acc + dtc_acc
        if summ>=3:
            self.majority_vote = True
        else:
            self.majority_vote = False

if __name__ == "__main__":
    demo = Demo()
    demo.gui()