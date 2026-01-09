import os
import cv2
import numpy as np

class FaceImage:
    def __init__(self):
        self.__initialize()
        self.start_get_face()

    def __initialize(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        # train 사진 경로
        # 개인
        self.train_one = os.path.join(current_path, 'image', 'train', 'one')
        self.train_one_save_path = os.path.join(current_path, 'image', 'train', 'one', 'face')
        # 타인
        self.train_other = os.path.join(current_path, 'image', 'train', 'other')
        self.train_other_save_path = os.path.join(current_path, 'image', 'train', 'other', 'face')

        # test 사진 경로
        # 개인
        self.test_one = os.path.join(current_path, 'image', 'test', 'one')
        self.test_one_save_path = os.path.join(current_path, 'image', 'test', 'one', 'face')
        # 타인
        self.test_other = os.path.join(current_path, 'image', 'test', 'other')
        self.test_other_save_path = os.path.join(current_path, 'image', 'test', 'other', 'face')

    #이미지 경로
    #개인
    #train
    def set_train_one_path(self,path):
        self.train_one = path
    #test
    def set_test_one_path(self,path):
        self.test_one = path

    #타인
    #train
    def set_train_other_path(self,path):
        self.train_other = path
    #test
    def set_test_other_path(self,path):
        self.test_other = path

    #다른이름으로 저장
    #개인
    #train
    def set_train_one_save_path(self, path):
        self.train_one_save_path = path
    #test
    def set_test_one_save_path(self, path):
        self.test_one_save_path = path

    #타인
    #trin
    def set_train_other_save_path(self, path):
        self.train_other_save_path = path
    #test
    def set_test_other_save_path(self, path):
        self.test_other_save_path = path


    def start_get_face(self):
        #Train데이터 개인
        self.__classification_face(self.train_one, self.train_one_save_path)
        # Train 데이터 타인
        self.__classification_face(self.train_other, self.train_other_save_path)

        # Test 데이터 개인
        self.__classification_face(self.test_one, self.test_one_save_path)
        # Test 데이터 타인
        self.__classification_face(self.test_other, self.test_other_save_path)


    def __classification_face(self,image_path, save_path):
        for filename in os.listdir(image_path):
            if os.path.exists(os.path.join(save_path,filename)):
                continue
            else:
                if filename.endswith(".png"):
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    img_full_path = os.path.join(image_path, filename)
                    img_array = np.fromfile(img_full_path, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is None:
                        return

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 연산 속도를 위해 흑백으로 전환
                    face = face_cascade.detectMultiScale(gray, scaleFactor=1.3,
                                                         minneighbors=5)  # sclaeFacotr = 얼굴을 얼마나 세밀하게 찾을까?, minNeighbors = 얼굴이라고 확신하려면 몇번이나 겹쳐야 할까?
                    if len(face) > 0:
                        (x, y, w, h) = face[0]
                        margin = 20
                        cropped = img[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin]
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        save_full_path = os.path.join(save_path, filename)
                        extension = os.path.splitext(filename)[1]
                        result, encoded_img = cv2.imencode(extension, cropped)
                        if result:
                            with open(save_full_path, mode='w+b') as f:
                                encoded_img.tofile(f)
                            print("얼굴을 잘 찾았음")
                    else:
                        print("얼굴을 찾지 못했음")
