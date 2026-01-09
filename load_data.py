import os
import cv2
import numpy as np

class LoadData:
    #파일 경로에서 이미지 one, other 가져오기
    def load_data(self,one_path, other_path, img_width=128, img_height=128):
        one_x,one_y = self.find_picture(one_path,1)
        other_x,other_y = self.find_picture(other_path, 0)

        X = one_x + other_x
        y = one_y + other_y

        X = np.array(X).reshape(-1, img_height, img_width, 3)
        y = np.array(y)

        return X, y

    def find_picture(self,path, label, img_width=128, img_height=128):
        X = []  # 이미지 저장
        y = []  # 라벨 저장
        if not os.path.exists(path):
            return X,y
        else:
            for filename in os.listdir(path):
                if filename.lower().endswith((".png", ".jpg", '.jpeg')):
                    img_path = os.path.join(path, filename)
                    img_array = np.fromfile(img_path, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.resize(img, (img_width, img_height))
                        X.append(img / 255)
                        y.append(label)
            return X, y
