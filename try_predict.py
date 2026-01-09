from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cv2
import numpy as np
import load_data as ld
import os

#저장된 모델로 예측 해보기
class TryPredict:
    def __init__(self):
        self.__set_korean_fonts()
        self.initailize_path()
        self.default_model = load_model(os.path.join(self.current_path, 'face_model.keras'))
    #public model 변경
    def set_model(self, model_path):
        self.model = load_model(model_path)

    #public test 사진 경로 초기화
    def initailize_path(self):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.test_one_path = os.path.join(self.current_path, 'image', 'test', 'one', 'face')
        self.test_other_path = os.path.join(self.current_path, 'image', 'test', 'other', 'face')


    #public test 사진 경로
    def set_test_one_path(self, path):
        self.test_one_path = path

    def set_test_other_path(self, path):
        self.test_other_path = path

    #예측 시작
    def start_predict(self, model=None):
        cur_model = model if model is not None else self.default_model
        X_test, y_test = ld.LoadData().load_data(self.test_one_path, self.test_other_path)
        for i in range(len(X_test)):
            img = X_test[i]
            one_or_other, score = self.__predict_signature(cur_model,img)
            self.__show_plt(one_or_other, score, img)

    #결과 계산
    def __predict_signature(self, cur_model, img):
        img = np.expand_dims(img, axis=0)  # (1, 128, 128, 3)으로 확장
        prediction = cur_model.predict(img, verbose=0)[0][0]  # 0~1 사이 값 하나 추출
        if prediction >= 0.8:
            predicted_label = 1  # 개인
            confidence = prediction
        else:
            predicted_label = 0  # 타인
            confidence = 1 - prediction  # 타인일 확률로 역산
        return predicted_label, confidence

    #결과 창
    def __show_plt(self,one_or_other,score,img):
        # 4. 결과 판별 로직
        if one_or_other == 1:
            result_text = f"Result: 개인"
            color = 'green'  # 맞으면 초록색 글씨
        else:
            result_text = f"Result: 타인"
            color = 'red'  # 아니면 빨간색 글씨
        # 5. plt로 이미지와 결과 띄우기
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor((img*255).astype(np.uint8),cv2.COLOR_BGR2RGB))
        # 제목에 결과와 확률 표시
        plt.title(f"{result_text}\nConfidence: {score*100:.2f}%",
                  fontsize=15, color=color, pad=20)
        plt.axis('off')  # 축 숨기기
        plt.show()

    #matplotlib 한글 폰트 적용하기
    def __set_korean_fonts(self):
        # 한글 폰트 설정
        # Windows에서 사용 가능한 한글 폰트 목록
        korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic', 'Gulim']
        # 시스템에 설치된 폰트 목록 가져오기
        font_list = [f.name for f in fm.fontManager.ttflist]
        # 사용 가능한 한글 폰트 찾기
        found_font = None
        for font_name in korean_fonts:
            # 폰트 목록에서 한글 폰트가 있는지 확인
            for system_font in font_list:
                if font_name in system_font:
                    found_font = font_name
                    break
            # 폰트를 찾았으면 더 이상 찾지 않음
            if found_font is not None:
                break
        # 한글 폰트 설정
        if found_font is not None:
            plt.rcParams['font.family'] = found_font
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        else:
            print("경고: 한글 폰트를 찾을 수 없습니다. 한글이 깨질 수 있습니다.")