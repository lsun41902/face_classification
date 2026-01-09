import os
import get_face, try_predict, try_study

if __name__ == "__main__":
    # 준비한 사진의 얼굴만 자르기
    face = get_face
    face.FaceImage().start_get_face()

    #현재 파일의 경로
    current = os.path.dirname(os.path.abspath(__file__))
    #현재 파일경로의 모델
    cur_model = os.path.join(current,"face_model.keras")

    #모델의 유무 체크
    if os.path.exists(cur_model):
        #모델이 있으면 예측 시작
        #predict 클래스 호출
        predict = try_predict.TryPredict()
        predict.start_predict()
    else:
        #모델이 없으면 학습 시작
        #study 클래스 호출
        study = try_study.Study()
        study.start_study()


