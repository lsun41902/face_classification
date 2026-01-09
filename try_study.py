from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import load_data as ld
import try_predict
import os

class Study:
    #생성자
    def __init__(self):
        self.set_path_initialize()
        self.setTrainTest()

    #public train, test다시 설정하기
    def setTrainTest(self):
        self.X_train, self.y_train = ld.LoadData().load_data(self.train_one_path, self.train_other_path)
        self.X_test, self.y_test = ld.LoadData().load_data(self.test_one_path, self.test_other_path)

    # public X_train 다시 설정하기
    def setXTrain(self, X_train):
        self.X_train = X_train

    # public y_train 다시 설정하기
    def setYTrain(self, y_train):
        self.y_train = y_train

    # public X_test 다시 설정하기
    def setXTest(self, X_test):
        self.X_test = X_test

    # public y_test 다시 설정하기
    def setYTest(self, y_test):
        self.y_test = y_test

    # public path초기화
    def set_path_initialize(self):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.train_one_path = os.path.join(self.current_path,'image','train','one','face')
        self.train_other_path = os.path.join(self.current_path,'image','train','other','face')
        self.test_one_path = os.path.join(self.current_path,'image','test','one','face')
        self.test_other_path = os.path.join(self.current_path,'image','test','other','face')

    #public path 설정
    def set_train_one_path(self, path):
        self.train_one_path = path
    def set_train_other_path(self, path):
        self.train_other_path = path
    def set_test_one_path(self, path):
        self.test_one_path = path
    def set_test_other_path(self, path):
        self.test_other_path = path

    def start_study(self):
        # 이미지 편집
        # 이미지는 하드에 물리적으로 저장하지 않고 RAM에서 작업후 소거됨
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        X_test = self.X_test.copy()
        y_test = self.y_test.copy()

        datagen = ImageDataGenerator(
            rotation_range=10,  # 더 많이 회전
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        more_imgs = datagen.flow(X_train, y_train, batch_size=8, shuffle=True)

        model = Sequential()
        # 1번째 기본 특징 분류(얼굴형)
        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)))
        model.add(MaxPooling2D((2, 2)))  # 최대한 정밀하게 저장하기 위해서

        # 2번째 대략적인 특징 분류 (눈,코,입)
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

        # 3번째 세세한 특징 분류(주름,점 위치 등등)
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

        # 분류기
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=Adam(learning_rate=0.0005),  # 1. 공부 방법 (가장 똑똑한 학생)
            loss='binary_crossentropy',  # 2. 채점 기준 (0 또는 1 판독 전용)
            metrics=['accuracy']  # 3. 성적표 (정확도 표시)
        )

        model.fit(
            more_imgs,
            steps_per_epoch=(len(X_train) // 8) * 2,
            epochs=100,
            validation_data=(X_test, y_test),
            verbose=1
        )
        dnn_loss, dnn_acc = model.evaluate(X_test, y_test, verbose=0)

        print(f"CNN 정확도:{dnn_acc * 100:.2f}% 입니다.")
        if  0.7 <= dnn_acc <= 0.88:
            save_path = os.path.join(self.current_path,'face_model.keras')
            model.save(save_path)
            print("모델을 저장 했습니다.")
            print("모델 검증을 시작합니다.")
            try_predict.TryPredict().start_predict(model)
        else:
            print("성능이 낮아 저장하지 않았습니다. 다시 수행합니다.")
            # self.start_study()

