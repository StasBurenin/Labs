# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# импорт необходимых модулей
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# создаем и инициализируем аргументы
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# загружаем нашу модель
print("[INFO] загрузка модели...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# запускаем видеопоток
print("[INFO] запуск видеопотока...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# цикл по кадрам для воспроизведения видео
while True:
    # захватываем кадр из видеопотока и изменяем его размер
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # захватваем размеры кадра и преобразовываем его в blob(числовой массив)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    data = open('data.txt', 'w')
    data.write(str(blob))

    # передаем blob в нейросеть и получаем обнаружение и значения прогноза
    net.setInput(blob)
    detections = net.forward()

    # цикл через значения прогноза
    for i in range(0, detections.shape[2]):
        # извлечение уверенности (то есть вероятности), связанной с прогнозом
        confidence = detections[0, 0, i, 2]

        # отфильтровываем слабые обнаружения, гарантируя, что полученная уверенность
        # больше минимальной уверенности
        if confidence < args["confidence"]:
            continue

        # вычисляем (x, y) - координаты ограничивающей рамки для объекта
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # рисуем рамку вокруг найденного лица
        text = "{:.2f}%".format(confidence * 100)
        print(confidence)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        d
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # вывод окна для видеопотока
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # по нажатию клавиши q выходим из видеопотока
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

