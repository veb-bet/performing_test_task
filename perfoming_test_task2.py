import torch
import requests
import numpy as np
import cv2

# загружаем модель YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# задаем размер видео
frame_width = 1240
frame_height = 680

# определяем искомый класс объектов
class_name = "cat"

# определяем имя выходного файла
output_file_name1 = f"{class_name}_output.avi"
# определяем имя выходного файла
output_file_name2 = f"{class_name}_output.mp4"
# задаем частоту кадров видео
fps = 30

# задаем формат записи видео
fourcc1 = cv2.VideoWriter_fourcc('M','J','P','G')
# задаем формат записи видео
fourcc2 = cv2.VideoWriter_fourcc('H','2','6','4')
# читаем исходное видео
cap = cv2.VideoCapture(r'C:\Users\user\Desktop\pythonProject1\videos\1.mp4')

# проверяем, удалось ли открыть видео
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# создаем объект для записи выходного видео
out1 = cv2.VideoWriter(output_file_name1, fourcc1, fps, (frame_width, frame_height))
# создаем объект для записи выходного видео
out2 = cv2.VideoWriter(output_file_name2, fourcc2, fps, (frame_width, frame_height))

# проходим по всем кадрам видео
while(cap.isOpened()):
    # захватываем кадр
    ret, frame = cap.read()

    if ret == True:
        # применяем модель для обнаружения объектов на кадре
        results = model(frame, size=640)
        detections = results.pred

        # проходим по найденным объектам
        for i, det in enumerate(detections):
            if det is not None:
                for j in range(det.shape[0]):
                    # определяем класс текущего объекта
                    class_index = int(det[j, 5])
                    class_conf = det[j, 4] * det[j, 5]

                    # если класс объекта соответствует искомому классу, сохраняем его кадр
                    if model.names[class_index] == class_name:
                        x1, y1, x2, y2 = det[j, :4].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        object_frame = frame[y1:y2, x1:x2, :]
                        out1.write(cv2.resize(object_frame, (frame_width, frame_height)))
                        out2.write(cv2.resize(object_frame, (frame_width, frame_height)))

        # выводим прогресс выполнения
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processed {current_frame}/{total_frames} frames")

    # если кадр не удалось захватить, заканчиваем цикл
    else:
        break

# освобождаем ресурсы
cap.release()
out1.release()
out2.release()
cv2.destroyAllWindows()

print("Output video saved to", output_file_name1, "and", output_file_name2)