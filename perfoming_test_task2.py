# стиль кода PEP8
# код работает с видео файлом, предназначен для обнаружения объектов на кадрах с заданным классом, сохранения их кадров в новый файл в форматах .avi, .mp4.
# для создания виртуального окружения в ОС Windows с использованием Anaconda: conda create -n myvenv python=3.8
# для активации виртуального окружения в ОС Windows с использованием Anaconda: conda activate myvenv
# для установки всех пакетов из `requirements.txt` необходимо выполнить команду: pip install -r requirements.txt

# импортируем необходимые библиотеки
import torch
import requests
import numpy as np
import cv2

# загружаем модель YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# читаем  путь к исходному видео
path_to_video = input("Enter path to input video: ")
cap = cv2.VideoCapture(path_to_video)

# задаем искомый класс объектов
class_name = input("Enter target object class name: ")

# проверяем, удалось ли открыть видео
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# получаем размеры исходного видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# определяем имя выходного файла
output_file_name1 = f"{class_name}_output.avi"
# определяем имя выходного файла
output_file_name2 = f"{class_name}_output.mp4"
# задаем частоту кадров видео
fps = 30

# задаем формат записи видео avi
fourcc1 = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# задаем формат записи видео mp4
fourcc2 = cv2.VideoWriter_fourcc('H', '2', '6', '4')

# создаем черное изображение для вставки объектов
black_image = np.zeros((frame_height, frame_width, 3), dtype="uint8")

# создаем объекты для записи выходного видео avi
out1 = cv2.VideoWriter(output_file_name1, fourcc1, fps, (frame_width, frame_height))
# создаем объекты для записи выходного видео mp4
out2 = cv2.VideoWriter(output_file_name2, fourcc2, fps, (frame_width, frame_height))

# проходим по всем кадрам видео
while (cap.isOpened()):
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

                        # создаем новое изображение размером с исходное видео и вставляем в него найденный объект в исходном размере
                        object_frame = black_image.copy()
                        cv2.rectangle(object_frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
                        object_frame = cv2.bitwise_and(object_frame, frame)

                        # записываем кадр в выходное видео
                        out1.write(object_frame)
                        out2.write(object_frame)

        # вывод прогресса выполнения работы алгоритма обработки видео
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = "=" * int(30 * current_frame / total_frames) + "-" * int(30 * (1 - current_frame / total_frames))
        percent_complete = current_frame / total_frames * 100
        print(f"\rProgress: [{progress}] {percent_complete:.2f}%", end="")

        # если кадр не удалось захватить, заканчиваем цикл
    else:
        break

# освобождаем ресурсы
cap.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
# вывод сообщения о завершении работы алгоритма и имена сохраненных видео файлов
print("\nOutput video saved to", output_file_name1, "and", output_file_name2)
