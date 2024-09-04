import cv2

# Загрузка каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier('models/haarcascade/haarcascade_frontalface_default.xml')

# Загрузка изображения
image = cv2.imread('images/example_image.jpg')

# Преобразование изображения в оттенки серого (для лучшей производительности)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Обнаружение лиц на изображении
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Рисование прямоугольников вокруг обнаруженных лиц
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Отображение изображения с обнаруженными лицами
cv2.imshow('Faces Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()