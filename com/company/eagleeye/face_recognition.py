import cv2
import face_recognition

# Загрузка изображений известных людей и извлечение их эмбеддингов
known_face_encodings = []
known_face_names = []
known_personal_info = []


# Пример добавления известных лиц
def load_known_face(image_path, name, personal_info):
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    known_personal_info.append(personal_info)


load_known_face("images/person1.jpg", "John Doe", "Age: 30, Occupation: Engineer")
load_known_face("images/person2.jpg", "Jane Smith", "Age: 25, Occupation: Data Scientist")
load_known_face("images/person3.jpg", "Brad Pitt", "Age: 45, Occupation: Actor")

# Загрузка изображения для распознавания лиц
image_to_recognize = cv2.imread('images/example_image.jpg')

# Преобразование изображения в RGB (face_recognition работает с RGB)
rgb_image = cv2.cvtColor(image_to_recognize, cv2.COLOR_BGR2RGB)

# Обнаружение лиц на изображении и извлечение их эмбеддингов
face_locations = face_recognition.face_locations(rgb_image)
face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

# Распознавание лиц
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"
    personal_info = "No additional information"

    # Если найдено совпадение, берем имя и информацию из базы
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
        personal_info = known_personal_info[first_match_index]

    # Рисование прямоугольника вокруг лица
    cv2.rectangle(image_to_recognize, (left, top), (right, bottom), (0, 255, 0), 2)

    # Вывод имени под лицом
    cv2.rectangle(image_to_recognize, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
    cv2.putText(image_to_recognize, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # Вывод дополнительной информации о человеке
    print(f"Person recognized: {name}")
    print(f"Info: {personal_info}")

# Отображение изображения с распознанными лицами
cv2.imshow('Faces Recognized', image_to_recognize)
cv2.waitKey(0)
cv2.destroyAllWindows()
