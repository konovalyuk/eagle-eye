import imageio
import numpy as np
import matplotlib.pyplot as plt

# # Создание примера массива данных изображения с типом float
image_data_float = np.random.rand(100, 100) * 255  # Пример случайных значений от 0 до 255

# # Преобразование массива данных изображения к типу uint8
image_data_uint8 = image_data_float.astype(np.uint8)

# # Сохранение изображения
imageio.imwrite('output_image.png', image_data_uint8)

plt.imshow(image_data_uint8)
plt.show()
# ----------------------
ascent_array = imageio.imread_v2('output_image.png')
type(ascent_array)
print(ascent_array.shape)
print(ascent_array.dtype)

#------------------------

ascent_array.tofile('ascent.raw')
ascent_raw = np.fromfile('ascent.raw', dtype=np.uint8)
# # ascent_raw.shape(262144,'ascent.raw')
plt.imshow(image_data_uint8, cmap=plt.cm.jet)
plt.show()
plt.imshow(image_data_uint8, cmap=plt.cm.grey, vmin=30, vmax=30)
plt.show()
plt.axis('off')
plt.show()
plt.contour(image_data_uint8)
plt.show()