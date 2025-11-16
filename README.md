## License-Plate-Detection-using-OpenCV-and-Haar-Cascade-Classifier
### AIM :
To detect human faces in an image, highlight them with bounding boxes
### PROGRAM :
```
NAME : NITHYA D
REGISTER NUMBER : 212223240110
```
```
import cv2
import matplotlib.pyplot as plt
import os
import urllib.request

image_path = 'image_03.jpg'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Please check the 'image_path' variable.")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
equalized = cv2.equalizeHist(blurred)

plt.imshow(equalized, cmap='gray')
plt.title("Preprocessed Image (Blur + Equalized)")
plt.axis('off')
plt.show()

cascade_path = 'haarcascade_frontalface_default.xml'

if not os.path.exists(cascade_path):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_path)

face_cascade = cv2.CascadeClassifier(cascade_path)

faces = face_cascade.detectMultiScale(
    equalized,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print(f"Total Faces Detected: {len(faces)}")

output = image.copy()
save_dir = "Detected_Faces"
os.makedirs(save_dir, exist_ok=True)

for i, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    face_crop = image[y:y+h, x:x+w]
    save_path = f"{save_dir}/face_{i+1}.jpg"
    cv2.imwrite(save_path, face_crop)

if len(faces) > 0:
    print(f"{len(faces)} face(s) saved in '{save_dir}' folder.")
else:
    print("⚠️ No faces detected. Try adjusting parameters or using a clearer image.")

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Faces")
plt.axis('off')
plt.show()
```

### OUTPUT :
<img width="641" height="461" alt="image" src="https://github.com/user-attachments/assets/62db4e08-2005-4818-afb0-0309b8502966" />
<img width="634" height="459" alt="image" src="https://github.com/user-attachments/assets/d3f3c40d-0c1c-4b49-9b23-d116e74cd98f" />
<img width="630" height="467" alt="image" src="https://github.com/user-attachments/assets/ae1f2b3f-7f8c-4af5-85f2-a791d88f5358" />
<img width="632" height="504" alt="image" src="https://github.com/user-attachments/assets/822c5b2d-3b47-424a-83d6-c032bf2cfe60" />


### RESULT :
Thus the program executed successfully.
