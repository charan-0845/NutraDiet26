import cv2

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    return image