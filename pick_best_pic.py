import cv2
import os
import dlib
from scipy.spatial import distance
import numpy as np

# Load images from folder
def load_images_from_folder(folder, ims):
    images = []
    for filename in ims:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((filename, img))
    return images

def eye_aspect_ratio(eye):
    A = distance.euclidean((eye[1].x, eye[1].y), (eye[5].x, eye[5].y))
    B = distance.euclidean((eye[2].x, eye[2].y), (eye[4].x, eye[4].y))
    C = distance.euclidean((eye[0].x, eye[0].y), (eye[3].x, eye[3].y))
    ear = (A + B) / (2.0 * C)
    return ear

def weighted_mean(scores, weights):
    return np.sum(np.array(scores) * np.array(weights)) / np.sum(weights)

def eyes_open_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return 0  # No face detected
    scores = []
    weights = []
    for face in faces:
        shape = predictor(gray, face)
        leftEye = [shape.part(i) for i in range(36, 42)]
        rightEye = [shape.part(i) for i in range(42, 48)]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        scores.append(ear)
        weights.append(face.width() * face.height())
    return weighted_mean(scores, weights)

# Smile detection
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def smile_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return 0  # No face detected
    scores = []
    weights = []
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        roi_gray = gray[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        if len(smiles) > 0:
            scores.append(1)
        else:
            scores.append(0)
        weights.append(face.width() * face.height())
    return weighted_mean(scores, weights)

# Blur detection
def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

# Lighting evaluation
def lighting_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean() / 255.0
    contrast = gray.std() / 255.0
    return mean_brightness, contrast

# Normalize scores
def normalize_scores(e_score, s_score, b_score, l_score):
    # Normalization ranges
    e_score = e_score / 0.4  # Normalize EAR (0 to 0.4) to 0 to 1
    b_score = min(b_score / 500.0, 1.0)  # Normalize blur (0 to 300) to 0 to 1
    brightness, contrast = l_score
    return e_score, s_score, b_score, brightness, contrast

# Composite score calculation
def composite_score(eyes_open, smile, blur, brightness, contrast):
    score = (eyes_open * 0.4 + smile * 0.3 + blur * 0.2 + brightness * 0.05 + contrast * 0.05)
    return score

# Eye Aspect Ratio calculation
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

if __name__ == '__main__':

    # Load images
    folder = 'album_images'
    ims = ['76.jpg', '75.jpg']
    images = load_images_from_folder(folder, ims)

    # Initialize lists to hold scores
    eyes_open_scores = []
    smile_scores = []
    blur_scores = []
    lighting_scores = []
    composite_scores = []

    # Calculate scores
    for filename, img in images:
        e_score = eyes_open_score(img)
        s_score = smile_score(img)
        b_score = blur_score(img)
        l_score = lighting_score(img)
        e_score, s_score, b_score, brightness, contrast = normalize_scores(e_score, s_score, b_score, l_score)
        c_score = composite_score(e_score, s_score, b_score, brightness, contrast)
        
        eyes_open_scores.append(e_score)
        smile_scores.append(s_score)
        blur_scores.append(b_score)
        lighting_scores.append((brightness, contrast))
        composite_scores.append(c_score)
        
        print(f"Image: {filename}")
        print(f"  Eyes Open Score (normalized): {e_score}")
        print(f"  Smile Score: {s_score}")
        print(f"  Blur Score (normalized): {b_score}")
        print(f"  Lighting Score (Brightness, Contrast): {brightness, contrast}")
        print(f"  Composite Score: {c_score}")
        print("")

    # Select the best image
    best_image_index = np.argmax(composite_scores)
    best_image_filename = ims[best_image_index]
    best_image = images[best_image_index][1]

    print(f"The best image is {best_image_filename} with a composite score of {composite_scores[best_image_index]}")

    # Save or display the best image
    cv2.imwrite('best_image.jpg', best_image)
    cv2.imshow('Best Image', best_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
