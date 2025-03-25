import cv2
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from .utils import crop_foreground

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def image_quality_score(image_bytes):
    try:
        if len(image_bytes) < 3 * 1024:
            return True

        image_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            return False

        img = crop_foreground(img)
        img = cv2.resize(img, (256, 256))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pixels = img.reshape(-1, 3)
        sample_pixels = pixels[np.random.choice(pixels.shape[0], 1000, replace=False)]
        kmeans = KMeans(n_clusters=5, n_init=10).fit(sample_pixels)
        dominant_color_percentage = (max(np.bincount(kmeans.labels_)) / len(sample_pixels)) * 100

        if dominant_color_percentage > 93:
            return True

        results = {}
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        results['blur_score'] = laplacian_var

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        is_portrait = len(faces) > 0

        blur_threshold = 30 if is_portrait else 60
        normalized_blur = min(laplacian_var / blur_threshold, 1.0)

        height, width = img.shape[:2]
        results['resolution'] = height * width

        results['brightness'] = np.mean(gray)
        results['contrast'] = np.std(gray)

        center_crop = gray[gray.shape[0]//4: 3*gray.shape[0]//4, gray.shape[1]//4: 3*gray.shape[1]//4]
        dft = np.fft.fft2(center_crop)
        dft_shift = np.fft.fftshift(dft)
        results['motion_blur'] = np.std(20 * np.log(np.abs(dft_shift)))

        edges = cv2.Canny(gray, 100, 200)
        results['edge_density'] = np.sum(edges > 0) / gray.size

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        results['color_std_dev'] = np.mean([np.std(hsv[:, :, i]) for i in range(3)])

        hist, _ = np.histogram(gray.ravel(), bins=256, range=[0, 256])
        results['entropy'] = entropy(hist, base=2)

        weights = {
            'blur_score': 0.1,
            'resolution': 0.1,
            'brightness': 0.1,
            'contrast': 0.1,
            'motion_blur': 0.1,
            'edge_density': 0.25,
            'color_std_dev': 0.15,
            'entropy': 0.1
        }

        final_score = sum(weights[key] * min(results[key] / max_value, 1.0)
                          for key, max_value in [
                              ('blur_score', blur_threshold),
                              ('resolution', 1000000),
                              ('brightness', 255),
                              ('contrast', 255),
                              ('motion_blur', 100),
                              ('edge_density', 0.05),
                              ('color_std_dev', 100),
                              ('entropy', 8.0)
                          ])

        results['final_score'] = final_score

        print(final_score)

        return final_score

    except Exception as e:
        print(f"Error in Quality Checking: {e}")
        return e
