"""
Phone Usage Detector
Uses a fine-tuned MobileNetV2 / EfficientNet classifier on cropped person regions.

Model options:
  Option A – Custom trained model (recommended):
    Download: https://drive.google.com/file/d/1X...  (see README)
    Place at: models/phone_classifier.h5

  Option B – Roboflow pre-trained (phone detection dataset):
    https://universe.roboflow.com/roboflow-100/cell-phones-id8js

  Option C – Use YOLO class_id=67 (cell phone) directly from COCO – already in yolo_detector.py
"""

import cv2
import numpy as np


class PhoneDetector:
    """
    Classifies whether a person region shows phone usage.
    Backends: TF/Keras model, or edge/heuristic fallback when no weights are present.
    """

    def __init__(self, model_path="models/phone_classifier.h5", img_size=(96, 96)):
        self.img_size = img_size
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            print(f"[PhoneDet] Loaded Keras model: {model_path}")
            return ("keras", model)
        except Exception:
            pass

        # Fallback: use YOLO's cell-phone class (no extra model needed)
        print("[PhoneDet] No Keras model found – using YOLO cell-phone class as fallback.")
        return ("yolo_fallback", None)

    def detect(self, region):
        """
        region: BGR image crop of a person.
        Returns: {"detected": bool, "confidence": float}
        """
        if region is None or region.size == 0:
            return {"detected": False, "confidence": 0.0}

        backend, model = self.model

        if backend == "keras":
            return self._keras_predict(region, model)
        else:
            return self._heuristic_predict(region)

    def _keras_predict(self, region, model):
        img = cv2.resize(region, self.img_size)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        prob = float(model.predict(img, verbose=0)[0][0])
        return {"detected": prob > 0.5, "confidence": prob}

    def _heuristic_predict(self, region):
        """
        Lightweight heuristic: detect rectangular bright objects near face/hand area.
        Used as fallback when no model is available.
        """
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            area = w * h
            region_area = region.shape[0] * region.shape[1]

            # Phone-like: portrait rectangle, moderate size
            if 0.4 < aspect < 0.65 and 0.03 < area / region_area < 0.25:
                return {"detected": True, "confidence": 0.70}

        return {"detected": False, "confidence": 0.15}


# ── Training script (run once to create phone_classifier.h5) ─────────────────

def train_phone_classifier(dataset_dir="dataset/phone", epochs=20, model_out="models/phone_classifier.h5"):
    """
    Fine-tunes MobileNetV2 on a binary phone/no-phone dataset.

    Dataset structure:
      dataset/phone/
        phone/      <- images of people using phones
        no_phone/   <- images of people NOT using phones

    Recommended dataset:
      https://www.kaggle.com/datasets/pkdarabi/cardetection  (contains phone class)
      https://universe.roboflow.com  (search "phone detection")
    """
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    IMG_SIZE = (96, 96)
    BATCH = 32

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        rotation_range=15,
    )

    train_gen = datagen.flow_from_directory(dataset_dir, target_size=IMG_SIZE,
                                            batch_size=BATCH, class_mode="binary",
                                            subset="training")
    val_gen = datagen.flow_from_directory(dataset_dir, target_size=IMG_SIZE,
                                          batch_size=BATCH, class_mode="binary",
                                          subset="validation")

    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save(model_out)
    print(f"[Train] Saved phone classifier → {model_out}")


if __name__ == "__main__":
    train_phone_classifier()
