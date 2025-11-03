"""predict.py

Lightweight inference script for the demo Siamese model.
Usage:
    python predict.py --input application_data/input_image/input_image.jpg --verification_dir application_data/verification_images --model siamese_model.keras --threshold 0.5

Notes:
- The script expects a saved model compatible with the notebook. It includes a local L1Dist layer
  definition (the notebook uses the same). If you changed custom classes in the notebook, update
  this file accordingly.
"""

import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf


class L1Dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)


def preprocess(path, size=(105, 105)):
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    arr = np.array(img).astype('float32') / 255.0
    return arr


def load_model(path):
    return tf.keras.models.load_model(path, custom_objects={'L1Dist': L1Dist})


def verify(model, input_img_path, verification_dir, detection_threshold=0.5):
    if not os.path.exists(input_img_path):
        raise FileNotFoundError(f"Input image not found: {input_img_path}")

    verification_images = [os.path.join(verification_dir, f) for f in os.listdir(verification_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(verification_images) == 0:
        raise FileNotFoundError(f"No verification images found in {verification_dir}")

    input_img = preprocess(input_img_path)
    results = []
    for vpath in verification_images:
        vimg = preprocess(vpath)
        x1 = np.expand_dims(input_img, axis=0)
        x2 = np.expand_dims(vimg, axis=0)
        score = model.predict([x1, x2])[0][0]
        results.append((os.path.basename(vpath), float(score)))

    # Count how many comparisons exceed the detection threshold
    positives = sum(1 for _, s in results if s > detection_threshold)
    verification = positives / len(results)
    return results, verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input image (application_data/input_image/input_image.jpg)')
    parser.add_argument('--verification_dir', required=True, help='Directory of verification images')
    parser.add_argument('--model', default='siamese_model.keras', help='Path to saved model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Per-comparison detection threshold (0-1)')
    args = parser.parse_args()

    print('Loading model...')
    model = load_model(args.model)
    print('Running verification...')
    results, verification = verify(model, args.input, args.verification_dir, detection_threshold=args.threshold)

    for name, score in results:
        print(f"{name}: {score:.4f}")

    print(f"Verification score: {verification:.3f} (threshold: {args.threshold})")
    ok = verification > 0.5
    print('VERIFIED' if ok else 'NOT VERIFIED')


if __name__ == '__main__':
    main()
