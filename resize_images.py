import cv2, os

dataset_path = "dataset"
img_size = (224, 224)  # standard input size for CLIP

print(" Resizing all images to 224x224...")

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not path.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Skipped unreadable file: {path}")
            continue
        img = cv2.resize(img, img_size)
        cv2.imwrite(path, img)

print(" All images resized successfully to 224x224!")
