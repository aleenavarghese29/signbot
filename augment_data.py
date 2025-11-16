from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

dataset_path = "dataset"

# Augmentation configuration
aug = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode="nearest"
)

print(" Starting data augmentation...")

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(folder, file)
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Save to "augmented" subfolder
        save_dir = os.path.join(folder, "augmented")
        os.makedirs(save_dir, exist_ok=True)

        # Create 3 new variations per original image
        i = 0
        for batch in aug.flow(
            x,
            batch_size=1,
            save_to_dir=save_dir,
            save_prefix=label,
            save_format='jpg'
        ):
            i += 1
            if i >= 3:
                break

print(" Data augmentation complete! Check each folder’s 'augmented/' subfolder for new images.")
