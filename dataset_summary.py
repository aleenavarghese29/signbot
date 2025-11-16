import os

dataset_path = "dataset"

print("\n Dataset + Augmented Summary (Corrected):")
print("-" * 50)

total_original = 0
total_augmented = 0

for label in sorted(os.listdir(dataset_path)):
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path):
        continue

    # Count original images (directly inside class folder)
    orig_count = len([
        f for f in os.listdir(label_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    total_original += orig_count

    # Count augmented images (inside 'augmented' subfolder)
    aug_path = os.path.join(label_path, "augmented")
    aug_count = 0
    if os.path.exists(aug_path):
        aug_count = len([
            f for f in os.listdir(aug_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        total_augmented += aug_count

    print(f"{label:<10} | Original: {orig_count:>3} | Augmented: {aug_count:>3} | Total: {orig_count + aug_count:>3}")

print("-" * 50)
print(f"Total Original Images:  {total_original}")
print(f"Total Augmented Images: {total_augmented}")
print(f"Total Combined Images:  {total_original + total_augmented}\n")
