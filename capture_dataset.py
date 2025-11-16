import cv2
import os
import time

# -----------------------------
# ✋ Your gesture labels
# -----------------------------
labels = ["hello", "thankyou", "friends", "help", "yes"]
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

# -----------------------------
# 📷 Camera setup
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found. Please check your webcam connection.")
    exit()

# -----------------------------
# ⚙️ Capture settings
# -----------------------------
num_images = 80             # number of images per gesture
delay_between_images = 1.2  # seconds between captures

print("\n✅ Camera started successfully.")
print("📸 You will capture 80 images per gesture.")
print("👉 Press ESC anytime to exit.\n")

# -----------------------------
# 🧠 Capture loop for each gesture
# -----------------------------
for label in labels:
    label_path = os.path.join(dataset_path, label)
    os.makedirs(label_path, exist_ok=True)

    print(f"\n✋ Prepare to show: '{label.upper()}'")
    print("Get ready... starting in:")

    # Countdown before starting capture
    for i in range(3, 0, -1):
        print(f"⏳ {i}...")
        time.sleep(1)

    print(f"🎬 Capturing '{label}' images now...")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame capture failed. Skipping...")
            continue

        # Flip horizontally for a natural mirror view
        frame = cv2.flip(frame, 1)

        # Display the label and progress
        cv2.putText(frame, f"Sign: {label} ({count+1}/{num_images})",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("✋ Dataset Capture", frame)

        # Save frame in the corresponding folder
        img_path = os.path.join(label_path, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"💾 Saved: {img_path}")
        count += 1

        # Delay before capturing the next image
        time.sleep(delay_between_images)

        # Stop early if ESC is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            print("🛑 Exiting early...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f"✅ Completed capturing for '{label}' ({num_images} images).")

# -----------------------------
# 🏁 Wrap up
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("\n🎉 Dataset capture complete! 80 images per gesture saved successfully.")
print(f"📂 Saved in: {os.path.abspath(dataset_path)}")
