import torch
import cv2
import joblib
import numpy as np
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pyttsx3
import threading
import collections


# Load CLIP Model and Trained Bundle

print("Loading CLIP model (OpenAI)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Loading trained model bundle...")
bundle = joblib.load("sign_model_bundle.pkl")
clf = bundle["model"]
scaler = bundle["scaler"]
labels = bundle["labels"]

print(f"Model loaded successfully! Classes: {labels}\n")


# Voice Output Setup

def speak_text(text):
    """Speak text asynchronously (non-blocking)."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 165)
    engine.setProperty('volume', 0.9)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

#  Webcam Setup

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found. Check your webcam connection.")
    exit()

print("Camera started — perform your signs! (Press ESC to exit)")



# Stabilization & Confidence Setup (replace your existing setup)
prediction_history = collections.deque(maxlen=15)  # Increased for better voting
confidence_history = collections.deque(maxlen=15)  # Track confidence per prediction
last_spoken = None
last_speak_time = 0
frame_count = 0

CONF_THRESHOLD = 60  # Base threshold


# Real-Time Prediction Loop (replace the while loop content)
stable_prediction = "no_sign"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue  # Skip alternate frames for speed

    # Convert frame for CLIP
    frame = cv2.resize(frame, (320, 240))
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img, return_tensors="pt").to(device)

    # Extract CLIP embeddings
    with torch.no_grad():
        embedding = model.get_image_features(**inputs).squeeze().cpu().numpy()

    # Normalize using saved scaler
    embedding = scaler.transform([embedding])[0]

    # Predict probabilities
    probs = clf.predict_proba([embedding])[0]
    pred_label = clf.classes_[np.argmax(probs)]
    confidence = np.max(probs) * 100

    # BALANCED CLASS THRESHOLDS
    
    
    class_thresholds = {
        "hello": 60,      
        "thankyou": 58,   
        "friends": 68,    
        "help": 48,       
        "yes": 55        
    }
    
    dynamic_threshold = class_thresholds.get(pred_label, CONF_THRESHOLD)
    
    
    # CONFIDENCE REBALANCING
   
    
    # Check if both "help" and "friends" are in recent history
    recent_5 = list(prediction_history)[-5:] if len(prediction_history) >= 5 else []
    has_help_recently = "help" in recent_5
    has_friends_recently = "friends" in recent_5
    
    # CRITICAL: Apply strong rebalancing to overcome model bias
    if has_help_recently and has_friends_recently:
        # Confusion detected - strongly favor help
        if pred_label == "help":
            confidence += 12  # MAJOR boost for help when confused
        elif pred_label == "friends":
            confidence -= 8   #MAJOR penalty for friends when confused
    else:
        # No confusion - moderate adjustments
        if pred_label == "help":
            confidence += 8   # Strong boost for help (compensate for weak model)
        elif pred_label == "friends":
            confidence -= 5   # Moderate penalty for friends (reduce dominance)
    
  
    # PROBABILITY REDISTRIBUTION 
    
    # Get the raw probabilities for help and friends
    help_idx = np.where(clf.classes_ == "help")[0]
    friends_idx = np.where(clf.classes_ == "friends")[0]
    
    if len(help_idx) > 0 and len(friends_idx) > 0:
        help_prob = probs[help_idx[0]]
        friends_prob = probs[friends_idx[0]]
        
        # If friends is winning by a large margin, steal some probability
        if friends_prob > help_prob and (friends_prob - help_prob) > 0.10:
            # Redistribute: take from friends, give to help
            steal_amount = min(0.15, (friends_prob - help_prob) * 0.5)
            
            # Update the predicted label if this changes the winner
            adjusted_probs = probs.copy()
            adjusted_probs[help_idx[0]] += steal_amount
            adjusted_probs[friends_idx[0]] -= steal_amount
            
            # Recalculate winner
            new_pred_idx = np.argmax(adjusted_probs)
            if clf.classes_[new_pred_idx] != pred_label:
                pred_label = clf.classes_[new_pred_idx]
                confidence = adjusted_probs[new_pred_idx] * 100
                print(f"  🔄 Probability redistribution: switched to {pred_label}")
    
    
    # Store confidence alongside prediction for weighted voting
    
    if confidence >= dynamic_threshold:
        prediction_history.append(pred_label)
        confidence_history.append(confidence)
  
    
    if len(prediction_history) >= 5:  # Need at least 5 samples
        # Method 1: Check for consistent predictions (3+ in last 5)
        recent_predictions = list(prediction_history)[-5:]
        vote_counts_recent = {}
        for pred in recent_predictions:
            vote_counts_recent[pred] = vote_counts_recent.get(pred, 0) + 1
        
        # Method 2: Check overall vote counts
        vote_counts_all = {}
        for pred in prediction_history:
            vote_counts_all[pred] = vote_counts_all.get(pred, 0) + 1
        
       
        
        valid_candidates = []
        
        for label in set(prediction_history):
            recent_votes = vote_counts_recent.get(label, 0)
            total_votes = vote_counts_all.get(label, 0)
            
            # Apply class-specific consistency rules
            is_valid = False
            
            if label == "friends":
                
               
                if recent_votes >= 4 or (total_votes >= 8 and recent_votes >= 3):
                    is_valid = True
            
            elif label == "help":
              
                if recent_votes >= 1 or total_votes >= 3:
                    is_valid = True
            
            else:
                # Other signs use standard rules
                if recent_votes >= 2 or total_votes >= 5:
                    is_valid = True
            
            if is_valid:
                # Calculate score: recent votes matter 2x more
                score = (recent_votes * 2.0) + (total_votes * 1.0)
                valid_candidates.append((label, score, recent_votes, total_votes))
        
        
        if valid_candidates:
            # Sort by score (highest first)
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Special case: If top 2 are "help" and "friends"
            if len(valid_candidates) >= 2:
                top1_label = valid_candidates[0][0]
                top2_label = valid_candidates[1][0]
                
                if set([top1_label, top2_label]) == set(["help", "friends"]):
                    # They're competing - use recency + confidence
                    help_data = next((x for x in valid_candidates if x[0] == "help"), None)
                    friends_data = next((x for x in valid_candidates if x[0] == "friends"), None)
                    
                    if help_data and friends_data:
                        help_recent = help_data[2]
                        friends_recent = friends_data[2]
                        
                        # If help has more recent votes, it wins
                        if help_recent > friends_recent:
                            stable_prediction = "help"
                        elif friends_recent > help_recent:
                            stable_prediction = "friends"
                        else:
                            # Tied on recency - ALWAYS prefer help
                            # This is the tiebreaker to overcome model bias
                            stable_prediction = "help"
                            print("Tie-breaker: choosing 'help' over 'friends'")
                    else:
                        stable_prediction = valid_candidates[0][0]
                else:
                    # No help/friends conflict - pick highest score
                    stable_prediction = valid_candidates[0][0]
            else:
                # Only one valid candidate
                stable_prediction = valid_candidates[0][0]
        else:
            stable_prediction = "no_sign"
    else:
        stable_prediction = "no_sign"
    
   
    # Prevent quick changes back to "no_sign"
    if stable_prediction == "no_sign" and time.time() - last_speak_time < 1.5:
        if last_spoken is not None:
            stable_prediction = last_spoken
    
    # Require minimum hold time before changing predictions
    # BUT: Special rules for help/friends transitions
    if stable_prediction != last_spoken:
        if last_spoken == "friends" and stable_prediction == "help":
            # Allow INSTANT correction from friends → help
            required_hold = 0.3  # ⚡ Ultra-fast transition (help overrides immediately)
        elif last_spoken == "help" and stable_prediction == "friends":
            # Require VERY long hold for help → friends (prevent false flips)
            required_hold = 1.8  # 🛑 Slow transition (friends must prove itself heavily)
        else:
            required_hold = 1.0  # Normal hold time for other transitions
        
        if time.time() - last_speak_time < required_hold:
            continue

    # CALCULATE DISPLAY CONFIDENCE

    # Average confidence of the stable prediction in recent history
    if stable_prediction != "no_sign" and len(prediction_history) > 0:
        stable_indices = [
            i for i, p in enumerate(prediction_history) 
            if p == stable_prediction
        ]
        if stable_indices:
            avg_conf = np.mean([confidence_history[i] for i in stable_indices])
        else:
            avg_conf = confidence
    else:
        avg_conf = confidence
    
   
    # DISPLAY OVERLAY
    if stable_prediction != "no_sign":
        color = (0, 255, 0) if avg_conf >= 60 else (0, 165, 255)
        cv2.rectangle(frame, (20, 20), (330, 100), color, 2)
        cv2.putText(
            frame, 
            f"Sign: {stable_prediction} ({avg_conf:.1f}%)",
            (30, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            color, 
            2
        )
        
        # DEBUG: Show vote distribution (optional - comment out if not needed)
        if len(prediction_history) >= 5 and 'vote_counts_recent' in locals():
            debug_y = 120
            cv2.putText(
                frame,
                "Recent votes (last 5 frames):",
                (30, debug_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1
            )
            debug_y += 18
            for label in ["help", "friends", "hello", "thankyou", "yes"]:
                count = vote_counts_recent.get(label, 0)
                if count > 0:
                    cv2.putText(
                        frame,
                        f"  {label}: {count}",
                        (30, debug_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255) if label == stable_prediction else (150, 150, 150),
                        1
                    )
                    debug_y += 18
    
    cv2.imshow("Real-Time Sign Detection", frame)
    
    
    # SPEAK OUTPUT (NON-BLOCKING)
   
    if (
        stable_prediction != last_spoken
        and stable_prediction != "no_sign"
        and time.time() - last_speak_time > 2
    ):
        print(f"You signed: {stable_prediction} (confidence: {avg_conf:.1f}%)")
        threading.Thread(
            target=speak_text,
            args=(f"You signed {stable_prediction}",),
            daemon=True
        ).start()
        last_spoken = stable_prediction
        last_speak_time = time.time()
    
    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    time.sleep(0.08)


# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Camera closed. Goodbye!")