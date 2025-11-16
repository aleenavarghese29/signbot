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
from datetime import datetime

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration



# PAGE CONFIGURATION


st.set_page_config(
    page_title="Sign Language Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sign-display {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)



# MODEL LOADING


@st.cache_resource
def load_models():
    """Load CLIP model and trained classifier."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load trained bundle
    bundle = joblib.load("sign_model_bundle.pkl")
    classifier = bundle["model"]
    scaler = bundle["scaler"]
    labels = bundle["labels"]
    
    return clip_model, clip_processor, classifier, scaler, labels, device


# Load models at startup
model, processor, clf, scaler, labels, DEVICE = load_models()



# TEXT-TO-SPEECH ENGINE


class TTSEngine:
    """Thread-safe Text-to-Speech engine."""
    
    def __init__(self):
        self.is_speaking = False
        self.lock = threading.Lock()
    
    def speak(self, text: str) -> None:
        """Speak text asynchronously."""
        def _speak():
            with self.lock:
                if self.is_speaking:
                    return
                self.is_speaking = True
            
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 165)
                engine.setProperty('volume', 0.9)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                with self.lock:
                    self.is_speaking = False
        
        threading.Thread(target=_speak, daemon=True).start()
    
    def is_currently_speaking(self) -> bool:
        with self.lock:
            return self.is_speaking


# Global TTS engine
tts_engine = TTSEngine()



# VIDEO PROCESSOR WITH FIXED PREDICTION LOGIC


class SignLanguageProcessor(VideoTransformerBase):
    """
    Video processor with HELP vs FRIENDS balancing fix.
    DO NOT MODIFY - This contains the working prediction logic.
    """
    
    def __init__(self):
        # Stabilization setup
        self.prediction_history = collections.deque(maxlen=15)
        self.confidence_history = collections.deque(maxlen=15)
        self.current_stable_sign = "no_sign"
        self.current_confidence = 0.0
        self.last_api_call_time = 0
        self.frame_count = 0
        self.lock = threading.Lock()
        
        # Constants
        self.CONF_THRESHOLD = 60
        self.COOLDOWN_PERIOD = 2.0
    
    def recv(self, frame):
        """Process incoming video frame."""
        self.frame_count += 1
        
        # Skip alternate frames for speed
        if self.frame_count % 2 != 0:
            return frame
        
        img = frame.to_image().convert("RGB")
        
        try:
            # Resize for processing
            img_resized = img.resize((224, 224))
            
            # CLIP preprocessing
            inputs = processor(images=img_resized, return_tensors="pt").to(DEVICE)
            
            # Extract CLIP embeddings
            with torch.no_grad():
                embedding = model.get_image_features(**inputs).squeeze().cpu().numpy()
            
            # Normalize using saved scaler
            embedding_scaled = scaler.transform([embedding])[0]
            
            # Predict probabilities
            probs = clf.predict_proba([embedding_scaled])[0]
            pred_label = clf.classes_[np.argmax(probs)]
            confidence = np.max(probs) * 100
            
            # FIX #1: BALANCED CLASS THRESHOLDS
            
            class_thresholds = {
                "hello": 60,
                "thankyou": 58,
                "friends": 68,    # Higher threshold
                "help": 48,       # Lower threshold
                "yes": 55
            }
            
            dynamic_threshold = class_thresholds.get(pred_label, self.CONF_THRESHOLD)
            
            
            # FIX #2: AGGRESSIVE CONFIDENCE REBALANCING
           
            recent_5 = list(self.prediction_history)[-5:] if len(self.prediction_history) >= 5 else []
            has_help_recently = "help" in recent_5
            has_friends_recently = "friends" in recent_5
            
            # Apply strong rebalancing to overcome model bias
            if has_help_recently and has_friends_recently:
                # Confusion detected - strongly favor help
                if pred_label == "help":
                    confidence += 12
                elif pred_label == "friends":
                    confidence -= 8
            else:
                # No confusion - moderate adjustments
                if pred_label == "help":
                    confidence += 8
                elif pred_label == "friends":
                    confidence -= 5
            
            
            # FIX #2b: PROBABILITY REDISTRIBUTION
            
            help_idx = np.where(clf.classes_ == "help")[0]
            friends_idx = np.where(clf.classes_ == "friends")[0]
            
            if len(help_idx) > 0 and len(friends_idx) > 0:
                help_prob = probs[help_idx[0]]
                friends_prob = probs[friends_idx[0]]
                
                if friends_prob > help_prob and (friends_prob - help_prob) > 0.10:
                    steal_amount = min(0.15, (friends_prob - help_prob) * 0.5)
                    
                    adjusted_probs = probs.copy()
                    adjusted_probs[help_idx[0]] += steal_amount
                    adjusted_probs[friends_idx[0]] -= steal_amount
                    
                    new_pred_idx = np.argmax(adjusted_probs)
                    if clf.classes_[new_pred_idx] != pred_label:
                        pred_label = clf.classes_[new_pred_idx]
                        confidence = adjusted_probs[new_pred_idx] * 100
            
            
            # FIX #3: TRACK PREDICTION AND CONFIDENCE
            
            if confidence >= dynamic_threshold:
                self.prediction_history.append(pred_label)
                self.confidence_history.append(confidence)
            
            
            # FIX #4: SMART MAJORITY VOTING WITH CONSISTENCY CHECK
            
            if len(self.prediction_history) >= 5:
                recent_predictions = list(self.prediction_history)[-5:]
                vote_counts_recent = {}
                for pred in recent_predictions:
                    vote_counts_recent[pred] = vote_counts_recent.get(pred, 0) + 1
                
                vote_counts_all = {}
                for pred in self.prediction_history:
                    vote_counts_all[pred] = vote_counts_all.get(pred, 0) + 1
                
                valid_candidates = []
                
                for label in set(self.prediction_history):
                    recent_votes = vote_counts_recent.get(label, 0)
                    total_votes = vote_counts_all.get(label, 0)
                    
                    is_valid = False
                    
                    if label == "friends":
                        if recent_votes >= 4 or (total_votes >= 8 and recent_votes >= 3):
                            is_valid = True
                    elif label == "help":
                        if recent_votes >= 1 or total_votes >= 3:
                            is_valid = True
                    else:
                        if recent_votes >= 2 or total_votes >= 5:
                            is_valid = True
                    
                    if is_valid:
                        score = (recent_votes * 2.0) + (total_votes * 1.0)
                        valid_candidates.append((label, score, recent_votes, total_votes))
                
                if valid_candidates:
                    valid_candidates.sort(key=lambda x: x[1], reverse=True)
                    
                    if len(valid_candidates) >= 2:
                        top1_label = valid_candidates[0][0]
                        top2_label = valid_candidates[1][0]
                        
                        if set([top1_label, top2_label]) == set(["help", "friends"]):
                            help_data = next((x for x in valid_candidates if x[0] == "help"), None)
                            friends_data = next((x for x in valid_candidates if x[0] == "friends"), None)
                            
                            if help_data and friends_data:
                                help_recent = help_data[2]
                                friends_recent = friends_data[2]
                                
                                if help_recent > friends_recent:
                                    stable_prediction = "help"
                                elif friends_recent > help_recent:
                                    stable_prediction = "friends"
                                else:
                                    stable_prediction = "help"
                            else:
                                stable_prediction = valid_candidates[0][0]
                        else:
                            stable_prediction = valid_candidates[0][0]
                    else:
                        stable_prediction = valid_candidates[0][0]
                else:
                    stable_prediction = "no_sign"
            else:
                stable_prediction = "no_sign"
            
            # Calculate display confidence
            if stable_prediction != "no_sign" and len(self.prediction_history) > 0:
                stable_indices = [
                    i for i, p in enumerate(self.prediction_history) 
                    if p == stable_prediction
                ]
                if stable_indices:
                    avg_conf = np.mean([self.confidence_history[i] for i in stable_indices])
                else:
                    avg_conf = confidence
            else:
                avg_conf = confidence
            
            # Update stable sign (thread-safe)
            with self.lock:
                self.current_stable_sign = stable_prediction
                self.current_confidence = avg_conf
                    
        except Exception as e:
            print(f"Frame processing error: {e}")
        
        return frame
    
    def get_stable_sign(self) -> tuple:
        """Thread-safe getter for current prediction."""
        with self.lock:
            return self.current_stable_sign, self.current_confidence
    
    def can_make_api_call(self) -> bool:
        """Check if cooldown period has passed."""
        with self.lock:
            current_time = time.time()
            if current_time - self.last_api_call_time >= self.COOLDOWN_PERIOD:
                self.last_api_call_time = current_time
                return True
            return False



# SESSION STATE INITIALIZATION


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'last_spoken_sign' not in st.session_state:
        st.session_state.last_spoken_sign = None
    
    if 'tts_enabled' not in st.session_state:
        st.session_state.tts_enabled = True


# MAIN UI


# Initialize session state
initialize_session_state()

# Header
st.markdown('<p class="main-header">Real-Time Sign Language Detection</p>', 
            unsafe_allow_html=True)

# Two columns: webcam and controls
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Webcam Feed")
    
    # WebRTC configuration
    rtc_configuration = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    # Start webcam stream
    webrtc_ctx = webrtc_streamer(
        key="sign-detection",
        rtc_configuration=rtc_configuration,
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Settings")
    
    # TTS toggle
    st.session_state.tts_enabled = st.checkbox(
        "Enable Voice Output",
        value=st.session_state.tts_enabled,
        help="Speak detected signs aloud"
    )
    
    st.divider()
    
    st.subheader("Supported Signs")
    st.markdown("• **Hello**")
    st.markdown("• **Thank You**")
    st.markdown("• **Help**")
    st.markdown("• **Friends**")
    st.markdown("• **Yes**")

# Sign detection display
st.divider()

sign_status_placeholder = st.empty()
detected_sign_placeholder = st.empty()

# Processing loop
if webrtc_ctx.video_processor:
    processor: SignLanguageProcessor = webrtc_ctx.video_processor
    
    while webrtc_ctx.state.playing:
        time.sleep(0.2)
        
        current_sign, confidence = processor.get_stable_sign()
        
        if current_sign == "no_sign":
            sign_status_placeholder.info("Waiting for sign...")
            detected_sign_placeholder.empty()
        else:
            sign_status_placeholder.success("Sign detected!")
            detected_sign_placeholder.markdown(
                f'<div class="sign-display">{current_sign.upper()} ({confidence:.1f}%)</div>',
                unsafe_allow_html=True
            )
            
            # Speak new sign
            if (current_sign != st.session_state.last_spoken_sign and 
                processor.can_make_api_call()):
                
                if st.session_state.tts_enabled and not tts_engine.is_currently_speaking():
                    tts_engine.speak(f"You signed {current_sign}")
                
                st.session_state.last_spoken_sign = current_sign
else:
    sign_status_placeholder.info("Click START above to begin detection")


