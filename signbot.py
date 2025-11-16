import os
import time
import json
import threading
import collections
from datetime import datetime
from typing import Dict, Tuple

# Environment and API setup
from dotenv import load_dotenv
load_dotenv()

# Streamlit and WebRTC
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ML/AI libraries
import torch
import joblib
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import google.generativeai as genai

# Text-to-speech
import pyttsx3


# CONFIGURATION & CONSTANTS


# API Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Gemini API key not found in .env file. Please add GOOGLE_API_KEY to your .env file.")
    st.stop()

genai.configure(api_key=API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

# PyTorch CPU optimization
torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) - 1)))
torch.set_grad_enabled(False)

# Cache file for Gemini responses
CACHE_FILE = "gemini_cache.json"

# Supported sign language gestures
SUPPORTED_SIGNS = ["hello", "thankyou", "help", "friends", "yes"]



# MODEL LOADING & INITIALIZATION


@st.cache_resource
def load_sign_models() -> Tuple:
    """
    Load and cache the CLIP model and sign language classifier.
    Returns: (clip_model, processor, classifier, scaler, labels, device)
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model for image feature extraction
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load trained sign language classifier
        bundle = joblib.load("sign_model_bundle.pkl")
        clf = bundle["model"]
        scaler = bundle["scaler"]
        labels = bundle["labels"]
        
        return clip_model, processor, clf, scaler, labels, device
    
    except FileNotFoundError:
        st.error("Model file 'sign_model_bundle.pkl' not found. Please ensure the model is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()


# Load models at startup
clip_model, processor, clf, scaler, LABELS, DEVICE = load_sign_models()



# GEMINI RESPONSE CACHE MANAGEMENT


def load_gemini_cache() -> Dict:
    """Load cached Gemini responses from disk."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_gemini_cache(cache: Dict) -> None:
    """Save Gemini response cache to disk."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


# Initialize cache
gemini_cache = load_gemini_cache()



# TEXT-TO-SPEECH FUNCTIONALITY


class TTSEngine:
    """Thread-safe Text-to-Speech engine wrapper."""
    
    def __init__(self):
        self.is_speaking = False
        self.lock = threading.Lock()
    
    def speak(self, text: str) -> None:
        """
        Speak text asynchronously using pyttsx3.
        Runs in a separate thread to avoid blocking the UI.
        """
        def _speak():
            with self.lock:
                if self.is_speaking:
                    return  # Prevent overlapping speech
                self.is_speaking = True
            
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)
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
        """Check if TTS is currently active."""
        with self.lock:
            return self.is_speaking


# Global TTS engine
tts_engine = TTSEngine()



# GEMINI API INTERACTION


def get_gemini_response(user_input: str, input_type: str = "text") -> str:
    """
    Get a response from Gemini API for the given user input.
    
    Args:
        user_input: The detected sign or typed text
        input_type: Type of input - 'sign' or 'text'
    
    Returns:
        Gemini's response text
    """
    # Check cache first
    cache_key = f"{input_type}:{user_input.lower()}"
    if cache_key in gemini_cache:
        return gemini_cache[cache_key]
    
    # Construct appropriate prompt based on input type
    if input_type == "sign":
        system_instruction = (
            "You are a warm, empathetic AI assistant communicating with a user through sign language detection. "
            "The user has signed a gesture. Respond naturally and kindly in 1-2 sentences, as if you're having a conversation."
        )
        prompt = f"{system_instruction}\n\nUser signed: '{user_input}'. Respond appropriately."
    else:  # text
        system_instruction = (
            "You are a helpful, empathetic AI assistant. "
            "Respond naturally and kindly, keeping your response concise (2-3 sentences max)."
        )
        prompt = f"{system_instruction}\n\nUser said: '{user_input}'\n\nRespond appropriately."
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        reply = response.text.strip() if hasattr(response, "text") else str(response)
        
        # Cache the response
        gemini_cache[cache_key] = reply
        save_gemini_cache(gemini_cache)
        
        return reply
    
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        return error_msg



# VIDEO PROCESSOR WITH PERFECT PREDICTION LOGIC FROM APP A


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
                embedding = clip_model.get_image_features(**inputs).squeeze().cpu().numpy()
            
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
    """Initialize Streamlit session state variables."""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'last_spoken_sign' not in st.session_state:
        st.session_state.last_spoken_sign = None
    
    if 'tts_enabled' not in st.session_state:
        st.session_state.tts_enabled = True


def add_to_conversation(user_input: str, response: str, input_type: str):
    """Add a conversation exchange to history."""
    st.session_state.conversation_history.append({
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'user_input': user_input,
        'response': response,
        'input_type': input_type  # 'sign' or 'text'
    })


# STREAMLIT UI CONFIGURATION


st.set_page_config(
    page_title="Sign Language Communication Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .conversation-box {
        background-color: #ffffff;
        padding: 0.8rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 0.8rem;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.3rem;
    }
    .assistant-message {
        background-color: #f1f8e9;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
initialize_session_state()



# MAIN UI LAYOUT


# Header
st.markdown('<p class="main-header">Sign Language Communication Assistant</p>', 
            unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-box">
    <h3>Welcome! This app supports two ways to communicate:</h3>
    <ul>
        <li><strong>Sign Language Detection:</strong> Show signs to the camera (supports: hello, thankyou, help, friends, yes)</li>
        <li><strong>Text Input:</strong> Type your message in the text box</li>
    </ul>
    <p>Responses are displayed as text and spoken aloud for accessibility.</p>
    
</div>
""", unsafe_allow_html=True)

# Create two columns for input methods
col1, col2 = st.columns([1, 1])



# COLUMN 1: SIGN LANGUAGE DETECTION


with col1:
    st.subheader("Sign Language Input")
    
    # WebRTC configuration for video streaming
    rtc_configuration = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    # Start webcam stream with sign processor
    webrtc_ctx = webrtc_streamer(
        key="sign-detection-stream",
        rtc_configuration=rtc_configuration,
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Display current detection status
    sign_status_placeholder = st.empty()
    detected_sign_placeholder = st.empty()


# COLUMN 2: TEXT INPUT


with col2:
    st.subheader("Text Input")
    
    # Text input form
    with st.form(key="text_input_form", clear_on_submit=True):
        user_text = st.text_area(
            "Type your message here:",
            placeholder="Enter your message...",
            height=150,
            help="Type a message and press Send to get a response from the AI assistant"
        )
        
        submit_button = st.form_submit_button("Send Message", use_container_width=True)
        
        if submit_button and user_text.strip():
            with st.spinner("Getting response..."):
                # Get Gemini response for text input
                response = get_gemini_response(user_text.strip(), input_type="text")
                
                # Add to conversation history
                add_to_conversation(user_text.strip(), response, 'text')
                
                # Speak response if TTS is enabled
                if st.session_state.tts_enabled:
                    tts_engine.speak(response)
                
                st.success("Message sent!")
                st.rerun()



# SIDEBAR: SETTINGS & CONVERSATION HISTORY


with st.sidebar:
    st.header("Settings")
    
    # TTS toggle
    st.session_state.tts_enabled = st.checkbox(
        "Enable Text-to-Speech",
        value=st.session_state.tts_enabled,
        help="Speak AI responses aloud for accessibility"
    )
    
    st.divider()
    
    # Supported signs reference
    st.header("Supported Signs")
    for sign in SUPPORTED_SIGNS:
        st.markdown(f"• **{sign.capitalize()}**")
    
    st.divider()
    
    # Conversation history
    st.header("Conversation History")
    
    if st.button("Clear History", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.last_spoken_sign = None
        st.rerun()
    
    # Display conversation history (most recent first)
    if st.session_state.conversation_history:
        for conv in reversed(st.session_state.conversation_history):
            input_icon = {
                'sign': '👋',
                'text': '⌨️',
            }.get(conv['input_type'], '💬')
            
            st.markdown(f"""
            <div class="conversation-box">
                <small><strong>{conv['timestamp']}</strong> {input_icon}</small>
                <div class="user-message">
                    <strong>You ({conv['input_type']}):</strong> {conv['user_input']}
                </div>
                <div class="assistant-message">
                    <strong>Assistant:</strong> {conv['response']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No conversation history yet. Start by signing or typing!")


# SIGN DETECTION PROCESSING LOOP

if webrtc_ctx.video_processor:
    processor: SignLanguageProcessor = webrtc_ctx.video_processor
    
    # Continuous monitoring while webcam is active
    while webrtc_ctx.state.playing:
        time.sleep(0.2)  # Check every 200ms
        
        current_sign, confidence = processor.get_stable_sign()
        
        if current_sign == "no_sign":
            sign_status_placeholder.info("Waiting for sign...")
            detected_sign_placeholder.empty()
        else:
            sign_status_placeholder.success("Sign detected!")
            detected_sign_placeholder.markdown(
                f"### Detected: **{current_sign.upper()}** ({confidence:.1f}%)"
            )
            
            # Process new sign if cooldown passed
            if (current_sign != st.session_state.last_spoken_sign and 
                processor.can_make_api_call()):
                
                with st.spinner(f"Processing '{current_sign}'..."):
                    # Get Gemini response
                    response = get_gemini_response(current_sign, input_type="sign")
                    
                    # Add to conversation history
                    add_to_conversation(current_sign, response, 'sign')
                    
                    # Speak response if TTS enabled
                    if st.session_state.tts_enabled and not tts_engine.is_currently_speaking():
                        tts_engine.speak(response)
                    
                    # Update last spoken sign
                    st.session_state.last_spoken_sign = current_sign
                    
                    # Refresh UI
                    st.rerun()
else:
    sign_status_placeholder.info("Click START above to begin sign detection")