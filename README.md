# SignBot - Sign Language Communication Assistant

A real-time sign language recognition system that converts hand gestures into text, processes them using Google Gemini AI, and provides intelligent responses through text and audio.

## Project Overview

SignBot is an AI-powered communication assistant that helps sign language users interact naturally with others. The system uses a camera to detect hand gestures, recognizes them using machine learning, converts them to text, and generates contextual responses using Google's Gemini AI. Responses are delivered both as text and audio, making communication seamless.

**Supported Gestures**: friends, hello, help, thankyou, yes

## Why This App Matters

### For Sign Language Users
- Enables natural communication without needing an interpreter
- Converts their signs into spoken language instantly
- Provides a bridge to communicate with anyone

### For Blind Users
- All responses are delivered through audio (text-to-speech)
- Can hear the conversation without needing to read text
- Makes sign language conversations accessible to them

### For Everyone Else
- No need to learn sign language to communicate
- Easy-to-use interface that anyone can operate
- Promotes inclusive communication for all

## Features

- **Real-Time Sign Recognition**: Detects and recognizes hand gestures from webcam
- **AI-Powered Responses**: Uses Google Gemini for intelligent, context-aware replies
- **Text-to-Speech Output**: Speaks responses aloud for accessibility
- **Text Input Option**: Users can also type messages directly
- **Response Caching**: Stores common responses to reduce API calls
- **User-Friendly Interface**: Simple Streamlit web app
- **High Accuracy Model**: SVM classifier trained on CLIP features

## Installation

### Prerequisites
- Python 3.8+
- Webcam
- Google Gemini API Key ([Get it here](https://makersuite.google.com/app/apikey))

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/signbot.git
cd signbot
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API key**

Create a `.env` file and add:
```
GEMINI_API_KEY=your_api_key_here
```

5. **Create datasets folder**
```bash
mkdir datasets
```

## How to Run

### Main Application (SignBot Assistant)
```bash
streamlit run signbot.py
```
This is the complete sign language communication assistant with Gemini AI integration.

### Real-Time Prediction App
```bash
streamlit run app.py
```
Simple app that shows live predictions with text-to-speech.

### Test Model Prediction
```bash
python predict_sign.py
```

## Requirements

- **numpy**: Numerical operations
- **pandas**: Data handling
- **scikit-learn**: Machine learning (SVM, Random Forest, MLP)
- **torch & torchvision**: Deep learning framework
- **transformers**: CLIP model
- **pillow**: Image processing
- **streamlit**: Web interface
- **pyttsx3**: Text-to-speech
- **google-generativeai**: Gemini AI API
- **opencv-python**: Computer vision
- **matplotlib**: Visualization

## How the Model Works

### Training Process
1. **Feature Extraction**: Uses OpenAI's CLIP model to convert images into feature vectors
2. **Classification**: Three models tested (SVM, Random Forest, MLP)
3. **Best Model**: SVM achieved highest accuracy and was selected
4. **Visualization**: PCA shows how CLIP clusters different gestures

### Prediction Flow
```
Camera → Capture Frame → CLIP Features → SVM Prediction → Text → Gemini AI → Response (Text + Audio)
```

### Why CLIP + SVM?
- CLIP provides powerful visual features without training from scratch
- SVM works well with high-dimensional data
- Fast inference for real-time predictions
- High accuracy with limited training data

## File Descriptions

### Data Collection
- **`capture_dataset.py`** - Collects dataset images for gestures: friends, hello, help, thankyou, yes
- **`resize_images.py`** - Resizes images to 224×224
- **`augment_data.py`** - Performs data augmentation

### Analysis & Training
- **`dataset_summary.py`** - Shows dataset summary
- **`train_model.py`** - Uses CLIP to extract features; trains SVM, Random Forest, and MLP; SVM gave highest accuracy; includes PCA visualization of feature clusters

### Prediction & Apps
- **`predict_sign.py`** - Loads the trained model and predicts gestures
- **`app.py`** - Streamlit app that predicts signs in real-time and speaks the prediction using TTS
- **`signbot.py`** - Main Sign Language Communication Assistant
  - Sign input → converted to text → sent to Gemini → Gemini responds with text or audio
  - Also includes text input: User can type a message, and Gemini responds through both text and audio
  - Responses are saved in `gemini_cache.json` to avoid API conflicts
  - Run using: `streamlit run signbot.py`

## Project Structure

```
signbot/
├── datasets/           # Training images (not uploaded to GitHub)
├── capture_dataset.py
├── resize_images.py
├── augment_data.py
├── dataset_summary.py
├── train_model.py
├── predict_sign.py
├── app.py
├── signbot.py         # Main application
├── requirements.txt
├── .gitignore
└── README.md
```

## Training Your Own Model

1. Collect images: `python capture_dataset.py`
2. Resize images: `python resize_images.py`
3. Augment data: `python augment_data.py`
4. Check dataset: `python dataset_summary.py`
5. Train model: `python train_model.py`

## Future Improvements

- Add more gestures
- Improve gesture smoothing
- Add voice input
- Mobile app version
- Continuous learning from feedback
