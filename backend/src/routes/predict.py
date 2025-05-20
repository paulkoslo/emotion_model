import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Blueprint, request, jsonify
import whisper # type: ignore
import tempfile
import logging
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log TensorFlow and Keras versions
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"Keras version: {tf.keras.__version__}")

predict_bp = Blueprint("predict_bp", __name__)

# --- Keras and Tokenizer Setup ---
model2_Class = None  # Using the same variable name as in notebook for consistency
tokenizer = None
MAX_SEQ_LENGTH = 900  # As specified by the user
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "model2_class.keras")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "tokenizer.pickle")

# Define emotion labels - same as in notebook
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire',
    'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise'
]

try:
    logger.info(f"Loading Keras model from {MODEL_PATH}...")
    if os.path.exists(MODEL_PATH):
        model2_Class = load_model(MODEL_PATH)
        logger.info("Keras model loaded successfully.")
        # Log model summary
        model2_Class.summary(print_fn=logger.info)
    else:
        logger.error(f"Keras model file not found at {MODEL_PATH}")

    logger.info(f"Loading Keras Tokenizer from {TOKENIZER_PATH}...")
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, "rb") as handle:
            tokenizer = pickle.load(handle)
        logger.info("Keras Tokenizer loaded successfully.")
        # Log tokenizer details
        logger.info(f"Tokenizer word_index size: {len(tokenizer.word_index)}")
        logger.info(f"Tokenizer document_count: {tokenizer.document_count}")
        # Log a few sample word-to-index mappings
        sample_words = list(tokenizer.word_index.items())[:10]
        logger.info(f"Sample word-to-index mappings: {sample_words}")
    else:
        logger.error(f"Keras Tokenizer file not found at {TOKENIZER_PATH}")

except Exception as e:
    logger.error(f"Error loading Keras model or tokenizer: {e}")
    import traceback
    logger.error(traceback.format_exc())

# Load Whisper model (using "base" as requested)
try:
    logger.info("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    whisper_model = None

# Using the exact analyze_emotion function from the notebook
def analyze_emotion(test_sentence, model2_Class, tokenizer, MAX_SEQ_LENGTH):
    """
    Analyze emotion in text using the same function as in the notebook.
    This ensures exact consistency between notebook and web app.
    """
    logger.info(f"Analyzing emotion for text: '{test_sentence}'")
    
    # Process the input sentence
    test_sequence = tokenizer.texts_to_sequences([test_sentence])
    logger.info(f"Tokenized sequence: {test_sequence}")
    
    test_padded = pad_sequences(test_sequence, maxlen=MAX_SEQ_LENGTH, padding='post')
    logger.info(f"Padded sequence shape: {test_padded.shape}")
    logger.info(f"Padded sequence content: {test_padded[0]}")
    
    predictions = model2_Class.predict(test_padded, verbose=0)
    logger.info(f"Raw prediction shape: {predictions.shape if predictions is not None else 'None'}")
    logger.info(f"Raw prediction values: {predictions[0] if predictions is not None else 'None'}")

    # Get predicted emotion
    predicted_emotion = EMOTION_LABELS[np.argmax(predictions[0])]
    logger.info(f"Predicted emotion: {predicted_emotion}")
    
    # Log top 5 emotions by probability
    emotion_probs = [(emotion, float(prob)) for emotion, prob in zip(EMOTION_LABELS, predictions[0])]
    top_emotions = sorted(emotion_probs, key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"Top 5 predicted emotions: {top_emotions}")
    
    return predicted_emotion, predictions[0]

# Comprehensive emotion to sentiment mapping
def map_emotion_to_sentiment(specific_emotion):
    sentiment_map = {
        'admiration': 'Positive', 'amusement': 'Positive', 'approval': 'Positive', 'caring': 'Positive',
        'desire': 'Positive', 'excitement': 'Positive', 'gratitude': 'Positive', 'joy': 'Positive',
        'love': 'Positive', 'optimism': 'Positive', 'pride': 'Positive', 'relief': 'Positive', 'surprise': 'Positive',
        'anger': 'Negative', 'annoyance': 'Negative', 'disappointment': 'Negative', 'disapproval': 'Negative',
        'disgust': 'Negative', 'embarrassment': 'Negative', 'fear': 'Negative', 'grief': 'Negative',
        'nervousness': 'Negative', 'remorse': 'Negative', 'sadness': 'Negative',
        'confusion': 'Neutral', 'curiosity': 'Neutral', 'realization': 'Neutral'
    }
    sentiment = sentiment_map.get(specific_emotion, 'Neutral')
    logger.info(f"Mapped specific emotion '{specific_emotion}' to sentiment '{sentiment}'")
    return sentiment

@predict_bp.route("/api/predict", methods=["POST"])
def predict_emotion():
    logger.info("Received request at /api/predict")
    input_text = None

    if not model2_Class or not tokenizer:
        logger.error("Keras model/tokenizer not available. Prediction endpoint is offline.")
        return jsonify({"error": "Emotion prediction service is currently unavailable due to model loading issues. Please try again later."}), 503

    if 'audio' in request.files:
        logger.info("Audio file detected in request.")
        audio_file = request.files['audio']
        if not whisper_model:
            logger.error("Whisper model not loaded, cannot process audio.")
            return jsonify({"error": "Speech processing service is unavailable"}), 500
        tmp_audio_path = None
        try:
            fd, tmp_audio_path = tempfile.mkstemp(suffix=".webm")
            with os.fdopen(fd, 'wb') as tmp_audio_file:
                 audio_file.save(tmp_audio_file)

            logger.info(f"Audio file saved temporarily to {tmp_audio_path}")
            logger.info("Starting audio transcription...")
            result = whisper_model.transcribe(tmp_audio_path)
            input_text = result["text"]
            logger.info(f"Transcription result: {input_text}")
        except Exception as e:
            logger.error(f"Error during audio processing: {e}")
            return jsonify({"error": f"Error processing audio: {str(e)}"}), 500
        finally:
            if tmp_audio_path and os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
                logger.info(f"Temporary audio file {tmp_audio_path} removed.")
    elif 'text' in request.form:
        input_text = request.form['text']
        logger.info(f"Text input detected: {input_text}")
    else:
        logger.warning("No audio or text input provided.")
        return jsonify({"error": "No audio or text input provided"}), 400

    if not input_text or input_text.strip() == "":
        logger.warning("Input text is empty after processing or only whitespace.")
        return jsonify({"error": "Input text is empty or invalid"}), 400

    # Add text normalization to match notebook behavior
    input_text = input_text.strip()
    logger.info(f"Normalized input text: '{input_text}'")

    try:
        # Use the analyze_emotion function from the notebook
        specific_emotion, raw_predictions = analyze_emotion(input_text, model2_Class, tokenizer, MAX_SEQ_LENGTH)
        
        # Convert raw predictions to dictionary for frontend
        emotion_probabilities = {emotion: float(prob) for emotion, prob in zip(EMOTION_LABELS, raw_predictions)}
        
        overall_sentiment = map_emotion_to_sentiment(specific_emotion)

        probabilities_for_frontend = sorted(
            [{ "name": emotion, "probability": round(prob, 4)} for emotion, prob in emotion_probabilities.items()],
            key=lambda x: x["probability"],
            reverse=True
        )

        response_data = {
            "sentiment": overall_sentiment,
            "specificEmotion": specific_emotion,
            "probabilities": probabilities_for_frontend,
            "transcribedText": input_text if 'audio' in request.files else None
        }
        logger.info(f"Sending response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error during emotion analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to analyze emotions in text"}), 500

@predict_bp.route("/api/health", methods=["GET"])
def health_check():
    keras_ready = model2_Class is not None and tokenizer is not None
    return jsonify({
        "status": "healthy", 
        "whisper_loaded": whisper_model is not None, 
        "keras_model_loaded": model2_Class is not None,
        "tokenizer_loaded": tokenizer is not None,
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "tokenizer_path": TOKENIZER_PATH
    }), 200
