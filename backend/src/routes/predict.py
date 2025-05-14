import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Blueprint, request, jsonify
import whisper # type: ignore
import tempfile
import logging
import numpy as np # Added for potential Keras/Numpy operations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predict_bp = Blueprint("predict_bp", __name__)

# --- Keras and Tokenizer Setup (USER TO MODIFY) ---
# It's recommended to load your model and tokenizer once when the Flask app starts,
# rather than loading them on each request, for performance reasons.
# You can place the loading logic outside the request handling functions, e.g., here.

# Example: 
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle # Or joblib, for loading the tokenizer

# global keras_model, tokenizer, MAX_SEQ_LENGTH # Declare as global if loaded here
# keras_model = None
# tokenizer = None
# MAX_SEQ_LENGTH = 100 # Replace with your model's expected sequence length

# try:
#     logger.info("Loading Keras model...")
#     # keras_model = load_model("path/to/your/keras_model.h5") # USER: Uncomment and provide path
#     logger.info("Keras model loaded successfully.")
    
#     logger.info("Loading Keras Tokenizer...")
#     # with open("path/to/your/tokenizer.pickle", "rb") as handle:
#     #     tokenizer = pickle.load(handle) # USER: Uncomment and provide path
#     logger.info("Keras Tokenizer loaded successfully.")
    
# except Exception as e:
#     logger.error(f"Error loading Keras model or tokenizer: {e}")
#     # Handle initialization failure, perhaps by disabling the prediction endpoint or using a fallback
#     pass
# --- End Keras and Tokenizer Setup ---


# Load Whisper model (using "base" as requested)
try:
    logger.info("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    whisper_model = None

# Keras model prediction function (USER TO REPLACE/COMPLETE)
def keras_model_predict(text_input):
    logger.info(f"Keras model received text for prediction: {text_input}")
    
    # --- USER: Keras Model Prediction Logic ---
    # This is where you will integrate your actual Keras model and tokenizer.
    # The following is a placeholder and needs to be replaced.

    # 1. Tokenize and Pad Input Text:
    #    - Ensure your tokenizer is loaded (see Keras and Tokenizer Setup section above).
    #    - Tokenize the input_text using your loaded tokenizer.
    #    - Pad the sequences to the MAX_SEQ_LENGTH your model expects.
    # Example (assuming tokenizer and MAX_SEQ_LENGTH are loaded globally):
    # if tokenizer and keras_model:
    #     sequences = tokenizer.texts_to_sequences([text_input])
    #     padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding=\'post\', truncating=\'post\')
    # else:
    #     logger.error("Keras model or tokenizer not loaded. Cannot predict.")
    #     return None # Or raise an exception

    # 2. Make Prediction:
    #    - Use your loaded Keras model to predict probabilities on the padded_sequences.
    # Example:
    #    raw_predictions = keras_model.predict(padded_sequences)
    #    # raw_predictions will likely be a numpy array, e.g., shape (1, 28)

    # 3. Format Output:
    #    - The model should output probabilities for the 28 emotions.
    #    - Create a dictionary mapping each emotion name to its predicted probability.
    emotions_list = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 
        'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    # Example (if raw_predictions is a 1D array/list of 28 probabilities):
    # predicted_probabilities = {emotion: float(prob) for emotion, prob in zip(emotions_list, raw_predictions[0])}

    # --- Placeholder Dummy Logic (REMOVE THIS WHEN YOU ADD YOUR MODEL) ---
    logger.warning("Using placeholder Keras model prediction logic. PLEASE REPLACE.")
    probabilities = {emotion: 0.01 for emotion in emotions_list}
    if "happy" in text_input.lower() or "great" in text_input.lower():
        probabilities['joy'] = 0.6; probabilities['optimism'] = 0.15; probabilities['excitement'] = 0.1
    elif "angry" in text_input.lower() or "hate" in text_input.lower():
        probabilities['anger'] = 0.6; probabilities['annoyance'] = 0.2; probabilities['disapproval'] = 0.1
    elif "sad" in text_input.lower():
        probabilities['sadness'] = 0.6; probabilities['disappointment'] = 0.15; probabilities['grief'] = 0.1
    else:
        probabilities['neutral'] = 0.5; probabilities['curiosity'] = 0.1
    total_prob = sum(probabilities.values())
    if total_prob > 0: predicted_probabilities = {e: p/total_prob for e,p in probabilities.items()}
    else: predicted_probabilities = {e: 1/len(emotions_list) for e in emotions_list}
    # --- End Placeholder Dummy Logic ---

    logger.info(f"Keras model output (probabilities): {predicted_probabilities}")
    return predicted_probabilities
    # --- End USER: Keras Model Prediction Logic ---

# Comprehensive emotion to sentiment mapping
def map_emotion_to_sentiment(specific_emotion):
    sentiment_map = {
        'admiration': 'Positive', 'amusement': 'Positive', 'approval': 'Positive', 'caring': 'Positive',
        'desire': 'Positive', 'excitement': 'Positive', 'gratitude': 'Positive', 'joy': 'Positive',
        'love': 'Positive', 'optimism': 'Positive', 'pride': 'Positive', 'relief': 'Positive', 'surprise': 'Positive',
        'anger': 'Negative', 'annoyance': 'Negative', 'disappointment': 'Negative', 'disapproval': 'Negative',
        'disgust': 'Negative', 'embarrassment': 'Negative', 'fear': 'Negative', 'grief': 'Negative',
        'nervousness': 'Negative', 'remorse': 'Negative', 'sadness': 'Negative',
        'confusion': 'Neutral', 'curiosity': 'Neutral', 'realization': 'Neutral', 'neutral': 'Neutral' 
    }
    sentiment = sentiment_map.get(specific_emotion, 'Neutral')
    logger.info(f"Mapped specific emotion \'{specific_emotion}\' to sentiment \'{sentiment}\"")
    return sentiment

@predict_bp.route("/api/predict", methods=["POST"])
def predict_emotion():
    logger.info("Received request at /api/predict")
    input_text = None

    # USER: Check if Keras model and tokenizer are loaded if you load them globally
    # if not keras_model or not tokenizer:
    #     logger.error("Keras model/tokenizer not available. Prediction endpoint is offline.")
    #     return jsonify({"error": "Emotion prediction service is currently unavailable. Please try again later."}), 503

    if 'audio' in request.files:
        logger.info("Audio file detected in request.")
        audio_file = request.files['audio']
        if not whisper_model:
            logger.error("Whisper model not loaded, cannot process audio.")
            return jsonify({"error": "Speech processing service is unavailable"}), 500
        tmp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_audio_file:
                audio_file.save(tmp_audio_file.name)
                tmp_audio_path = tmp_audio_file.name
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

    # Get emotion probabilities from Keras model (or placeholder)
    emotion_probabilities = keras_model_predict(input_text)

    if not emotion_probabilities:
        logger.error("Emotion probabilities are empty or null from Keras model function.")
        return jsonify({"error": "Failed to get emotion predictions from model"}), 500
        
    specific_emotion = max(emotion_probabilities, key=emotion_probabilities.get)
    logger.info(f"Predicted specific emotion: {specific_emotion}")

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

@predict_bp.route("/api/health", methods=["GET"])
def health_check():
    # Extended health check could also verify Keras model/tokenizer status
    # keras_ready = keras_model is not None and tokenizer is not None
    # return jsonify({"status": "healthy", "whisper_loaded": whisper_model is not None, "keras_ready": keras_ready}), 200
    return jsonify({"status": "healthy", "whisper_loaded": whisper_model is not None}), 200


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Blueprint, request, jsonify
import whisper # type: ignore
import tempfile
import logging
import numpy as np
import pickle
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predict_bp = Blueprint("predict_bp", __name__)

# --- Keras and Tokenizer Setup ---
keras_model = None
tokenizer = None
MAX_SEQ_LENGTH = 50  # As specified by the user
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "model2_class.keras")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "tokenizer.pickle")

EMOTIONS_LIST = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire',
    'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

try:
    logger.info(f"Loading Keras model from {MODEL_PATH}...")
    if os.path.exists(MODEL_PATH):
        keras_model = load_model(MODEL_PATH)
        logger.info("Keras model loaded successfully.")
    else:
        logger.error(f"Keras model file not found at {MODEL_PATH}")

    logger.info(f"Loading Keras Tokenizer from {TOKENIZER_PATH}...")
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, "rb") as handle:
            tokenizer = pickle.load(handle)
        logger.info("Keras Tokenizer loaded successfully.")
    else:
        logger.error(f"Keras Tokenizer file not found at {TOKENIZER_PATH}")

except Exception as e:
    logger.error(f"Error loading Keras model or tokenizer: {e}")
    # Handle initialization failure, perhaps by disabling the prediction endpoint or using a fallback
    pass
# --- End Keras and Tokenizer Setup ---


# Load Whisper model (using "base" as requested)
try:
    logger.info("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    whisper_model = None

# Keras model prediction function
def keras_model_predict(text_input):
    logger.info(f"Keras model received text for prediction: {text_input}")

    if not keras_model or not tokenizer:
        logger.error("Keras model or tokenizer not loaded. Cannot predict.")
        # Return a default probability distribution or raise an error
        # For now, returning None to be handled by the caller
        return None

    try:
        # 1. Tokenize and Pad Input Text:
        sequences = tokenizer.texts_to_sequences([text_input])
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')

        # 2. Make Prediction:
        raw_predictions = keras_model.predict(padded_sequences)
        # Assuming raw_predictions is a numpy array of shape (1, num_emotions)

        # 3. Format Output:
        if raw_predictions is not None and len(raw_predictions) > 0:
            # Ensure probabilities sum to 1 if they don't already (e.g. if model outputs logits)
            # For softmax output, this might not be strictly necessary but good for consistency
            # prediction_values = raw_predictions[0]
            # if not np.isclose(np.sum(prediction_values), 1.0):
            #     prediction_values = np.exp(prediction_values) / np.sum(np.exp(prediction_values)) # Softmax if logits
            
            predicted_probabilities = {emotion: float(prob) for emotion, prob in zip(EMOTIONS_LIST, raw_predictions[0])}
        else:
            logger.error("Keras model returned empty or invalid predictions.")
            return None

        logger.info(f"Keras model output (probabilities): {predicted_probabilities}")
        return predicted_probabilities

    except Exception as e:
        logger.error(f"Error during Keras model prediction: {e}")
        return None

# Comprehensive emotion to sentiment mapping
def map_emotion_to_sentiment(specific_emotion):
    sentiment_map = {
        'admiration': 'Positive', 'amusement': 'Positive', 'approval': 'Positive', 'caring': 'Positive',
        'desire': 'Positive', 'excitement': 'Positive', 'gratitude': 'Positive', 'joy': 'Positive',
        'love': 'Positive', 'optimism': 'Positive', 'pride': 'Positive', 'relief': 'Positive', 'surprise': 'Positive',
        'anger': 'Negative', 'annoyance': 'Negative', 'disappointment': 'Negative', 'disapproval': 'Negative',
        'disgust': 'Negative', 'embarrassment': 'Negative', 'fear': 'Negative', 'grief': 'Negative',
        'nervousness': 'Negative', 'remorse': 'Negative', 'sadness': 'Negative',
        'confusion': 'Neutral', 'curiosity': 'Neutral', 'realization': 'Neutral', 'neutral': 'Neutral'
    }
    sentiment = sentiment_map.get(specific_emotion, 'Neutral')
    logger.info(f"Mapped specific emotion '{specific_emotion}' to sentiment '{sentiment}'")
    return sentiment

@predict_bp.route("/api/predict", methods=["POST"])
def predict_emotion():
    logger.info("Received request at /api/predict")
    input_text = None

    if not keras_model or not tokenizer:
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
            # Use a more robust way to get a temporary file name if needed
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

    emotion_probabilities = keras_model_predict(input_text)

    if not emotion_probabilities:
        logger.error("Emotion probabilities are empty or null from Keras model function.")
        return jsonify({"error": "Failed to get emotion predictions from model"}), 500

    specific_emotion = max(emotion_probabilities, key=emotion_probabilities.get)
    logger.info(f"Predicted specific emotion: {specific_emotion}")

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

@predict_bp.route("/api/health", methods=["GET"])
def health_check():
    keras_ready = keras_model is not None and tokenizer is not None
    return jsonify({
        "status": "healthy", 
        "whisper_loaded": whisper_model is not None, 
        "keras_model_loaded": keras_model is not None,
        "tokenizer_loaded": tokenizer is not None
    }), 200

