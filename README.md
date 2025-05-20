# Emotion Analyzer Web Application

This application analyzes emotions in text input, predicting both overall sentiment (Positive, Neutral, Negative) and specific emotions from a set of 27 predefined emotions.

## Repository Structure

```
emotion_model/
├── backend/        # Flask backend application
│   ├── src/
│   │   ├── ml_models/  # Directory containing model and tokenizer files
│   │   │   ├── model2_class.keras  # The emotion classification model
│   │   │   └── tokenizer.pickle    # The tokenizer used for text preprocessing
│   │   ├── routes/
│   │   │   ├── predict.py  # API endpoint for predictions
│   │   │   └── user.py     # User-related endpoints
│   │   └── main.py     # Flask app entry point
│   ├── venv/         # Python virtual environment
│   └── requirements.txt # Python dependencies
└── frontend/       # React frontend application
```

## Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd emotion_model/backend
   ```

2. **Create and activate a Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   On Windows, activation is `venv\Scripts\activate`.

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the backend server:**
   ```bash
   python src/main.py
   ```
   The backend will start on `http://localhost:5001`.

## Changing Model and Tokenizer

To update the model or tokenizer:

1. The model and tokenizer files are located in:
   ```
   backend/src/ml_models/
   ```

2. Replace these files with your new versions:
   - `model2_class.keras` - The emotion classification model
   - `tokenizer.pickle` - The tokenizer for text preprocessing

3. To ensure consistency between your notebook and web app:
   ```python
   # In your notebook, save the model and tokenizer with these exact commands:
   model2_Class.save('model2_class.keras')
   
   import pickle
   with open('tokenizer.pickle', 'wb') as handle:
       pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
   ```

4. Copy these files to the `backend/src/ml_models/` directory

5. Restart the backend server for changes to take effect
