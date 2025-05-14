# Emotion Analyzer Web Application

This application allows users to input text or voice to get an analysis of the emotion conveyed. It predicts an overall sentiment (Positive, Neutral, Negative) and a specific emotion from a list of 28 predefined emotions. It also displays the probability distribution of these specific emotions.

## Project Structure

```
emotion_analyzer_app/
├── backend/        # Flask backend application
│   ├── src/
│   │   ├── models/ # (Currently unused, for potential database models)
│   │   ├── routes/
│   │   │   └── predict.py  # API endpoint for predictions, Keras model placeholder
│   │   ├── static/ # (Currently unused, for general static files if needed)
│   │   └── main.py     # Flask app entry point
│   ├── venv/         # Python virtual environment
│   └── requirements.txt # Python dependencies
├── frontend/       # React frontend application
│   ├── dist/         # Production build of the frontend
│   ├── public/       # Static assets for frontend
│   ├── src/
│   │   ├── App.tsx     # Main React component
│   │   └── App.css     # Styles for the application
│   ├── index.html    # Entry HTML file for React app
│   ├── package.json  # Frontend dependencies and scripts
│   ├── pnpm-lock.yaml # Lockfile for pnpm
│   └── tsconfig.json # TypeScript configuration
└── todo.md           # Task checklist for development
```

## Backend Setup (Flask)

1.  **Navigate to the backend directory:**
    ```bash
    cd emotion_analyzer_app/backend
    ```

2.  **Create and activate a Python virtual environment:**
    If you don't have `venv` installed, you might need to install it first (e.g., `sudo apt-get install python3-venv` on Debian/Ubuntu).
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    On Windows, activation is `venv\Scripts\activate`.

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install Flask, Whisper, PyTorch, and other necessary libraries. Note that PyTorch and Whisper can be large downloads.

4.  **Integrate Your Keras Model:**
    *   Open `backend/src/routes/predict.py`.
    *   Locate the section marked `--- Keras and Tokenizer Setup (USER TO MODIFY) ---` and `def keras_model_predict(text_input):`.
    *   Follow the instructions in the comments to load your trained Keras model (e.g., `.h5` file) and your tokenizer (e.g., a pickled `Tokenizer` object).
    *   Update the `keras_model_predict` function to use your loaded model and tokenizer to process the `text_input` and return a dictionary of emotion probabilities.
    *   Ensure `MAX_SEQ_LENGTH` is set correctly if you load the model globally.

5.  **Run the backend server:**
    ```bash
    python src/main.py
    ```
    The backend will start, by default, on `http://localhost:5001`.
    The first time it runs, the Whisper model (`base` version) will be downloaded, which might take a few minutes.

## Frontend Setup (React)

1.  **Navigate to the frontend directory:**
    ```bash
    cd emotion_analyzer_app/frontend
    ```

2.  **Install dependencies using pnpm (or npm/yarn):**
    The project was set up with `pnpm`. If you don't have `pnpm`, you can install it (`npm install -g pnpm`) or adapt to `npm` or `yarn`.
    ```bash
    pnpm install 
    # OR if using npm: npm install
    # OR if using yarn: yarn install
    ```

3.  **Run the frontend development server:**
    ```bash
    pnpm run dev
    # OR if using npm: npm run dev
    # OR if using yarn: yarn dev
    ```
    The frontend will start, by default, on `http://localhost:5173` (or another port if 5173 is busy) and will connect to the backend API running on port 5001.

4.  **Access the application:**
    Open your browser and go to the URL provided by the frontend development server (e.g., `http://localhost:5173`).

## Production Build

The frontend has already been built into the `frontend/dist` directory. The Flask backend is configured in `backend/src/main.py` to serve these static files when not in debug mode or when a direct path to an API endpoint is not hit. For a more robust production deployment, consider using a dedicated WSGI server for Flask (like Gunicorn or uWSGI) and a reverse proxy (like Nginx) to serve both the frontend and backend under a single domain.

## Emotion Mapping

The mapping from the 28 specific emotions to Positive/Neutral/Negative sentiment is defined in `backend/src/routes/predict.py` within the `map_emotion_to_sentiment` function. You can adjust this mapping if needed.

## Voice Input

Voice input requires microphone access. Ensure your browser has permission to access the microphone when prompted.

