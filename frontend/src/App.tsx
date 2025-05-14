import { useState, useRef } from 'react';
import { Mic, Send, Loader2 } from 'lucide-react'; // Added Send and Loader2 icons
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

interface EmotionProbability {
  name: string;
  probability: number;
}

interface PredictionResponse {
  sentiment: string;
  specificEmotion: string;
  probabilities: EmotionProbability[];
  transcribedText?: string; // Optional, for voice input
}

const API_BASE_URL = 'http://localhost:5001'; // Backend API URL

function App() {
  const [textInput, setTextInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false); // For loading state
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [transcribedText, setTranscribedText] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const handleSubmit = async (data: FormData) => {
    setIsLoading(true);
    setError(null);
    setPrediction(null);
    setTranscribedText(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        body: data,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ error: 'Failed to process request. Server returned an error.' }));
        throw new Error(errData.error || `Server error: ${response.status}`);
      }

      const result: PredictionResponse = await response.json();
      setPrediction({
        sentiment: result.sentiment,
        specificEmotion: result.specificEmotion,
        // Ensure probabilities are sorted for consistent display if not already sorted by backend
        probabilities: result.probabilities.sort((a, b) => b.probability - a.probability),
      });
      if (result.transcribedText) {
        setTranscribedText(result.transcribedText);
      }
    } catch (err: any) {
      console.error('API call failed:', err);
      setError(err.message || 'An unexpected error occurred.');
      setPrediction(null);
    }
    setIsLoading(false);
  };

  const handleTextSubmit = async () => {
    if (!textInput.trim()) {
      setError('Please enter some text.');
      return;
    }
    const formData = new FormData();
    formData.append('text', textInput);
    await handleSubmit(formData);
    setTextInput(''); // Clear input after submission
  };

  const handleMicClick = async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      // setIsRecording(false); // Will be set in onstop
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorderRef.current = new MediaRecorder(stream);
        audioChunksRef.current = [];

        mediaRecorderRef.current.ondataavailable = (event) => {
          audioChunksRef.current.push(event.data);
        };

        mediaRecorderRef.current.onstop = async () => {
          setIsRecording(false);
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          audioChunksRef.current = [];
          stream.getTracks().forEach(track => track.stop()); // Stop microphone access
          
          if (audioBlob.size === 0) {
            setError("Recording was empty. Please try again.");
            return;
          }

          const formData = new FormData();
          formData.append('audio', audioBlob, 'recording.webm');
          await handleSubmit(formData);
        };

        mediaRecorderRef.current.start();
        setIsRecording(true);
        setError(null);
        setPrediction(null);
        setTranscribedText(null);
      } catch (err) {
        console.error('Error accessing microphone:', err);
        setError('Could not access microphone. Please ensure permission is granted and try again.');
        setIsRecording(false);
      }
    }
  };

  return (
    <div className="container mx-auto p-4 max-w-3xl font-sans">
      <header className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800">Emotion Analyzer</h1>
        <p className="text-lg text-gray-600">Analyze emotions from text or voice.</p>
      </header>

      <div className="bg-white shadow-xl rounded-lg p-6 mb-8">
        <div className="flex items-center space-x-3 mb-4">
          <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="Enter text here..."
            className="flex-grow p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition-shadow duration-150 shadow-sm hover:shadow-md"
            onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleTextSubmit()}
            disabled={isLoading || isRecording}
          />
          <button
            onClick={handleTextSubmit}
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-5 rounded-lg transition duration-150 ease-in-out shadow-md hover:shadow-lg disabled:opacity-50 flex items-center justify-center"
            disabled={isLoading || isRecording || !textInput.trim()}
            title="Submit text"
          >
            {isLoading && !isRecording ? <Loader2 className="animate-spin mr-2 h-5 w-5" /> : <Send size={20} className="mr-2" />} Submit
          </button>
          <button
            onClick={handleMicClick}
            className={`p-3 rounded-lg transition duration-150 ease-in-out shadow-md hover:shadow-lg disabled:opacity-50 flex items-center justify-center ${isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'} text-white`}
            title={isRecording ? 'Stop recording' : 'Start recording'}
            disabled={isLoading && isRecording}
          >
            {isRecording ? <Loader2 className="animate-spin mr-2 h-5 w-5" /> : <Mic size={24} className="mr-0" />}
            <span className="ml-2">{isRecording ? 'Stop' : 'Record'}</span>
          </button>
        </div>
        {error && <p className="text-red-600 text-sm bg-red-50 p-3 rounded-md">Error: {error}</p>}
      </div>

      {isLoading && (
        <div className="flex justify-center items-center p-6 bg-white shadow-xl rounded-lg mb-8">
          <Loader2 className="animate-spin h-12 w-12 text-indigo-600" />
          <p className="ml-4 text-xl text-gray-700">Analyzing...</p>
        </div>
      )}

      {transcribedText && !isLoading && (
        <div className="bg-white shadow-xl rounded-lg p-6 mb-8">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">Transcribed Text:</h3>
            <p className="text-gray-600 italic">"{transcribedText}"</p>
        </div>
      )}

      {prediction && !isLoading && (
        <div className="bg-white shadow-xl rounded-lg p-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-6 text-center">Prediction Results</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="bg-indigo-50 p-5 rounded-lg shadow">
              <h3 className="text-lg font-medium text-indigo-700 mb-1">Overall Sentiment:</h3>
              <p className={`text-2xl font-bold ${prediction.sentiment === 'Positive' ? 'text-green-600' : prediction.sentiment === 'Negative' ? 'text-red-600' : 'text-gray-600'}`}>{prediction.sentiment}</p>
            </div>
            <div className="bg-purple-50 p-5 rounded-lg shadow">
              <h3 className="text-lg font-medium text-purple-700 mb-1">Specific Emotion:</h3>
              <p className="text-2xl font-bold text-purple-600">{prediction.specificEmotion}</p>
            </div>
          </div>

          <h3 className="text-xl font-semibold text-gray-700 mb-4">Emotion Probabilities (Top 10)</h3>
          <div style={{ width: '100%', height: 400 }}>
            <ResponsiveContainer>
              <BarChart
                data={prediction.probabilities.slice(0, 10)} // Show top 10
                layout="vertical" // Vertical bar chart for better readability of labels
                margin={{
                  top: 5,
                  right: 30,
                  left: 50, // Increased left margin for labels
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 1]} />
                <YAxis dataKey="name" type="category" width={100} interval={0} />
                <Tooltip formatter={(value: number) => value.toFixed(4)} />
                <Legend />
                <Bar dataKey="probability" fill="#8884d8" barSize={20} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

