import React, { useState } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";
import "./App.css";

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

const MEDICAL_TYPES = {
  chest_xray: "🫁 Chest X-Ray",
  brain_mri: "🧠 Brain MRI",
  skin_lesion: "🩺 Skin Lesion",
  retinal: "👁 Retinal",
  cardiac: "🫀 Cardiac",
};

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [medicalType, setMedicalType] = useState("chest_xray");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file || !medicalType) return;

    const formData = new FormData();
    formData.append("image", file);

    setLoading(true);
    try {
      const res = await axios.post(
        `http://localhost:8000/predict/${medicalType}`,
        formData
      );
      setResult(res.data);
    } catch (error) {
      console.error("Prediction error:", error.response || error.message);
      alert("Prediction failed: " + (error.response?.data?.detail || error.message));
    }
    setLoading(false);
  };

  const chartData = result
    ? {
        labels: result.classes,
        datasets: [
          {
            label: "Probability (%)",
            data: result.probabilities.map((p) => (p * 100).toFixed(2)),
            backgroundColor: "#007bff",
            borderRadius: 8,
          },
        ],
      }
    : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
      },
    },
  };

  return (
    <div className="app-container">
      <div className="app-card shadow">
        <h1 className="app-title">🏥 MediNet</h1>
        <p className="subtitle">Your AI-powered medical diagnosis assistant</p>

        <div className="form-group">
          <label className="label">🧪 Select Scan Type</label>
          <select
            value={medicalType}
            onChange={(e) => setMedicalType(e.target.value)}
            className="select-input"
          >
            {Object.entries(MEDICAL_TYPES).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label className="label">📤 Upload Image</label>
          <input type="file" accept="image/*" onChange={handleFileChange} className="file-input" />
          {file && <p className="file-name">📎 {file.name}</p>}
        </div>

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="analyze-btn"
        >
          {loading ? "🔍 Analyzing..." : "🔬 Analyze Image"}
        </button>

        {result && (
          <div className="result-section fade-in">
            <h2>📊 Results</h2>
            <p><strong>Prediction:</strong> <span className="highlight">{result.prediction}</span></p>
            <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>

            <div style={{ marginTop: "20px" }}>
              <Bar data={chartData} options={chartOptions} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
