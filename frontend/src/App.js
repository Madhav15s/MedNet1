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

// ✅ Medical scan info for each type
const MEDICAL_INFO = {
  chest_xray:
    "Chest X-rays help detect lung conditions like pneumonia, tuberculosis, and lung cancer. AI can assist radiologists in spotting abnormalities faster.",
  brain_mri:
    "Brain MRIs are used to identify tumors, bleeding, nerve damage, and neurological conditions such as stroke or multiple sclerosis.",
  skin_lesion:
    "Skin lesion scans help in early detection of skin cancer types like melanoma. AI can aid in classifying lesion severity.",
  retinal:
    "Retinal imaging is key for diagnosing diabetic retinopathy, glaucoma, and age-related macular degeneration (AMD).",
  cardiac:
    "Cardiac scans like echocardiograms or MRIs evaluate heart size, structure, and function, helping diagnose heart diseases.",
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

    <p><strong>🧠 AI Prediction:</strong> <span className="highlight">{result.prediction}</span></p>
    <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>

    {/* ✅ Consultation Advice */}
    <p className="consult-message">
      Based on the scan, the AI model predicts <strong>{result.prediction}</strong>. 
      <br />
      <span style={{ color: "#e74c3c", fontWeight: "600" }}>
        Please consult a certified medical professional for further diagnosis and appropriate treatment.
      </span>
    </p>

    <div style={{ marginTop: "20px" }}>
      <Bar data={chartData} options={chartOptions} />
    </div>

    {/* ✅ Additional Scan Info */}
    <div style={{ marginTop: "25px", fontSize: "0.95rem", color: "#333", lineHeight: "1.5" }}>
      <strong>ℹ️ About this Scan:</strong>
      <p style={{ marginTop: "0.5rem" }}>{MEDICAL_INFO[medicalType]}</p>
    </div>
  </div>
)}

      </div>
    </div>
  );
}

export default App;
