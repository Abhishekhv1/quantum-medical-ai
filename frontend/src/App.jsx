import React, { useState } from 'react';
import { Upload, Activity, Cpu, CheckCircle, AlertTriangle, Clock, RefreshCw } from 'lucide-react';
import axios from 'axios';

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResults(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/predict', formData);
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Inference failed. Check backend connection.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8 font-sans">
      <header className="max-w-6xl mx-auto mb-8 border-b border-slate-800 pb-6 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Activity className="text-cyan-400 w-8 h-8" />
            Hybrid Quantum-Classical Medical AI
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Comparative Pneumonia Diagnostic System • 4-Qubit Variational Quantum Circuit vs Classical CNN
          </p>
        </div>
        <span className="px-3 py-1 bg-cyan-950/60 border border-cyan-800/80 rounded-full text-xs font-mono text-cyan-400 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          Simulation: default.qubit
        </span>
      </header>

      <main className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-12 gap-8">
        {/* Upload & Preview Column */}
        <section className="md:col-span-4 bg-slate-900/80 border border-slate-800 rounded-xl p-6 flex flex-col items-center">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4 w-full">
            Chest Radiograph Input
          </h2>

          <label className="w-full aspect-square border-2 border-dashed border-slate-700 hover:border-cyan-500 rounded-lg flex flex-col items-center justify-center cursor-pointer transition relative overflow-hidden bg-slate-950/40">
            {preview ? (
              <img src={preview} alt="Upload preview" className="w-full h-full object-cover" />
            ) : (
              <div className="text-center p-4">
                <Upload className="w-10 h-10 text-slate-500 mx-auto mb-2" />
                <span className="text-xs text-slate-400">Select chest X-Ray (JPEG/PNG)</span>
              </div>
            )}
            <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
          </label>

          <button
            onClick={handleAnalyze}
            disabled={!file || loading}
            className={`w-full mt-5 py-2.5 rounded-lg font-medium text-sm transition flex items-center justify-center gap-2 ${
              !file || loading
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/20'
            }`}
          >
            {loading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Evaluating Circuit States...
              </>
            ) : (
              'Run Dual Diagnosis'
            )}
          </button>

          {error && (
            <div className="mt-4 p-3 bg-rose-950/50 border border-rose-800 rounded text-xs text-rose-300 w-full flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              {error}
            </div>
          )}
        </section>

        {/* Diagnostic Results Column */}
        <section className="md:col-span-8 space-y-6">
          {!results ? (
            <div className="h-full min-h-[350px] border border-slate-800/80 rounded-xl bg-slate-900/40 flex flex-col items-center justify-center text-slate-500">
              <Cpu className="w-12 h-12 stroke-1 mb-2 text-slate-600" />
              <p className="text-sm">Upload an X-Ray to execute side-by-side inference</p>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Classical Card */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
                  <div className="flex justify-between items-start mb-3">
                    <h3 className="font-semibold text-slate-300 text-sm">Classical CNN Baseline</h3>
                    <span className="text-[11px] font-mono text-slate-500 flex items-center gap-1">
                      <Clock className="w-3 h-3" /> {results.classical.latency_ms} ms
                    </span>
                  </div>
                  <div className="text-2xl font-bold mb-1">
                    <span className={results.classical.diagnosis === 'PNEUMONIA' ? 'text-amber-400' : 'text-emerald-400'}>
                      {results.classical.diagnosis}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mb-4">Confidence: {results.classical.confidence}%</p>
                  <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-full ${results.classical.diagnosis === 'PNEUMONIA' ? 'bg-amber-400' : 'bg-emerald-400'}`}
                      style={{ width: `${results.classical.confidence}%` }}
                    ></div>
                  </div>
                </div>

                {/* Quantum Card */}
                <div className="bg-cyan-950/20 border border-cyan-800/50 rounded-xl p-5 shadow-sm relative overflow-hidden">
                  <div className="flex justify-between items-start mb-3">
                    <h3 className="font-semibold text-cyan-300 text-sm flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-cyan-400" /> Hybrid Quantum (VQC)
                    </h3>
                    <span className="text-[11px] font-mono text-cyan-500 flex items-center gap-1">
                      <Clock className="w-3 h-3" /> {results.quantum.latency_ms} ms
                    </span>
                  </div>
                  <div className="text-2xl font-bold mb-1">
                    <span className={results.quantum.diagnosis === 'PNEUMONIA' ? 'text-amber-400' : 'text-emerald-400'}>
                      {results.quantum.diagnosis}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mb-4">Confidence: {results.quantum.confidence}%</p>
                  <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-full ${results.quantum.diagnosis === 'PNEUMONIA' ? 'bg-amber-400' : 'bg-cyan-400'}`}
                      style={{ width: `${results.quantum.confidence}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Quantum Telemetry Section */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-4">
                  4-Qubit Variational Observables & Latent Rotations
                </h4>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {results.quantum.telemetry.pauli_z_expectations.map((expVal, idx) => (
                    <div key={idx} className="bg-slate-950 p-3 rounded-lg border border-slate-800/80">
                      <div className="text-[11px] font-mono text-slate-400 mb-1">Qubit {idx} ⟨Z⟩</div>
                      <div className="text-lg font-bold text-cyan-400 font-mono">{expVal}</div>
                      <div className="text-[10px] text-slate-500 font-mono mt-1">
                        Angle: {results.quantum.telemetry.angles_rad[idx]} rad
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
