import { useState, useCallback, useRef } from "react";

interface DetectionResult {
  status: string;
  timestamp: string;
  verdict: "REAL" | "FAKE" | "UNCERTAIN";
  confidence: number;
  confidence_percent: number;
  filename: string;
  model_scores: {
    neural_network_fake_prob: number;
    neural_network_real_prob: number;
    fft_anomaly_score: number;
    metadata_suspicious: boolean;
    ensemble_score: number;
  };
  artifacts_detected: string[];
  manipulation_type: {
    type: string;
    description: string;
  };
  explanation: string;
  face_detection: {
    faces_found: number;
    face_regions: { x: number; y: number; w: number; h: number }[];
  };
  frequency_analysis: {
    fft_anomaly_score: number;
    spectral_peak_ratio: number;
    has_gan_grid: boolean;
    peak_pixel_count: number;
    fft_spectrum_image: string; // base64
  };
  metadata_analysis: {
    file_hash_sha256: string;
    file_size_kb: number;
    image_dimensions: string;
    aspect_ratio: number;
    exif_fields_found: number;
    ai_tool_detected: string | null;
    metadata_flags: string[];
    metadata_suspicious: boolean;
  };
  exif_data: Record<string, string>;
  gradcam_heatmap: string; // base64
  legal: {
    file_hash_sha256: string;
    analysis_timestamp: string;
    model_version: string;
    tamper_proof_note: string;
  };
  whatsapp_summary: string;
}

const API_BASE = "http://localhost:8000";

const verdictConfig = {
  FAKE: {
    bg: "bg-rose-50 border-rose-100",
    border: "border-rose-200",
    text: "text-rose-600",
    icon: "⚠️",
    label: "FAKE",
  },
  REAL: {
    bg: "bg-emerald-50 border-emerald-100",
    border: "border-emerald-200",
    text: "text-emerald-600",
    icon: "✅",
    label: "REAL",
  },
  UNCERTAIN: {
    bg: "bg-amber-50 border-amber-100",
    border: "border-amber-200",
    text: "text-amber-600",
    icon: "⚠️",
    label: "UNCERTAIN",
  },
};

function ConfidenceBar({
  value,
  color,
}: {
  value: number;
  color: string;
}) {
  return (
    <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-700 ${color}`}
        style={{ width: `${Math.round(value * 100)}%` }}
      />
    </div>
  );
}

function ScoreRow({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-slate-600 font-medium">{label}</span>
        <span className="font-mono text-slate-800 font-bold">
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <ConfidenceBar value={value} color={color} />
    </div>
  );
}

export default function ImageDetectorTab() {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<
    "overview" | "heatmap" | "fft" | "metadata" | "legal"
  >("overview");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file (JPEG, PNG, WEBP, BMP).");
      return;
    }
    setFile(f);
    setResult(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(f));
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/analyze/image`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(
          errData?.detail?.message ||
            errData?.detail ||
            `Server error: ${response.status}`
        );
      }

      const data: DetectionResult = await response.json();
      setResult(data);
      setActiveTab("overview");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const downloadPDF = async () => {
    if (!result) return;
    try {
      const response = await fetch(`${API_BASE}/analyze/image/report`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ result }),
      });
      
      if (!response.ok) throw new Error("Failed to generate PDF");
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `DeepShield_Report_${result.legal.file_hash_sha256.substring(0, 8)}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error("PDF Download failed:", err);
      alert("Failed to generate PDF. Please make sure the backend is responding.");
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-6 py-8 space-y-6 bg-white rounded-3xl shadow-xl text-slate-800">
      {!result && (
        <div
          className={`
            relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer
            transition-all duration-200
            ${isDragging
              ? "border-blue-400 bg-blue-50 scale-[1.01]"
              : "border-slate-300 hover:border-blue-300 bg-slate-50 hover:bg-blue-50/50"
            }
          `}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={onFileChange}
          />

          {previewUrl ? (
            <div className="space-y-4">
              <img
                src={previewUrl}
                alt="Preview"
                className="max-h-64 mx-auto rounded-xl object-contain shadow-md"
              />
              <p className="text-slate-500 text-sm font-medium">
                {file?.name} · {((file?.size ?? 0) / 1024).toFixed(1)} KB
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="text-5xl text-blue-500">🖼️</div>
              <p className="text-slate-700 font-semibold text-lg">
                Drop an image here or click to upload
              </p>
              <p className="text-slate-500 text-sm">
                JPEG, PNG, WEBP, BMP · Max 20MB
              </p>
            </div>
          )}
        </div>
      )}

      {file && !result && (
        <div className="flex gap-3">
          <button
            onClick={analyze}
            disabled={loading}
            className="
              flex-1 py-3 px-6 rounded-xl font-bold text-white
              bg-blue-500 hover:bg-blue-600 disabled:opacity-50
              disabled:cursor-not-allowed transition-all duration-200
              flex items-center justify-center gap-2 shadow-md
            "
          >
            {loading ? (
              <>
                <span className="animate-spin text-lg">⚙️</span>
                Analyzing...
              </>
            ) : (
              <>🔍 Analyze Image</>
            )}
          </button>
          <button
            onClick={reset}
            className="
              px-5 py-3 rounded-xl font-medium text-slate-600
              border-2 border-slate-200 hover:border-slate-300
              hover:bg-slate-50 transition-all duration-200
            "
          >
            Clear
          </button>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-600 text-sm font-medium">
          ⚠️ {error}
        </div>
      )}

      {result && (
        <div className="space-y-6">

          {(() => {
            const cfg = verdictConfig[result.verdict];
            return (
              <div
                className={`
                  ${cfg.bg} border-2 ${cfg.border}
                  rounded-2xl p-6 flex flex-col sm:flex-row items-center justify-between gap-4 shadow-sm
                `}
              >
                <div className="flex flex-col items-center sm:items-start">
                  <div className="flex items-center gap-3 mb-1">
                    <span className="text-3xl">{cfg.icon}</span>
                    <span className={`text-4xl font-extrabold tracking-tight ${cfg.text}`}>
                      {cfg.label}
                    </span>
                  </div>
                  <p className="text-slate-600 text-sm font-medium text-center sm:text-left">
                    Confidence: {result.confidence_percent}% ·{" "}
                    {result.manipulation_type.type}
                  </p>
                </div>
                <div className="flex flex-col items-center sm:items-end">
                  <div className="text-slate-500 text-xs font-bold tracking-wider mb-1">ENSEMBLE SCORE</div>
                  <div className={`text-5xl font-black font-mono tracking-tighter ${cfg.text}`}>
                    {result.confidence_percent}%
                  </div>
                </div>
              </div>
            );
          })()}

          <div className="flex flex-col sm:flex-row justify-between items-center bg-blue-50 border border-blue-100 rounded-xl p-4 gap-4 shadow-sm">
            <div className="text-blue-800 text-sm font-semibold flex items-center gap-2">
              <span>🛡️</span> Get a detailed forensic record of this analysis
            </div>
            <button
              onClick={downloadPDF}
              className="w-full sm:w-auto py-2.5 px-6 shrink-0 bg-blue-500 hover:bg-blue-600 text-white font-bold rounded-xl shadow-md transition-all duration-200 flex items-center justify-center gap-2"
            >
              <span>📄</span> Download PDF Report
            </button>
          </div>

          <div className="flex gap-2 bg-slate-100 rounded-xl p-1.5 overflow-x-auto shadow-inner">
            {(["overview", "heatmap", "fft", "metadata", "legal"] as const).map(
              (tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`
                    flex-1 min-w-max py-2.5 px-4 rounded-lg text-sm font-bold capitalize
                    transition-all duration-200
                    ${activeTab === tab
                      ? "bg-white text-blue-600 shadow-sm ring-1 ring-slate-200/50"
                      : "text-slate-500 hover:text-slate-700 hover:bg-slate-200/50"
                    }
                  `}
                >
                  {tab === "heatmap" ? "GradCAM" :
                   tab === "fft"     ? "FFT Spectrum" :
                   tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              )
            )}
          </div>

          {activeTab === "overview" && (
            <div className="space-y-4">

              <div className="bg-white border border-slate-100 shadow-sm rounded-xl p-6">
                <h3 className="text-slate-800 font-bold mb-5 flex items-center gap-2">
                  <span className="text-blue-500">📊</span> Multi-Layer Analysis Breakdown
                </h3>
                <ScoreRow
                  label="Neural Network (Vision Transformer)"
                  value={result.model_scores.neural_network_fake_prob}
                  color="bg-rose-500"
                />
                <ScoreRow
                  label="FFT Frequency Anomaly"
                  value={result.model_scores.fft_anomaly_score}
                  color="bg-indigo-500"
                />
                <ScoreRow
                  label="Ensemble Score (Final)"
                  value={result.model_scores.ensemble_score}
                  color={
                    result.verdict === "FAKE"
                      ? "bg-rose-500"
                      : result.verdict === "UNCERTAIN"
                      ? "bg-amber-500"
                      : "bg-emerald-500"
                  }
                />
              </div>

              <div className="bg-white border border-slate-100 shadow-sm rounded-xl p-6">
                <h3 className="text-slate-800 font-bold mb-3 flex items-center gap-2">
                  <span className="text-indigo-500">🧬</span> Manipulation Classification
                </h3>
                <div className="inline-flex bg-indigo-50 border border-indigo-200 text-indigo-700 text-sm font-bold px-3 py-1.5 rounded-full mb-3 shadow-sm">
                  {result.manipulation_type.type}
                </div>
                <p className="text-slate-600 text-sm leading-relaxed font-medium">
                  {result.manipulation_type.description}
                </p>
              </div>

              <div className="bg-white border border-slate-100 shadow-sm rounded-xl p-6">
                <h3 className="text-slate-800 font-bold mb-4 flex items-center gap-2">
                  <span className="text-rose-500">⚠️</span> Detected Anomalies
                </h3>
                <ul className="space-y-3">
                  {result.artifacts_detected.map((a, i) => (
                    <li key={i} className="flex items-start gap-3 text-sm bg-slate-50 border border-slate-100 p-3 rounded-lg">
                      <span className="text-rose-500 mt-0.5 shrink-0">●</span>
                      <span className="text-slate-700 font-medium">{a}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="bg-white border border-slate-100 shadow-sm rounded-xl p-6">
                <h3 className="text-slate-800 font-bold mb-3 flex items-center gap-2">
                  <span className="text-emerald-500">📝</span> Analysis Summary
                </h3>
                <p className="text-slate-600 text-sm leading-relaxed font-medium bg-slate-50 p-4 rounded-lg border border-slate-100">
                  {result.explanation}
                </p>
              </div>

              <div className="bg-white border border-slate-100 shadow-sm rounded-xl p-6">
                <h3 className="text-slate-800 font-bold mb-2 flex items-center gap-2">
                  <span className="text-blue-400">👤</span> Face Detection
                </h3>
                <p className="text-slate-600 text-sm font-medium">
                  {result.face_detection.faces_found === 0
                    ? "No faces detected in this image."
                    : `${result.face_detection.faces_found} face(s) detected and analyzed.`
                  }
                </p>
              </div>

              <button
                onClick={reset}
                className="
                  w-full py-4 rounded-xl font-bold text-slate-500 bg-slate-50
                  border-2 border-slate-200 hover:border-slate-300
                  hover:text-slate-700 hover:bg-slate-100 transition-all duration-200 shadow-sm
                "
              >
                + Analyze Another Image
              </button>
            </div>
          )}

          {activeTab === "heatmap" && (
            <div className="bg-white border border-slate-100 shadow-sm rounded-xl p-6 space-y-4">
              <h3 className="text-slate-800 font-bold flex items-center gap-2">
                <span className="text-orange-500">🔥</span> GradCAM Explanation Heatmap
              </h3>
              <p className="text-slate-600 text-sm font-medium">
                Highlighted regions show which areas of the image most
                influenced the FAKE classification. <span className="text-rose-500 font-bold">Red</span> = high influence.
              </p>
              {result.gradcam_heatmap ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mt-4">
                  <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
                    <p className="text-slate-500 text-xs font-bold mb-3 uppercase tracking-wider text-center">
                      Original
                    </p>
                    {previewUrl && (
                      <img
                        src={previewUrl}
                        alt="Original"
                        className="rounded-xl w-full object-contain max-h-64 shadow-sm"
                      />
                    )}
                  </div>
                  <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
                    <p className="text-slate-500 text-xs font-bold mb-3 uppercase tracking-wider text-center">
                      GradCAM Overlay
                    </p>
                    <img
                      src={`data:image/png;base64,${result.gradcam_heatmap}`}
                      alt="GradCAM"
                      className="rounded-xl w-full object-contain max-h-64 shadow-sm"
                    />
                  </div>
                </div>
              ) : (
                <div className="text-slate-500 font-medium text-sm text-center py-8 bg-slate-50 rounded-xl border border-slate-100 border-dashed">
                  GradCAM not generated (low confidence or error).
                </div>
              )}
            </div>
          )}

          {activeTab === "fft" && (
            <div className="bg-white border border-slate-100 shadow-sm rounded-xl p-6 space-y-4">
              <h3 className="text-slate-800 font-bold flex items-center gap-2">
                <span className="text-purple-500">🌌</span> Frequency Domain Analysis (FFT)
              </h3>
              <p className="text-slate-600 text-sm font-medium bg-blue-50 p-4 rounded-lg border border-blue-100">
                GAN-generated images leave characteristic periodic patterns in
                the frequency domain. Bright spots away from center = GAN artifacts.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 text-sm">
                <div className="bg-slate-50 rounded-xl p-5 border border-slate-200 shadow-sm space-y-4">
                  <div className="flex justify-between items-center bg-white p-3 rounded-lg border border-slate-100">
                    <span className="text-slate-600 font-bold">Anomaly Score</span>
                    <span className="font-mono text-slate-800 font-bold text-lg">
                      {(result.frequency_analysis.fft_anomaly_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center bg-white p-3 rounded-lg border border-slate-100">
                    <span className="text-slate-600 font-bold">GAN Grid Pattern</span>
                    <span className={result.frequency_analysis.has_gan_grid
                      ? "text-rose-500 font-bold px-2 py-1 bg-rose-50 rounded"
                      : "text-emerald-500 font-bold px-2 py-1 bg-emerald-50 rounded"
                    }>
                      {result.frequency_analysis.has_gan_grid ? "DETECTED" : "Not found"}
                    </span>
                  </div>
                  <div className="flex justify-between items-center bg-white p-3 rounded-lg border border-slate-100">
                    <span className="text-slate-600 font-bold">Peak Pixel Count</span>
                    <span className="font-mono text-slate-800 font-bold">
                      {result.frequency_analysis.peak_pixel_count}
                    </span>
                  </div>
                  <div className="flex justify-between items-center bg-white p-3 rounded-lg border border-slate-100">
                    <span className="text-slate-600 font-bold">Spectral Peak Ratio</span>
                    <span className="font-mono text-slate-800 font-bold">
                      {result.frequency_analysis.spectral_peak_ratio.toFixed(4)}
                    </span>
                  </div>
                </div>
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col items-center justify-center">
                  {result.frequency_analysis.fft_spectrum_image ? (
                    <img
                      src={`data:image/png;base64,${result.frequency_analysis.fft_spectrum_image}`}
                      alt="FFT Spectrum"
                      className="rounded-xl w-full shadow-sm"
                    />
                  ) : (
                    <div className="text-slate-500 font-medium text-sm text-center py-4">
                      FFT spectrum unavailable
                    </div>
                  )}
                  <p className="text-slate-500 font-bold uppercase tracking-wider text-xs mt-4 text-center">
                    FFT Magnitude Spectrum
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeTab === "metadata" && (
            <div className="bg-white border border-slate-100 shadow-sm rounded-xl p-6 space-y-6">
              <h3 className="text-slate-800 font-bold flex items-center gap-2">
                <span className="text-teal-500">📎</span> Metadata & EXIF Analysis
              </h3>

              {result.metadata_analysis.metadata_suspicious && (
                <div className="bg-rose-50 border border-rose-200 rounded-xl p-4 text-sm font-bold text-rose-600 shadow-sm">
                  ⚠️ Suspicious metadata detected
                </div>
              )}

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                {[
                  ["File Size", `${result.metadata_analysis.file_size_kb} KB`],
                  ["Dimensions", result.metadata_analysis.image_dimensions],
                  ["Aspect Ratio", result.metadata_analysis.aspect_ratio],
                  ["EXIF Fields", result.metadata_analysis.exif_fields_found],
                  ["AI Tool", result.metadata_analysis.ai_tool_detected ?? "None detected"],
                ].map(([k, v]) => (
                  <div key={String(k)} className="bg-slate-50 border border-slate-200 shadow-sm rounded-xl p-4">
                    <div className="text-slate-500 text-xs font-bold uppercase tracking-wider mb-2">
                      {k}
                    </div>
                    <div className="text-slate-800 font-mono font-bold">{String(v)}</div>
                  </div>
                ))}
              </div>

              {result.metadata_analysis.metadata_flags.length > 0 && (
                <div className="bg-amber-50/50 p-4 rounded-xl border border-amber-100">
                  <h4 className="text-slate-800 font-bold mb-3 flex items-center gap-2 text-sm">
                    <span className="text-amber-500">🚩</span> Metadata Flags
                  </h4>
                  <ul className="space-y-2">
                    {result.metadata_analysis.metadata_flags.map((f, i) => (
                      <li key={i} className="flex items-start gap-3 text-sm bg-white p-2 rounded-lg border border-amber-100 shadow-sm">
                        <span className="text-amber-500 shrink-0 mt-0.5">●</span>
                        <span className="text-slate-700 font-medium">{f}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {Object.keys(result.exif_data).length > 0 && (
                <div>
                  <h4 className="text-slate-800 font-bold mb-3 flex items-center gap-2 text-sm">
                    <span className="text-blue-500">🏷️</span> Raw EXIF Data
                  </h4>
                  <div className="bg-slate-900 rounded-xl p-4 space-y-1.5 max-h-64 overflow-y-auto shadow-inner">
                    {Object.entries(result.exif_data).map(([k, v]) => (
                      <div key={k} className="flex gap-4 text-xs border-b border-white/5 pb-1 last:border-0 last:pb-0">
                        <span className="text-blue-300 font-mono font-semibold w-40 shrink-0">
                          {k}
                        </span>
                        <span className="text-slate-300 truncate">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === "legal" && (
            <div className="bg-white border border-slate-100 shadow-sm rounded-xl p-6 space-y-5">
              <h3 className="text-slate-800 font-bold flex items-center gap-2">
                <span className="text-emerald-600">🔒</span> Tamper-Proof Legal Report
              </h3>
              <p className="text-slate-600 text-sm font-medium bg-slate-50 p-4 rounded-lg border border-slate-100">
                A SHA-256 cryptographic hash was generated from the raw file bytes
                at the moment of upload. This can be used to verify the report
                against the original file in legal or forensic proceedings.
              </p>

              <div className="bg-slate-50 border border-slate-200 shadow-sm rounded-xl p-5 space-y-4">
                <div className="bg-white p-3 rounded-lg border border-slate-100">
                  <div className="text-slate-500 text-xs font-bold uppercase tracking-wider mb-1">
                    File Hash (SHA-256)
                  </div>
                  <div className="font-mono font-bold text-xs text-blue-600 break-all select-all">
                    {result.legal.file_hash_sha256}
                  </div>
                </div>
                <div className="bg-white p-3 rounded-lg border border-slate-100">
                  <div className="text-slate-500 text-xs font-bold uppercase tracking-wider mb-1">
                    Analysis Timestamp (UTC)
                  </div>
                  <div className="font-mono font-bold text-sm text-slate-800">
                    {result.legal.analysis_timestamp}
                  </div>
                </div>
                <div className="bg-white p-3 rounded-lg border border-slate-100">
                  <div className="text-slate-500 text-xs font-bold uppercase tracking-wider mb-1">
                    Model Version
                  </div>
                  <div className="font-mono font-bold text-sm text-slate-800">
                    {result.legal.model_version}
                  </div>
                </div>
                <div className="bg-white p-4 rounded-lg border border-slate-100 flex justify-between items-center">
                  <div className="text-slate-500 text-xs font-bold uppercase tracking-wider">
                    Verdict on Record
                  </div>
                  <div
                    className={`font-mono text-sm font-bold px-3 py-1.5 rounded-lg ${
                      result.verdict === "FAKE"
                        ? "bg-rose-100 text-rose-700"
                        : result.verdict === "REAL"
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-amber-100 text-amber-700"
                    }`}
                  >
                    {result.verdict} — {result.confidence_percent}% confidence
                  </div>
                </div>
              </div>

              <p className="text-slate-500 text-xs font-medium text-center italic">
                {result.legal.tamper_proof_note}
              </p>

              <button
                onClick={downloadPDF}
                className="w-full mt-4 py-3.5 px-4 bg-blue-500 hover:bg-blue-600 text-white font-bold rounded-xl shadow-md transition-all duration-200 flex items-center justify-center gap-2"
              >
                <span>📄</span> Download Certified PDF Report
              </button>

              <div className="mt-8 border-t border-slate-100 pt-6">
                <h4 className="text-slate-800 font-bold mb-3 flex items-center gap-2 text-sm">
                  <span className="text-green-500">💬</span> WhatsApp Report Summary
                </h4>
                <div className="relative group">
                  <pre className="bg-slate-900 shadow-inner rounded-xl p-5 text-xs text-slate-200 whitespace-pre-wrap font-mono leading-relaxed">
                    {result.whatsapp_summary}
                  </pre>
                  <button
                    onClick={() =>
                      navigator.clipboard.writeText(result.whatsapp_summary)
                    }
                    className="
                      absolute top-3 right-3 px-3 py-1.5 text-xs font-bold rounded-lg
                      bg-white/10 hover:bg-white/20 text-white border border-white/20
                      backdrop-blur-sm transition-all duration-200 opacity-0 group-hover:opacity-100
                    "
                  >
                    📋 Copy
                  </button>
                </div>
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  );
}