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
    bg: "bg-red-500/15",
    border: "border-red-500/40",
    text: "text-red-400",
    icon: "❌",
    label: "FAKE",
  },
  REAL: {
    bg: "bg-emerald-500/15",
    border: "border-emerald-500/40",
    text: "text-emerald-400",
    icon: "✅",
    label: "REAL",
  },
  UNCERTAIN: {
    bg: "bg-amber-500/15",
    border: "border-amber-500/40",
    text: "text-amber-400",
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
    <div className="w-full bg-white/10 rounded-full h-2.5 overflow-hidden">
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
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-white font-medium">
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
    <div className="w-full max-w-4xl mx-auto px-6 py-8 space-y-6 bg-slate-900 rounded-3xl shadow-2xl text-slate-200">

      {!result && (
        <div
          className={`
            relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer
            transition-all duration-200
            ${isDragging
              ? "border-blue-400 bg-blue-500/10 scale-[1.01]"
              : "border-white/20 hover:border-white/40 bg-white/5"
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
                className="max-h-64 mx-auto rounded-xl object-contain shadow-xl"
              />
              <p className="text-slate-400 text-sm">
                {file?.name} · {((file?.size ?? 0) / 1024).toFixed(1)} KB
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="text-5xl">🖼️</div>
              <p className="text-white font-medium text-lg">
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
              flex-1 py-3 px-6 rounded-xl font-semibold text-white
              bg-blue-600 hover:bg-blue-500 disabled:opacity-50
              disabled:cursor-not-allowed transition-all duration-200
              flex items-center justify-center gap-2
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
              px-5 py-3 rounded-xl font-medium text-slate-400
              border border-white/20 hover:border-white/40
              hover:text-white transition-all duration-200
            "
          >
            Clear
          </button>
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400 text-sm">
          ⚠️ {error}
        </div>
      )}

      {result && (
        <div className="space-y-4">

          {(() => {
            const cfg = verdictConfig[result.verdict];
            return (
              <div
                className={`
                  ${cfg.bg} border ${cfg.border}
                  rounded-2xl p-6 flex items-center justify-between
                `}
              >
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <span className="text-3xl">{cfg.icon}</span>
                    <span className={`text-4xl font-black ${cfg.text}`}>
                      {cfg.label}
                    </span>
                  </div>
                  <p className="text-slate-400 text-sm">
                    Confidence: {result.confidence_percent}% ·{" "}
                    {result.manipulation_type.type}
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-slate-500 text-xs mb-1">ENSEMBLE SCORE</div>
                  <div className={`text-5xl font-black font-mono ${cfg.text}`}>
                    {result.confidence_percent}%
                  </div>
                </div>
              </div>
            );
          })()}

          <div className="flex gap-1 bg-white/5 rounded-xl p-1 overflow-x-auto">
            {(["overview", "heatmap", "fft", "metadata", "legal"] as const).map(
              (tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`
                    flex-1 min-w-max py-2 px-3 rounded-lg text-sm font-medium capitalize
                    transition-all duration-150
                    ${activeTab === tab
                      ? "bg-blue-600 text-white"
                      : "text-slate-400 hover:text-white"
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

              <div className="bg-white/5 border border-white/10 rounded-xl p-5">
                <h3 className="text-white font-semibold mb-4">
                  Multi-Layer Analysis Breakdown
                </h3>
                <ScoreRow
                  label="Neural Network (Vision Transformer)"
                  value={result.model_scores.neural_network_fake_prob}
                  color="bg-red-500"
                />
                <ScoreRow
                  label="FFT Frequency Anomaly"
                  value={result.model_scores.fft_anomaly_score}
                  color="bg-purple-500"
                />
                <ScoreRow
                  label="Ensemble Score (Final)"
                  value={result.model_scores.ensemble_score}
                  color={
                    result.verdict === "FAKE"
                      ? "bg-red-500"
                      : result.verdict === "UNCERTAIN"
                      ? "bg-amber-500"
                      : "bg-emerald-500"
                  }
                />
              </div>

              <div className="bg-white/5 border border-white/10 rounded-xl p-5">
                <h3 className="text-white font-semibold mb-2">
                  Manipulation Classification
                </h3>
                <div className="inline-block bg-blue-500/20 border border-blue-500/40 text-blue-300 text-sm font-medium px-3 py-1 rounded-full mb-3">
                  {result.manipulation_type.type}
                </div>
                <p className="text-slate-400 text-sm leading-relaxed">
                  {result.manipulation_type.description}
                </p>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-xl p-5">
                <h3 className="text-white font-semibold mb-3">
                  Detected Anomalies
                </h3>
                <ul className="space-y-2">
                  {result.artifacts_detected.map((a, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm">
                      <span className="text-red-400 mt-0.5 flex-shrink-0">●</span>
                      <span className="text-slate-300">{a}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-xl p-5">
                <h3 className="text-white font-semibold mb-2">
                  Analysis Summary
                </h3>
                <p className="text-slate-400 text-sm leading-relaxed">
                  {result.explanation}
                </p>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-xl p-5">
                <h3 className="text-white font-semibold mb-2">Face Detection</h3>
                <p className="text-slate-400 text-sm">
                  {result.face_detection.faces_found === 0
                    ? "No faces detected in this image."
                    : `${result.face_detection.faces_found} face(s) detected and analyzed.`
                  }
                </p>
              </div>

              <button
                onClick={reset}
                className="
                  w-full py-3 rounded-xl font-medium text-slate-400
                  border border-white/20 hover:border-white/40
                  hover:text-white transition-all duration-200
                "
              >
                + Analyze Another Image
              </button>
            </div>
          )}

          {activeTab === "heatmap" && (
            <div className="bg-white/5 border border-white/10 rounded-xl p-5 space-y-4">
              <h3 className="text-white font-semibold">
                GradCAM Explanation Heatmap
              </h3>
              <p className="text-slate-500 text-sm">
                Highlighted regions show which areas of the image most
                influenced the FAKE classification. Red = high influence.
              </p>
              {result.gradcam_heatmap ? (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-slate-500 text-xs mb-2 uppercase tracking-wider">
                      Original
                    </p>
                    {previewUrl && (
                      <img
                        src={previewUrl}
                        alt="Original"
                        className="rounded-xl w-full object-contain max-h-64"
                      />
                    )}
                  </div>
                  <div>
                    <p className="text-slate-500 text-xs mb-2 uppercase tracking-wider">
                      GradCAM Overlay
                    </p>
                    <img
                      src={`data:image/png;base64,${result.gradcam_heatmap}`}
                      alt="GradCAM"
                      className="rounded-xl w-full object-contain max-h-64"
                    />
                  </div>
                </div>
              ) : (
                <div className="text-slate-500 text-sm text-center py-8">
                  GradCAM not generated (low confidence or error).
                </div>
              )}
            </div>
          )}

          {activeTab === "fft" && (
            <div className="bg-white/5 border border-white/10 rounded-xl p-5 space-y-4">
              <h3 className="text-white font-semibold">
                Frequency Domain Analysis (FFT)
              </h3>
              <p className="text-slate-500 text-sm">
                GAN-generated images leave characteristic periodic patterns in
                the frequency domain. Bright spots away from center = GAN artifacts.
              </p>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="bg-white/5 rounded-xl p-4 space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Anomaly Score</span>
                    <span className="font-mono text-white">
                      {(result.frequency_analysis.fft_anomaly_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">GAN Grid Pattern</span>
                    <span className={result.frequency_analysis.has_gan_grid
                      ? "text-red-400 font-medium"
                      : "text-emerald-400"
                    }>
                      {result.frequency_analysis.has_gan_grid ? "DETECTED" : "Not found"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Peak Pixel Count</span>
                    <span className="font-mono text-white">
                      {result.frequency_analysis.peak_pixel_count}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Spectral Peak Ratio</span>
                    <span className="font-mono text-white">
                      {result.frequency_analysis.spectral_peak_ratio.toFixed(4)}
                    </span>
                  </div>
                </div>
                <div>
                  {result.frequency_analysis.fft_spectrum_image ? (
                    <img
                      src={`data:image/png;base64,${result.frequency_analysis.fft_spectrum_image}`}
                      alt="FFT Spectrum"
                      className="rounded-xl w-full"
                    />
                  ) : (
                    <div className="text-slate-500 text-sm text-center py-4">
                      FFT spectrum unavailable
                    </div>
                  )}
                  <p className="text-slate-600 text-xs mt-1 text-center">
                    FFT Magnitude Spectrum
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeTab === "metadata" && (
            <div className="bg-white/5 border border-white/10 rounded-xl p-5 space-y-4">
              <h3 className="text-white font-semibold">Metadata & EXIF Analysis</h3>

              {result.metadata_analysis.metadata_suspicious && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 text-sm text-red-400">
                  ⚠️ Suspicious metadata detected
                </div>
              )}

              <div className="grid grid-cols-2 gap-3 text-sm">
                {[
                  ["File Size", `${result.metadata_analysis.file_size_kb} KB`],
                  ["Dimensions", result.metadata_analysis.image_dimensions],
                  ["Aspect Ratio", result.metadata_analysis.aspect_ratio],
                  ["EXIF Fields", result.metadata_analysis.exif_fields_found],
                  ["AI Tool", result.metadata_analysis.ai_tool_detected ?? "None detected"],
                ].map(([k, v]) => (
                  <div key={String(k)} className="bg-white/5 rounded-lg p-3">
                    <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">
                      {k}
                    </div>
                    <div className="text-white font-mono">{String(v)}</div>
                  </div>
                ))}
              </div>

              {result.metadata_analysis.metadata_flags.length > 0 && (
                <div>
                  <h4 className="text-white font-medium mb-2 text-sm">
                    Metadata Flags
                  </h4>
                  <ul className="space-y-1">
                    {result.metadata_analysis.metadata_flags.map((f, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm">
                        <span className="text-amber-400 flex-shrink-0">●</span>
                        <span className="text-slate-300">{f}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {Object.keys(result.exif_data).length > 0 && (
                <div>
                  <h4 className="text-white font-medium mb-2 text-sm">
                    EXIF Data
                  </h4>
                  <div className="bg-black/30 rounded-xl p-3 space-y-1 max-h-48 overflow-y-auto">
                    {Object.entries(result.exif_data).map(([k, v]) => (
                      <div key={k} className="flex gap-2 text-xs">
                        <span className="text-blue-400 font-mono w-32 flex-shrink-0">
                          {k}
                        </span>
                        <span className="text-slate-400 truncate">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === "legal" && (
            <div className="bg-white/5 border border-white/10 rounded-xl p-5 space-y-4">
              <h3 className="text-white font-semibold">
                🔒 Tamper-Proof Legal Report
              </h3>
              <p className="text-slate-500 text-sm">
                A SHA-256 cryptographic hash was generated from the raw file bytes
                at the moment of upload. This can be used to verify the report
                against the original file in legal or forensic proceedings.
              </p>

              <div className="bg-black/40 border border-white/10 rounded-xl p-4 space-y-3">
                <div>
                  <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">
                    File Hash (SHA-256)
                  </div>
                  <div className="font-mono text-xs text-emerald-400 break-all">
                    {result.legal.file_hash_sha256}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">
                    Analysis Timestamp (UTC)
                  </div>
                  <div className="font-mono text-xs text-white">
                    {result.legal.analysis_timestamp}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">
                    Model Version
                  </div>
                  <div className="font-mono text-xs text-white">
                    {result.legal.model_version}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">
                    Verdict on Record
                  </div>
                  <div
                    className={`font-mono text-sm font-bold ${
                      result.verdict === "FAKE"
                        ? "text-red-400"
                        : result.verdict === "REAL"
                        ? "text-emerald-400"
                        : "text-amber-400"
                    }`}
                  >
                    {result.verdict} — {result.confidence_percent}% confidence
                  </div>
                </div>
              </div>

              <p className="text-slate-600 text-xs">
                {result.legal.tamper_proof_note}
              </p>

              <button
                onClick={downloadPDF}
                className="w-full mt-4 py-3 px-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold rounded-xl shadow-lg transition-all duration-200 flex items-center justify-center gap-2"
              >
                <span>📄</span> Download Certified PDF Report
              </button>

              <div>
                <h4 className="text-white font-medium mb-2 text-sm mt-4">
                  WhatsApp Report Summary
                </h4>
                <pre className="bg-black/40 rounded-xl p-4 text-xs text-slate-300 whitespace-pre-wrap font-mono leading-relaxed">
                  {result.whatsapp_summary}
                </pre>
                <button
                  onClick={() =>
                    navigator.clipboard.writeText(result.whatsapp_summary)
                  }
                  className="
                    mt-2 px-4 py-2 text-xs rounded-lg
                    bg-blue-600/30 border border-blue-500/40
                    text-blue-300 hover:bg-blue-600/50
                    transition-all duration-150
                  "
                >
                  📋 Copy to Clipboard
                </button>
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  );
}