import { useState } from 'react';
import { UploadCloud, FileText, CheckCircle, AlertOctagon, Activity, Zap, Crown, Shield, Eye, Type, Fingerprint, Layers, Download } from 'lucide-react';

interface DocumentDetectorTabProps {
  credits: number;
  setCredits: (val: number) => void;
}

const ANALYSIS_ICONS: Record<string, { icon: string; label: string }> = {
  metadata_forensics: { icon: '🔍', label: 'Metadata' },
  structural_analysis: { icon: '🏗️', label: 'Structure' },
  visual_ela: { icon: '👁️', label: 'Visual/ELA' },
  text_consistency: { icon: '📝', label: 'Text Analysis' },
  digital_signature: { icon: '🔐', label: 'Signature' },
};

export default function DocumentDetectorTab({ credits, setCredits }: DocumentDetectorTabProps) {
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [showPaywall, setShowPaywall] = useState(false);
  const [generatingReport, setGeneratingReport] = useState(false);

  const COST = 50;

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const f = e.dataTransfer.files[0];
      if (f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf')) {
        setFile(f);
        setResult(null);
        setShowPaywall(false);
      }
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    if (credits < COST) {
      setShowPaywall(true);
      return;
    }

    setAnalyzing(true);
    setResult(null);
    setShowPaywall(false);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/analyze/document', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(typeof err.detail === 'string' ? err.detail : err.detail?.message || 'Analysis failed');
      }

      const data = await response.json();
      setResult(data);
      setCredits(credits - COST);
    } catch (error: any) {
      setResult({
        verdict: 'ERROR',
        confidence: 0,
        media_type: 'document',
        artifacts_detected: ['Connection Error'],
        explanation: error.message || 'Could not connect to the backend. Make sure the FastAPI server is running on port 8000.',
      });
    } finally {
      setAnalyzing(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!result) return;
    setGeneratingReport(true);
    try {
      const response = await fetch('/api/analyze/document/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result }),
      });
      if (!response.ok) throw new Error('Failed to generate report');
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Niriksha_Document_Report_${(file?.name || 'document').replace(/\.[^.]+$/, '')}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Report generation failed:', err);
    } finally {
      setGeneratingReport(false);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Header */}
      <div className="text-center">
        <p className="text-slate-500 max-w-2xl mx-auto font-medium">
          Upload a PDF document for multi-layer forensic analysis. Detects metadata tampering, structural anomalies, image manipulation (ELA), AI-generated text, and missing digital signatures.
        </p>
      </div>

      {/* Upload Zone */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className={`clay-card p-12 text-center transition-all duration-300
          ${file ? 'border-2 border-blue-400 bg-blue-50/50' : 'hover:scale-[1.02] cursor-pointer'}`}
      >
        <input
          type="file"
          id="file-upload-document"
          className="hidden"
          accept="application/pdf,.pdf"
          onChange={(e) => {
            if (e.target.files) {
              setFile(e.target.files[0]);
              setResult(null);
            }
          }}
        />
        <label htmlFor="file-upload-document" className="cursor-pointer flex flex-col items-center space-y-4">
          <div className="bg-white p-4 rounded-full shadow-[4px_4px_8px_rgba(170,190,230,0.4),inset_-2px_-2px_4px_rgba(170,190,230,0.2),inset_2px_2px_4px_white] mb-2 inline-block">
            {file ? (
              <FileText size={48} className="text-blue-500 drop-shadow-md" />
            ) : (
              <UploadCloud size={48} className="text-blue-400 drop-shadow-sm" />
            )}
          </div>
          <div className="flex flex-col items-center justify-center space-y-2">
            <div className="flex items-center justify-center space-x-4">
              <span className="text-xl font-bold text-slate-700">
                {file ? file.name : 'Click to upload or drag & drop your PDF'}
              </span>
              {file && (
                <span className="text-sm font-extrabold text-blue-500 bg-blue-100/50 px-3 py-1 rounded-full border border-blue-200/50 shadow-sm">
                  Cost: {COST} Credits
                </span>
              )}
            </div>
            {!file && <p className="text-sm text-slate-500 font-medium">Supports PDF documents (Max 50MB)</p>}
          </div>
        </label>
      </div>

      {/* Analyze Button */}
      {file && !result && !showPaywall && (
        <div className="flex justify-center mt-8">
          <button
            onClick={handleAnalyze}
            disabled={analyzing}
            className="clay-btn px-10 py-4 text-lg w-full max-w-md disabled:opacity-70 flex justify-center items-center space-x-3"
          >
            {analyzing && <Activity className="animate-spin" size={24} />}
            <span>{analyzing ? 'Analyzing Document...' : 'Scan Document for Forgery'}</span>
          </button>
        </div>
      )}

      {/* Paywall Banner */}
      {showPaywall && (
        <div className="bg-gradient-to-br from-amber-50 to-amber-100 p-8 md:p-10 rounded-[2.5rem] border-2 border-white shadow-[8px_8px_16px_rgba(251,191,36,0.2),inset_-4px_-4px_8px_rgba(251,191,36,0.1),inset_4px_4px_8px_white] space-y-6 text-center mt-8 animate-in fade-in duration-500">
          <div className="w-20 h-20 bg-white rounded-full flex items-center justify-center mx-auto shadow-[4px_4px_8px_rgba(251,191,36,0.3),inset_-2px_-2px_4px_rgba(251,191,36,0.1),inset_2px_2px_4px_white]">
            <Crown size={40} className="text-amber-500 drop-shadow-sm" />
          </div>
          <h2 className="text-3xl font-black text-amber-900">Out of Credits</h2>
          <p className="text-lg text-amber-800/80 font-bold max-w-md mx-auto leading-relaxed">
            You need <span className="text-xl text-amber-600">{COST} credits</span> to analyze a document, but you only have <span className="text-xl text-amber-600">{credits}</span> left.
          </p>
          <button className="px-10 py-4 mt-2 text-lg font-bold w-full max-w-md rounded-[1.5rem] bg-gradient-to-r from-amber-400 to-yellow-500 text-yellow-950 border-none transition-all active:scale-95 shadow-[8px_8px_16px_rgba(245,158,11,0.3),inset_-4px_-4px_8px_rgba(217,119,6,0.5),inset_4px_4px_8px_rgba(253,230,138,0.8)] hover:brightness-105">
            Upgrade to Premium Pro
          </button>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="clay-card p-8 md:p-10 space-y-8 mt-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <h2 className="text-3xl font-extrabold flex items-center space-x-2 text-slate-800 border-b border-slate-200 pb-5">
            <span>Analysis Results</span>
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Verdict Card */}
            <div className={`p-8 rounded-[2rem] border-0 text-white font-bold
              ${result.verdict === 'FAKE'
                ? 'bg-rose-500 shadow-[8px_8px_16px_rgba(244,63,94,0.3),inset_-4px_-4px_8px_rgba(225,29,72,0.5),inset_4px_4px_8px_rgba(251,113,133,0.8)]'
                : result.verdict === 'ERROR'
                  ? 'bg-slate-500 shadow-[8px_8px_16px_rgba(100,116,139,0.3),inset_-4px_-4px_8px_rgba(71,85,105,0.5),inset_4px_4px_8px_rgba(148,163,184,0.8)]'
                  : 'bg-emerald-500 shadow-[8px_8px_16px_rgba(16,185,129,0.3),inset_-4px_-4px_8px_rgba(5,150,105,0.5),inset_4px_4px_8px_rgba(52,211,153,0.8)]'}`}>
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-sm font-extrabold uppercase tracking-wider mb-2 opacity-90">Final Verdict</h3>
                  <p className="text-5xl font-black drop-shadow-md">{result.verdict}</p>
                </div>
                {result.verdict === 'FAKE' ? (
                  <AlertOctagon size={48} className="drop-shadow-lg opacity-90" />
                ) : (
                  <CheckCircle size={48} className="drop-shadow-lg opacity-90" />
                )}
              </div>
              <div className="mt-8 flex items-center justify-between border-t border-white/20 pt-6">
                <span className="opacity-90 font-semibold text-lg">AI Confidence Score</span>
                <span className="font-extrabold text-3xl drop-shadow-sm">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>

            {/* Anomalies */}
            <div className="bg-blue-50/50 p-8 rounded-[2rem] border-2 border-white shadow-[inset_4px_4px_8px_rgba(170,190,230,0.3),inset_-4px_-4px_8px_white] space-y-6 text-slate-700 font-medium">
              <h3 className="text-sm font-extrabold uppercase tracking-wider mb-2 text-slate-400">Detected Anomalies</h3>
              <ul className="space-y-3 mb-4 max-h-64 overflow-y-auto">
                {(result.artifacts_detected || []).map((artifact: string, idx: number) => (
                  <li key={idx} className="flex items-start space-x-3 bg-white/70 p-3 rounded-xl shadow-sm">
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-500 shadow-inner mt-1.5 shrink-0"></span>
                    <span className="font-bold text-slate-700 text-sm">{artifact}</span>
                  </li>
                ))}
              </ul>
              <p className="text-base leading-relaxed text-slate-600 bg-white/70 p-4 rounded-xl shadow-sm border border-blue-100/50">
                {result.explanation}
              </p>
            </div>
          </div>

          {/* Sub-scores breakdown */}
          {result.sub_scores && (
            <div className="clay-card border-none p-8 space-y-6 text-slate-700">
              <h3 className="text-sm font-extrabold uppercase tracking-wider text-slate-400">Multi-Layer Analysis Breakdown</h3>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                {Object.entries(ANALYSIS_ICONS).map(([key, { icon, label }]) => {
                  const score = result.sub_scores[key] ?? 0;
                  const pct = (score * 100).toFixed(1);
                  const isSuspicious = score > 0.3;
                  return (
                    <div key={key} className={`rounded-[1.5rem] p-5 text-center transition-all ${isSuspicious ? 'bg-rose-50 border-2 border-rose-200 text-rose-700 shadow-sm' : 'bg-emerald-50 border-2 border-emerald-200 text-emerald-700 shadow-sm'}`}>
                      <div className="text-3xl mb-2 drop-shadow-sm">{icon}</div>
                      <div className="text-xs font-bold uppercase tracking-wider mb-2 opacity-80">{label}</div>
                      <div className={`text-xl font-black drop-shadow-sm ${isSuspicious ? 'text-rose-600' : 'text-emerald-600'}`}>{pct}%</div>
                      <div className="text-sm font-bold text-slate-500 mt-2">{isSuspicious ? 'Suspicious' : 'Normal'}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Metadata Details */}
          {result.metadata_info && (
            <div className="clay-card border-none p-8 space-y-4">
              <h3 className="text-sm font-extrabold uppercase tracking-wider text-slate-400 flex items-center space-x-2">
                <Fingerprint size={16} />
                <span>Document Metadata</span>
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {[
                  { label: 'Producer', value: result.metadata_info.producer },
                  { label: 'Creator', value: result.metadata_info.creator },
                  { label: 'Author', value: result.metadata_info.author },
                  { label: 'Created', value: result.metadata_info.creation_date },
                  { label: 'Modified', value: result.metadata_info.modification_date },
                  { label: 'File Size', value: `${result.metadata_info.file_size_kb} KB` },
                  { label: 'SHA-256', value: result.metadata_info.file_hash_sha256?.slice(0, 24) + '...' },
                ].map((item, idx) => (
                  <div key={idx} className="flex items-center space-x-3 bg-white/70 p-3 rounded-xl shadow-sm">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-400 w-20 shrink-0">{item.label}</span>
                    <span className="font-mono text-sm text-slate-700 truncate">{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Structure Info */}
          {result.structure_info && (
            <div className="clay-card border-none p-8 space-y-4">
              <h3 className="text-sm font-extrabold uppercase tracking-wider text-slate-400 flex items-center space-x-2">
                <Layers size={16} />
                <span>Document Structure</span>
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: 'Pages', value: result.structure_info.page_count },
                  { label: 'Fonts', value: result.structure_info.total_fonts },
                  { label: 'Images', value: result.structure_info.total_images },
                  { label: 'Annotations', value: result.structure_info.total_annotations },
                ].map((item, idx) => (
                  <div key={idx} className="bg-white/70 p-4 rounded-xl shadow-sm text-center">
                    <div className="text-2xl font-black text-blue-600">{item.value}</div>
                    <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mt-1">{item.label}</div>
                  </div>
                ))}
              </div>
              {result.signature_info && (
                <div className="flex items-center space-x-3 mt-4">
                  <Shield size={18} className={result.signature_info.has_digital_signature ? 'text-emerald-500' : 'text-rose-400'} />
                  <span className={`font-bold text-sm ${result.signature_info.has_digital_signature ? 'text-emerald-600' : 'text-rose-500'}`}>
                    {result.signature_info.has_digital_signature
                      ? `Digital signature found (${result.signature_info.signature_count} signature${result.signature_info.signature_count > 1 ? 's' : ''})`
                      : 'No digital signature detected'}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Download Report Button */}
          <div className="flex justify-center pt-4">
            <button
              onClick={handleDownloadReport}
              disabled={generatingReport}
              className="clay-btn px-8 py-3 text-base flex items-center space-x-3 disabled:opacity-70"
            >
              {generatingReport ? (
                <Activity className="animate-spin" size={20} />
              ) : (
                <Download size={20} />
              )}
              <span>{generatingReport ? 'Generating Report...' : 'Download PDF Report'}</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
