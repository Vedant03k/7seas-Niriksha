import { useState } from 'react';
import { UploadCloud, FileVideo, Image as ImageIcon, CheckCircle, AlertOctagon, Activity, FileAudio, FileText, Zap, Crown, Download, Brain, BarChart3, Music, Mic } from 'lucide-react';

interface DetectorTabProps {
  acceptType: string;
  typeLabel: string;
  description: string;
  credits: number;
  setCredits: (val: number) => void;
}

export default function DetectorTab({ acceptType, typeLabel, description, credits, setCredits }: DetectorTabProps) {
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [showPaywall, setShowPaywall] = useState(false);
  const [generatingReport, setGeneratingReport] = useState(false);

  const getCost = (fileType: string) => fileType.includes('video') ? 100 : 50;

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setResult(null);
      setShowPaywall(false);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    
    const cost = getCost(file.type);
    if (credits < cost) {
      setShowPaywall(true);
      return;
    }
    
    setAnalyzing(true);
    setResult(null);
    setShowPaywall(false);
    
    let endpoint = "";
    if (file.type.includes('audio')) endpoint = "http://127.0.0.1:8000/analyze/audio";
    else if (file.type.includes('video')) endpoint = "http://127.0.0.1:8000/analyze/video";
    else throw new Error("Unsupported filetype");

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('API processing failed');

      const data = await response.json();
      setResult(data);
      setCredits(credits - cost);
    } catch (error: any) {
      console.error('Error analyzing media:', error);
      setResult({
        verdict: 'ERROR',
        confidence: 0,
        media_type: file.type.includes('audio') ? 'audio' : 'video',
        artifacts_detected: ['Connection Error'],
        explanation: 'Could not connect to the backend. Make sure FastAPI server runs on port 8000.'
      });
    } finally {
      setAnalyzing(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!result) return;
    setGeneratingReport(true);
    try {
      if (result.media_type === 'audio') {
        const payload = { ...result, filename: file?.name || 'unknown' };
        const response = await fetch('http://127.0.0.1:8000/analyze/audio/report', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ result: payload }),
        });
        if (!response.ok) throw new Error('Failed to generate report');
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Niriksha_Audio_Report_${(file?.name || 'audio').replace(/\.[^.]+$/, '')}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      } else {
        const resp = await fetch('http://127.0.0.1:8000/generate/report', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            verdict: result.verdict,
            confidence: result.confidence,
            media_type: result.media_type,
            artifacts_detected: result.artifacts_detected || [],
            explanation: result.explanation || ''
          })
        });
        if (!resp.ok) throw new Error('Failed to generate report');
        
        const blob = await resp.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `Niriksha_Report_${Date.now()}.pdf`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
      }
    } catch (err) {
      console.error('Report generation failed:', err);
    } finally {
      setGeneratingReport(false);
    }
  };

  const renderIcon = () => {
    if (file) {
      if (file.type.includes('video')) return <FileVideo size={48} className="text-blue-500 drop-shadow-md" />;
      if (file.type.includes('audio')) return <FileAudio size={48} className="text-blue-500 drop-shadow-md" />;
      return <ImageIcon size={48} className="text-blue-500 drop-shadow-md" />;
    }
    return <UploadCloud size={48} className="text-blue-400 drop-shadow-sm" />;
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Header Description */}
      <div className="text-center">
        <p className="text-slate-500 max-w-2xl mx-auto font-medium">{description}</p>
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
          id={`file-upload-${typeLabel}`}
          className="hidden" 
          accept={acceptType}
          onChange={(e) => {
            if (e.target.files) {
              setFile(e.target.files[0]);
              setResult(null);
            }
          }}
        />
        <label htmlFor={`file-upload-${typeLabel}`} className="cursor-pointer flex flex-col items-center space-y-4">
          <div className="bg-white p-4 rounded-full shadow-[4px_4px_8px_rgba(170,190,230,0.4),inset_-2px_-2px_4px_rgba(170,190,230,0.2),inset_2px_2px_4px_white] mb-2 inline-block">
            {renderIcon()}
          </div>
          <div className="flex flex-col items-center justify-center space-y-2">
            <div className="flex items-center justify-center space-x-4">
              <span className="text-xl font-bold text-slate-700">
                {file ? file.name : `Click to upload or drag & drop your ${typeLabel}`}
              </span>
              {file && (
                <span className="text-sm font-extrabold text-blue-500 bg-blue-100/50 px-3 py-1 rounded-full border border-blue-200/50 shadow-sm">
                  Cost: {getCost(file.type)} Credits
                </span>
              )}
            </div>
            {!file && <p className="text-sm text-slate-500 font-medium">Supports {typeLabel} format (Max 50MB)</p>}
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
            <span>{analyzing ? `Analyzing ${typeLabel}...` : `Scan ${typeLabel} for Deepfakes`}</span>
          </button>
        </div>
      )}

      {/* Paywall Banner */}
      {showPaywall && (
        <div className="bg-gradient-to-br from-amber-50 to-amber-100 p-8 md:p-10 rounded-[2.5rem] border-2 border-white shadow-[8px_8px_16px_rgba(251,191,36,0.2),inset_-4px_-4px_8px_rgba(251,191,36,0.1),inset_4px_4px_8px_white] space-y-6 text-center mt-8 animate-in fade-in duration-500">
          <div className="w-20 h-20 bg-white rounded-full flex items-center justify-center mx-auto shadow-[4px_4px_8px_rgba(251,191,36,0.3),inset_-2px_-2px_4px_rgba(251,191,36,0.1),inset_2px_2px_4px_white]">
            <Crown size={40} className="text-amber-500 drop-shadow-sm" />
          </div>
          
          <h2 className="text-3xl font-black text-amber-900 bg-clip-text">Out of Credits</h2>
          <p className="text-lg text-amber-800/80 font-bold max-w-md mx-auto leading-relaxed">
            You need <span className="text-xl text-amber-600">{file ? getCost(file.type) : 50} credits</span> to analyze a {typeLabel.toLowerCase()}, but you only have <span className="text-xl text-amber-600">{credits}</span> left.
          </p>
          
          <button className="px-10 py-4 mt-2 text-lg font-bold w-full max-w-md rounded-[1.5rem] bg-gradient-to-r from-amber-400 to-yellow-500 text-yellow-950 border-none transition-all active:scale-95 shadow-[8px_8px_16px_rgba(245,158,11,0.3),inset_-4px_-4px_8px_rgba(217,119,6,0.5),inset_4px_4px_8px_rgba(253,230,138,0.8)] hover:brightness-105">
            Upgrade to Premium Pro
          </button>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="clay-card p-8 md:p-10 space-y-8 mt-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="flex items-center justify-between border-b border-slate-200 pb-5">
            <h2 className="text-3xl font-extrabold flex items-center space-x-2 text-slate-800">
              <span>Analysis Results</span>
            </h2>
            {result.verdict !== 'ERROR' && (
              <button 
                onClick={handleDownloadReport}
                className="clay-btn bg-slate-800 text-white hover:bg-slate-700 px-5 py-2.5 rounded-xl flex items-center space-x-2 font-bold transition-all shadow-[4px_4px_8px_rgba(15,23,42,0.3),inset_-2px_-2px_4px_rgba(15,23,42,0.5),inset_2px_2px_4px_rgba(148,163,184,0.4)]"
              >
                <FileText size={20} />
                <span>Export PDF Report</span>
              </button>
            )}
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Verdict Card */}
            <div className={`p-8 rounded-[2rem] border-0 text-white font-bold
              ${result.verdict === 'FAKE' 
                ? 'bg-rose-500 shadow-[8px_8px_16px_rgba(244,63,94,0.3),inset_-4px_-4px_8px_rgba(225,29,72,0.5),inset_4px_4px_8px_rgba(251,113,133,0.8)]' 
                : 'bg-emerald-500 shadow-[8px_8px_16px_rgba(16,185,129,0.3),inset_-4px_-4px_8px_rgba(5,150,105,0.5),inset_4px_4px_8px_rgba(52,211,153,0.8)]'}`}>
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-sm font-extrabold uppercase tracking-wider mb-2 opacity-90">Final Verdict</h3>
                  <p className="text-5xl font-black drop-shadow-md">
                    {result.verdict}
                  </p>
                </div>
                {result.verdict === 'FAKE' ? <AlertOctagon size={48} className="drop-shadow-lg opacity-90" /> : <CheckCircle size={48} className="drop-shadow-lg opacity-90" />}
              </div>
              <div className="mt-8 flex items-center justify-between border-t border-white/20 pt-6">
                <span className="opacity-90 font-semibold text-lg">AI Confidence Score</span>
                <span className="font-extrabold text-3xl drop-shadow-sm">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>

            {/* Specifics */}
            <div className="bg-blue-50/50 p-8 rounded-[2rem] border-2 border-white shadow-[inset_4px_4px_8px_rgba(170,190,230,0.3),inset_-4px_-4px_8px_white] space-y-6 text-slate-700 font-medium">
              <h3 className="text-sm font-extrabold uppercase tracking-wider mb-2 text-slate-400">Detected Anomalies</h3>
              <ul className="space-y-3 mb-4">
                {(result.artifacts_detected || []).map((artifact: string, idx: number) => (
                  <li key={idx} className="flex items-center space-x-3 bg-white/70 p-3 rounded-xl shadow-sm">
                     <span className="w-2.5 h-2.5 rounded-full bg-rose-500 shadow-inner"></span>
                     <span className="font-bold text-slate-700">{artifact}</span>
                  </li>
                ))}
              </ul>
              <p className="text-base leading-relaxed text-slate-600 bg-white/70 p-4 rounded-xl shadow-sm border border-blue-100/50">
                {result.explanation}
              </p>
            </div>
          </div>
          
          {/* Heatmap Placeholder (Only show if not audio, or show waveform for audio) */}
          {result.media_type !== 'audio' ? (
            <div className="clay-card border-none flex flex-col items-center justify-center min-h-[250px] relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-300/10 to-indigo-300/10 mix-blend-overlay"></div>
              <Activity size={48} className="text-blue-300 mb-3 group-hover:text-blue-500 transition-colors drop-shadow-md" />
              <p className="text-slate-600 font-extrabold text-lg">Grad-CAM Heatmap Analysis</p>
              <p className="text-sm text-slate-500 font-medium mt-1">Visualizing manipulated pixel regions</p>
            </div>
          ) : (
            <div className="clay-card border-none p-8 space-y-6 text-slate-700">
              <h3 className="text-sm font-extrabold uppercase tracking-wider text-slate-400">Multi-Layer Analysis Breakdown</h3>
              {result.sub_scores ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                  {[
                    { label: 'Neural Model', key: 'neural_model', Icon: Brain },
                    { label: 'Spectral', key: 'spectral_analysis', Icon: BarChart3 },
                    { label: 'MFCC', key: 'mfcc_analysis', Icon: Music },
                    { label: 'Pitch/Prosody', key: 'pitch_prosody', Icon: Mic },
                  ].map((item) => {
                    const score = result.sub_scores[item.key] ?? 0;
                    const pct = (score * 100).toFixed(1);
                    const isSuspicious = score > 0.5;
                    return (
                      <div key={item.key} className="bg-white rounded-2xl p-5 text-center transition-all border-2 border-white shadow-[4px_4px_8px_rgba(170,190,230,0.4),inset_-2px_-2px_4px_rgba(170,190,230,0.2),inset_2px_2px_4px_white]">
                        <div className={`w-10 h-10 mx-auto mb-3 rounded-xl flex items-center justify-center ${isSuspicious ? 'bg-rose-50' : 'bg-emerald-50'}`}>
                          <item.Icon size={20} className={isSuspicious ? 'text-rose-500' : 'text-emerald-500'} />
                        </div>
                        <div className="text-xs font-bold uppercase tracking-wider mb-1 text-slate-500">{item.label}</div>
                        <div className={`text-xl font-black ${isSuspicious ? 'text-rose-600' : 'text-emerald-600'}`}>{pct}%</div>
                        <div className={`text-xs font-semibold mt-1 ${isSuspicious ? 'text-rose-500' : 'text-emerald-500'}`}>{isSuspicious ? 'Suspicious' : 'Normal'}</div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center min-h-[150px] relative overflow-hidden group bg-blue-50/50 rounded-[2rem] border-2 border-white shadow-inner">
                  <Activity size={48} className="text-blue-300 mb-3 group-hover:text-blue-500 transition-colors drop-shadow-md" />
                  <p className="text-slate-600 font-extrabold text-lg">Frequency / Spectrogram Analysis</p>
                </div>
              )}
            </div>
          )}

          {/* Download Report Button (Audio only) */}
          {result.media_type === 'audio' && (
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
          )}
        </div>
      )}
    </div>
  );
}