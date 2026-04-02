import { useState } from 'react';
import { UploadCloud, FileVideo, Image as ImageIcon, CheckCircle, AlertOctagon, Activity, FileAudio } from 'lucide-react';

interface DetectorTabProps {
  acceptType: string;
  typeLabel: string;
  description: string;
}

export default function DetectorTab({ acceptType, typeLabel, description }: DetectorTabProps) {
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setAnalyzing(true);
    setResult(null);
    
    // If it's an audio file, send it to the real backend
    if (file.type.includes('audio')) {
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        const response = await fetch('/api/analyze/audio', {
          method: 'POST',
          body: formData,
        });
        
        const data = await response.json();
        setResult(data);
      } catch (error: any) {
        console.error('Error analyzing audio:', error);
        setResult({
          verdict: 'ERROR',
          confidence: 0,
          media_type: 'audio',
          artifacts_detected: ['Connection Error'],
          explanation: 'Could not connect to the backend. Make sure the FastAPI server is running on port 8000.'
        });
      } finally {
        setAnalyzing(false);
      }
    } else {
      // Mock API call to backend for images and video for now until those endpoints are built
      setTimeout(() => {
        setResult({
          verdict: 'FAKE',
          confidence: 0.94,
          media_type: file.type.includes('video') ? 'video' : 'image',
          artifacts_detected: ['GAN grid pattern anomalies', 'boundary inconsistency detected'],
          explanation: `Suspicious high-frequency artifacts detected in the ${typeLabel.toLowerCase()} source structure.`
        });
        setAnalyzing(false);
      }, 2000);
    }
  };

  const renderIcon = () => {
    if (file) {
      if (file.type.includes('video')) return <FileVideo size={48} className="text-emerald-400" />;
      if (file.type.includes('audio')) return <FileAudio size={48} className="text-emerald-400" />;
      return <ImageIcon size={48} className="text-emerald-400" />;
    }
    return <UploadCloud size={48} className="text-slate-400" />;
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Header Description */}
      <div className="text-center">
        <p className="text-slate-400 max-w-2xl mx-auto">{description}</p>
      </div>

      {/* Upload Zone */}
      <div 
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors
          ${file ? 'border-emerald-500 bg-slate-800/50' : 'border-slate-600 hover:border-slate-500 bg-slate-800/20'}`}
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
          {renderIcon()}
          <div>
            <span className="text-lg font-medium">
              {file ? file.name : `Click to upload or drag & drop your ${typeLabel}`}
            </span>
            {!file && <p className="text-sm text-slate-400 mt-1">Supports {typeLabel} format (Max 50MB)</p>}
          </div>
        </label>
      </div>

      {/* Analyze Button */}
      {file && !result && (
        <div className="flex justify-center">
          <button 
            onClick={handleAnalyze}
            disabled={analyzing}
            className="px-8 py-3 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-semibold shadow-lg shadow-emerald-900/20 transition-all flex items-center space-x-2"
          >
            {analyzing && <Activity className="animate-spin" size={20} />}
            <span>{analyzing ? `Analyzing ${typeLabel}...` : `Scan ${typeLabel} for Deepfakes`}</span>
          </button>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="bg-slate-800 rounded-xl p-8 space-y-6 shadow-xl border border-slate-700 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <h2 className="text-2xl font-bold flex items-center space-x-2 border-b border-slate-700 pb-4">
            <span>Analysis Results</span>
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Verdict Card */}
            <div className={`p-6 rounded-lg border ${result.verdict === 'FAKE' ? 'bg-rose-950/30 border-rose-900/50 text-rose-200' : 'bg-emerald-950/30 border-emerald-900/50 text-emerald-200'}`}>
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-wider mb-1 opacity-80">Final Verdict</h3>
                  <p className={`text-4xl font-black ${result.verdict === 'FAKE' ? 'text-rose-400' : 'text-emerald-400'}`}>
                    {result.verdict}
                  </p>
                </div>
                {result.verdict === 'FAKE' ? <AlertOctagon size={40} className="text-rose-500" /> : <CheckCircle size={40} className="text-emerald-500" />}
              </div>
              <div className="mt-4 flex items-center justify-between border-t border-current/20 pt-4">
                <span className="opacity-80">AI Confidence Score</span>
                <span className="font-bold text-xl">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>

            {/* Specifics */}
            <div className="bg-slate-900/50 rounded-lg p-6 border border-slate-700 text-slate-300">
              <h3 className="text-sm font-semibold uppercase tracking-wider mb-4 text-slate-400">Detected Anomalies</h3>
              <ul className="space-y-2 mb-4">
                {(result.artifacts_detected || []).map((artifact: string, idx: number) => (
                  <li key={idx} className="flex items-center space-x-2">
                     <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
                     <span>{artifact}</span>
                  </li>
                ))}
              </ul>
              <p className="text-sm leading-relaxed text-slate-400 border-t border-slate-700 pt-4">
                {result.explanation}
              </p>
            </div>
          </div>
          
          {/* Heatmap Placeholder (Only show if not audio, or show waveform for audio) */}
          {result.media_type !== 'audio' ? (
            <div className="bg-slate-900 rounded-lg p-6 border border-slate-700 flex flex-col items-center justify-center min-h-[250px] relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/10 to-purple-500/10 mix-blend-overlay"></div>
              <Activity size={48} className="text-slate-600 mb-3 group-hover:text-indigo-400 transition-colors" />
              <p className="text-slate-400 font-medium">Grad-CAM Heatmap Analysis</p>
              <p className="text-sm text-slate-500 mt-1">Visualizing manipulated pixel regions</p>
            </div>
          ) : (
            <div className="bg-slate-900 rounded-lg p-6 border border-slate-700 flex flex-col items-center justify-center min-h-[150px] relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-tr from-emerald-500/10 to-teal-500/10 mix-blend-overlay"></div>
              <Activity size={48} className="text-slate-600 mb-3 group-hover:text-emerald-400 transition-colors" />
              <p className="text-slate-400 font-medium">Frequency / Spectrogram Analysis</p>
              <p className="text-sm text-slate-500 mt-1">Flagging synthetic vocal signatures</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}