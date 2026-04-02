import { useState } from 'react';
import { Activity, Layers, Image as ImageIcon, FileVideo, Music, LogOut, LogIn, Loader2 } from 'lucide-react';
import { useAuth0 } from '@auth0/auth0-react';
import DetectorTab from './components/DetectorTab';

type TabType = 'COMBINED' | 'IMAGE' | 'VIDEO' | 'AUDIO';

function App() {
  const { isAuthenticated, isLoading, loginWithRedirect, logout, user } = useAuth0();
  const [activeTab, setActiveTab] = useState<TabType>('COMBINED');

  const TABS = [
    { id: 'COMBINED', label: 'All-in-One Detector', icon: Layers, accept: 'image/*,video/*,audio/*', desc: 'Upload any image, video, or audio file. Our multimodal AI engine will automatically route it to the appropriate model for deepfake analysis.' },
    { id: 'IMAGE', label: 'Image Forensics', icon: ImageIcon, accept: 'image/*', desc: 'Specially tuned Xception/EfficientNet engine designed to detect GAN artifacts and pixel-level inconsistencies in static images.' },
    { id: 'VIDEO', label: 'Video Analysis', icon: FileVideo, accept: 'video/*', desc: 'Analyzes temporal consistency across frames, utilizing lip-sync matching and facial landmark drift checking.' },
    { id: 'AUDIO', label: 'Audio & Voice', icon: Music, accept: 'audio/*', desc: 'Uses Wav2Vec2 and LCNN to extract spectrogram features detecting modern voice cloning (TTS, voice-conversion).' }
  ];

  const currentTabInfo = TABS.find(t => t.id === activeTab);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-900 flex flex-col items-center justify-center text-emerald-400">
        <Loader2 className="animate-spin mb-4" size={48} />
        <p className="text-slate-400 text-lg animate-pulse">Loading secure environment...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-slate-900 text-slate-100 flex items-center justify-center p-8 font-sans antialiased selection:bg-emerald-500/30">
        <div className="bg-slate-800/60 p-10 rounded-2xl border border-slate-700/50 backdrop-blur-sm max-w-lg w-full text-center space-y-8 shadow-2xl relative overflow-hidden">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3/4 h-32 bg-emerald-500/10 blur-[100px] pointer-events-none" />
          
          <div className="flex items-center justify-center space-x-3 text-emerald-400 mx-auto">
            <Activity fill="currentColor" className="text-emerald-900 bg-emerald-400 rounded-full p-1" size={56} />
          </div>
          
          <h1 className="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-emerald-400 to-teal-200 bg-clip-text text-transparent">
            Welcome to Niriksha
          </h1>
          
          <p className="text-slate-400 text-lg leading-relaxed">
            Your advanced AI-powered platform to determine the authenticity of media and maintain digital trust.
            Please authenticate to access the forensic tools.
          </p>

          <button
            onClick={() => loginWithRedirect()}
            className="flex items-center justify-center w-full space-x-2 bg-emerald-500 hover:bg-emerald-400 text-slate-900 font-bold py-3.5 px-6 rounded-xl transition-all duration-200 shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:shadow-[0_0_25px_rgba(16,185,129,0.5)] transform hover:-translate-y-0.5"
          >
            <LogIn size={20} />
            <span>Secure Login with Auth0</span>
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-8 font-sans antialiased selection:bg-emerald-500/30">
      <div className="max-w-5xl mx-auto space-y-10">
        
        <header className="relative text-center space-y-4 pt-12 mt-4">
          {/* User Profile & Logout Button */}
          <div className="absolute -top-4 right-0 flex items-center mb-6 space-x-3">
            <div className="hidden sm:flex items-center space-x-3 bg-slate-800/80 px-4 py-2 rounded-full border border-slate-700/50">
              {user?.picture ? (
                <img src={user.picture} alt={user?.name || 'User'} className="w-8 h-8 rounded-full border-2 border-emerald-500/50" />
              ) : (
                <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-emerald-400 font-bold">
                  {user?.name?.[0]?.toUpperCase() || 'U'}
                </div>
              )}
              <span className="text-sm font-medium text-slate-300 max-w-[150px] truncate">
                {user?.name || user?.email || 'Authenticated User'}
              </span>
            </div>
            
            <button
              onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}
              className="flex items-center space-x-2 bg-slate-800 hover:bg-red-500/10 text-slate-400 hover:text-red-400 border border-slate-700 hover:border-red-500/30 py-2.5 px-4 rounded-xl transition-all duration-200 text-sm font-semibold shadow-sm"
              title="Logout"
            >
              <LogOut size={16} />
              <span className="hidden sm:inline">Logout</span>
            </button>
          </div>

          <div className="flex items-center justify-center space-x-3 text-emerald-400">
            <Activity fill="currentColor" className="text-emerald-900 bg-emerald-400 rounded-full p-1" size={44} />
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-emerald-400 to-teal-200 bg-clip-text text-transparent">
              Niriksha Deepfake
            </h1>
          </div>
          <p className="text-slate-400 max-w-2xl mx-auto text-lg">
            An advanced AI-powered platform to determine the authenticity of media and maintain digital trust.
          </p>
        </header>

        <div className="bg-slate-800/60 p-1.5 rounded-xl border border-slate-700/50 backdrop-blur-sm max-w-3xl mx-auto">
          <nav className="flex items-center space-x-1 overflow-x-auto select-none" aria-label="Tabs">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as TabType)}
                  className={"flex-1 flex min-w-[120px] items-center justify-center space-x-2 py-3 px-4 rounded-lg text-sm font-semibold transition-all duration-200 outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-slate-900 " + (isActive ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shadow-sm" : "text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-transparent")}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <Icon size={18} className={isActive ? 'text-emerald-400' : 'opacity-70'} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        <main className="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-6 md:p-10 shadow-2xl relative overflow-hidden backdrop-blur-sm">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3/4 h-32 bg-emerald-500/10 blur-[100px] pointer-events-none" />
          
          {currentTabInfo && (
            <DetectorTab 
              key={activeTab} 
              acceptType={currentTabInfo.accept}
              typeLabel={currentTabInfo.id !== 'COMBINED' ? currentTabInfo.label : 'Media'}
              description={currentTabInfo.desc}
            />
          )}
        </main>
        
        <footer className="text-center text-slate-500 text-sm mt-12 pb-4">
           Powered by 7seas
        </footer>
      </div>
    </div>
  );
}

export default App;
