import { useState } from 'react';
import { Activity, Layers, Image as ImageIcon, FileVideo, Music, LogOut, LogIn, Loader2, Zap } from 'lucide-react';
import { useAuth0 } from '@auth0/auth0-react';
import DetectorTab from './components/DetectorTab';
import nirikshaLogo from './assets/nirikshalogo.png';

type TabType = 'COMBINED' | 'IMAGE' | 'VIDEO' | 'AUDIO';

function App() {
  const { isAuthenticated, isLoading, loginWithRedirect, logout, user } = useAuth0();
  const [activeTab, setActiveTab] = useState<TabType>('COMBINED');
  const [credits, setCredits] = useState<number>(1000);

  const TABS = [
    { id: 'COMBINED', label: 'All-in-One Detector', icon: Layers, accept: 'image/*,video/*,audio/*', desc: 'Upload any image, video, or audio file. Our multimodal AI engine will automatically route it to the appropriate model for deepfake analysis.' },
    { id: 'IMAGE', label: 'Image Forensics', icon: ImageIcon, accept: 'image/*', desc: 'Specially tuned Xception/EfficientNet engine designed to detect GAN artifacts and pixel-level inconsistencies in static images.' },
    { id: 'VIDEO', label: 'Video Analysis', icon: FileVideo, accept: 'video/*', desc: 'Analyzes temporal consistency across frames, utilizing lip-sync matching and facial landmark drift checking.' },
    { id: 'AUDIO', label: 'Audio & Voice', icon: Music, accept: 'audio/*', desc: 'Uses Wav2Vec2 and LCNN to extract spectrogram features detecting modern voice cloning (TTS, voice-conversion).' }
  ];

  const currentTabInfo = TABS.find(t => t.id === activeTab);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#e5efff] flex flex-col items-center justify-center text-blue-500">
        <Loader2 className="animate-spin mb-4" size={48} />
        <p className="text-blue-600 text-lg animate-pulse font-bold">Loading secure environment...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-[#e5efff] text-slate-800 flex items-center justify-center p-8 font-sans antialiased selection:bg-blue-300/50">
        <div className="clay-card p-10 max-w-lg w-full text-center space-y-8 relative overflow-hidden">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3/4 h-32 bg-blue-300/30 blur-[80px] pointer-events-none" />
          
          <div className="flex items-center justify-center space-x-3 mx-auto">
            <img src={nirikshaLogo} alt="Niriksha Logo" className="h-28 w-auto object-contain drop-shadow-lg" />
          </div>
          
          <h1 className="text-4xl font-extrabold tracking-tight bg-gradient-to-br from-blue-600 to-indigo-500 bg-clip-text text-transparent pb-1">
            Welcome to Niriksha
          </h1>
          
          <p className="text-slate-500 text-lg leading-relaxed font-medium">
            Your advanced AI-powered platform to determine the authenticity of media and maintain digital trust.
            Please authenticate to access the forensic tools.
          </p>

          <button
            onClick={() => loginWithRedirect()}
            className="clay-btn flex items-center justify-center w-full space-x-2 py-4 px-6 hover:brightness-110"
          >
            <LogIn size={20} />
            <span className="text-lg">Secure Login with Auth0</span>
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#e5efff] text-slate-800 p-8 font-sans antialiased selection:bg-blue-300/50">
      <div className="max-w-5xl mx-auto space-y-10 relative">
        
        <header className="relative text-center space-y-4 pt-12 mt-4">
          {/* User Profile & Logout Button */}
          <div className="absolute -top-4 right-0 flex items-center mb-6 space-x-4">
            
            {/* Credit Display */}
            <div className="hidden sm:flex items-center space-x-2 bg-gradient-to-br from-amber-200 to-yellow-400 px-4 py-2.5 rounded-[1.5rem] border-2 border-white shadow-[4px_4px_8px_rgba(251,191,36,0.4),inset_2px_2px_4px_white,inset_-2px_-2px_4px_rgba(217,119,6,0.3)] min-w-max">
              <Zap size={18} className="text-yellow-700 drop-shadow-sm" />
              <span className="text-sm font-extrabold text-yellow-800 tracking-wide">{credits} Credits</span>
            </div>

            <div className="hidden sm:flex items-center space-x-3 bg-white/60 px-5 py-2.5 rounded-[1.5rem] shadow-[4px_4px_8px_rgba(170,190,230,0.3),inset_-2px_-2px_4px_rgba(170,190,230,0.2),inset_2px_2px_4px_white]">
              {user?.picture ? (
                <img src={user.picture} alt={user?.name || 'User'} className="w-10 h-10 rounded-full shadow-inner" />
              ) : (
                <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 font-bold border-2 border-white shadow-inner">
                  {user?.name?.[0]?.toUpperCase() || 'U'}
                </div>
              )}
              <span className="text-sm font-bold text-slate-600 max-w-[150px] truncate">
                {user?.name || user?.email || 'Authenticated User'}
              </span>
            </div>
            
            <button
              onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}
              className="clay-tab flex items-center space-x-2 py-3 px-5 text-sm font-bold text-red-500 hover:text-red-600"
              title="Logout"
            >
              <LogOut size={16} />
              <span className="hidden sm:inline">Logout</span>
            </button>
          </div>

          <div className="flex items-center justify-center space-x-3">
            <img src={nirikshaLogo} alt="Niriksha Logo" className="h-16 w-auto object-contain drop-shadow-md" />
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-br from-blue-600 to-indigo-500 bg-clip-text text-transparent pb-1">
              Niriksha Deepfake
            </h1>
          </div>
          <p className="text-slate-500 max-w-2xl mx-auto text-lg font-medium">
            An advanced AI-powered platform to determine the authenticity of media and maintain digital trust.
          </p>
        </header>

        <div className="clay-card p-3 max-w-4xl mx-auto">
          <nav className="flex items-center space-x-3 overflow-x-auto select-none p-1" aria-label="Tabs">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as TabType)}
                  className={`flex-1 flex min-w-[140px] items-center justify-center space-x-2 py-3.5 px-4 clay-tab ${isActive ? 'active' : 'hover:scale-105'}`}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <Icon size={18} className={isActive ? 'text-white' : 'text-blue-500'} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        <main className="clay-card p-6 md:p-10 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-blue-400/10 rounded-full blur-[60px] pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-indigo-400/10 rounded-full blur-[60px] pointer-events-none" />
          
          <div className="relative z-10">
            {currentTabInfo && (
              <DetectorTab 
                key={activeTab} 
                acceptType={currentTabInfo.accept}
                typeLabel={currentTabInfo.id !== 'COMBINED' ? currentTabInfo.label : 'Media'}
                description={currentTabInfo.desc}
                credits={credits}
                setCredits={setCredits}
              />
            )}
          </div>
        </main>
        
        <footer className="text-center text-slate-400 font-bold text-sm mt-12 pb-4">
           Powered by 7seas
        </footer>
      </div>
    </div>
  );
}

export default App;
