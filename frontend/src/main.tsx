import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

console.log("Main.tsx is running", document.getElementById('root'));

try {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
} catch (e) {
  console.error("Render error:", e);
  document.body.innerHTML += '<div style="color:red;font-size:20px;position:fixed;top:100px;left:0;z-index:9999;background:yellow;padding:1rem;">Render Error: ' + e.message + '</div>';
}
