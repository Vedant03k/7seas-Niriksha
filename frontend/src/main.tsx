import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Auth0Provider } from '@auth0/auth0-react'
import './index.css'
import App from './App.tsx'

console.log("Main.tsx is running", document.getElementById('root'));

// Replace these with your actual Auth0 parameters in a .env file
const domain = import.meta.env.VITE_AUTH0_DOMAIN || "YOUR_AUTH0_DOMAIN";
const clientId = import.meta.env.VITE_AUTH0_CLIENT_ID || "YOUR_AUTH0_CLIENT_ID";

try {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <Auth0Provider
        domain={domain}
        clientId={clientId}
        authorizationParams={{
          redirect_uri: window.location.origin
        }}
      >
        <App />
      </Auth0Provider>
    </StrictMode>,
  )
} catch (e) {
  console.error("Render error:", e);
  document.body.innerHTML += '<div style="color:red;font-size:20px;position:fixed;top:100px;left:0;z-index:9999;background:yellow;padding:1rem;">Render Error: ' + e.message + '</div>';
}
