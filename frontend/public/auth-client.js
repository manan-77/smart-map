// Next.js-friendly Auth0 client wrapper (copied from legacy auth.js)

// Auth0 configuration — loaded from globals (browser-safe, no Node `process`)
const auth0Config = {
  domain: (window.__NAV_AI_AUTH0_DOMAIN__ || window.NEXT_PUBLIC_AUTH0_DOMAIN || 'YOUR_AUTH0_DOMAIN'),
  clientId: (window.__NAV_AI_AUTH0_CLIENT_ID__ || window.NEXT_PUBLIC_AUTH0_CLIENT_ID || 'YOUR_AUTH0_CLIENT_ID'),
  audience: (window.__NAV_AI_AUTH0_AUDIENCE__ || window.NEXT_PUBLIC_AUTH0_AUDIENCE || 'YOUR_AUTH0_AUDIENCE'),
  redirectUri: window.location.origin
};

// Keys we use to persist a simple "logged-in" state locally
const AUTH_STORAGE_KEYS = {
  token: 'navai_access_token',
  isAuthenticated: 'navai_is_authenticated'
};

let auth0Client = null;
let accessToken = null;
let isAuthenticated = false;

// Initialize Auth0 (idempotent)
async function initAuth0() {
  if (auth0Client) return;

  if (typeof auth0 === 'undefined') {
    console.error('Auth0 SPA JS not loaded yet.');
    return;
  }

  auth0Client = await auth0.createAuth0Client({
    domain: auth0Config.domain,
    clientId: auth0Config.clientId,
    authorizationParams: {
      audience: auth0Config.audience,
      redirect_uri: auth0Config.redirectUri
    }
  });

  // 1) First, try to restore from our own localStorage keys
  try {
    const storedToken = localStorage.getItem(AUTH_STORAGE_KEYS.token);
    const storedAuthFlag = localStorage.getItem(AUTH_STORAGE_KEYS.isAuthenticated);
    if (storedToken && storedAuthFlag === 'true') {
      accessToken = storedToken;
      isAuthenticated = true;
      console.debug('[Auth] Restored token from localStorage');
      unlockChat();
      return;
    }
  } catch (e) {
    console.warn('[Auth] Unable to read localStorage:', e);
  }

  // 2) Handle callback if we just returned from Auth0
  const query = window.location.search;
  if (query.includes('code=') && query.includes('state=')) {
    await auth0Client.handleRedirectCallback();
    accessToken = await auth0Client.getTokenSilently();
    isAuthenticated = true;

    try {
      localStorage.setItem(AUTH_STORAGE_KEYS.token, accessToken);
      localStorage.setItem(AUTH_STORAGE_KEYS.isAuthenticated, 'true');
      console.debug('[Auth] Saved token to localStorage after callback');
    } catch (e) {
      console.warn('[Auth] Unable to write localStorage after callback:', e);
    }

    window.history.replaceState({}, document.title, '/');
    unlockChat();
    return;
  }

  // 3) Otherwise, ask Auth0 if there is an existing session
  isAuthenticated = await auth0Client.isAuthenticated();

  if (isAuthenticated) {
    accessToken = await auth0Client.getTokenSilently();

    try {
      localStorage.setItem(AUTH_STORAGE_KEYS.token, accessToken);
      localStorage.setItem(AUTH_STORAGE_KEYS.isAuthenticated, 'true');
      console.debug('[Auth] Saved token to localStorage from existing session');
    } catch (e) {
      console.warn('[Auth] Unable to write localStorage from existing session:', e);
    }

    unlockChat();
    return;
  }

  // 4) No session – keep chat locked
  lockChat();
}

function lockChat() {
  const locked = document.getElementById('lockedInput');
  const unlocked = document.getElementById('unlockedInput');
  const authButton = document.getElementById('authButton');
  const unlockButton = document.querySelector('.unlock-button');
  if (!locked || !unlocked || !authButton) return;

  locked.style.display = 'flex';
  unlocked.style.display = 'none';
  authButton.textContent = 'Login';
  authButton.onclick = handleAuth;
  if (unlockButton) {
    unlockButton.onclick = handleAuth;
  }
}

function unlockChat() {
  const locked = document.getElementById('lockedInput');
  const unlocked = document.getElementById('unlockedInput');
  const authButton = document.getElementById('authButton');
  const unlockButton = document.querySelector('.unlock-button');
  if (!locked || !unlocked || !authButton) return;

  locked.style.display = 'none';
  unlocked.style.display = 'flex';
  authButton.textContent = 'Logout';
  authButton.onclick = handleAuth;
  if (unlockButton) {
    unlockButton.onclick = handleAuth;
  }

  // Update welcome message
  const chatMessages = document.getElementById('chatMessages');
  if (!chatMessages) return;

  chatMessages.innerHTML = `
        <div class="message agent-message">
            <div class="message-content">
                Welcome back! I'm your navigation assistant. Where would you like to go?
            </div>
        </div>
    `;

  // Notify app.js that auth is ready so conversation sidebar can load.
  // Also set a flag so late listeners can detect that auth is already ready.
  window.navaiAuthReady = true;
  window.dispatchEvent(new CustomEvent('navai-auth-ready'));
}

async function login() {
  if (!auth0Client) {
    await initAuth0();
  }
  if (!auth0Client) {
    console.error('Auth0 client not initialized');
    return;
  }
  await auth0Client.loginWithRedirect();
}

async function logout() {
  if (!auth0Client) {
    await initAuth0();
  }
  if (!auth0Client) {
    console.error('Auth0 client not initialized');
    return;
  }
  await auth0Client.logout({
    logoutParams: {
      returnTo: window.location.origin
    }
  });
  isAuthenticated = false;

  // Clear our own persisted auth state
  try {
    localStorage.removeItem(AUTH_STORAGE_KEYS.token);
    localStorage.removeItem(AUTH_STORAGE_KEYS.isAuthenticated);
    console.debug('[Auth] Cleared localStorage on logout');
  } catch (e) {
    console.warn('[Auth] Unable to clear localStorage on logout:', e);
  }

  lockChat();
}

async function handleAuth() {
  if (!auth0Client) {
    await initAuth0();
  }

  if (!auth0Client) {
    console.error('Auth0 client not initialized');
    return;
  }

  if (isAuthenticated) {
    logout();
  } else {
    login();
  }
}

// Expose helpers to other scripts
window.handleAuth = handleAuth;
window.getAccessToken = function () {
  return accessToken;
};

// Initialize as soon as this script loads
initAuth0();

