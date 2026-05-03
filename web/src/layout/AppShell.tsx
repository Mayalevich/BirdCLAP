import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useAppPreferences } from "@/context/AppPreferences";
import { isApiConfigured, probeBackendReachable } from "@/api/backend";

type ApiBadgeState = "off" | "checking" | "live" | "down";

export function AppShell() {
  const { compareSlots } = useAppPreferences();
  const compareReady = compareSlots[0] && compareSlots[1];
  const configured = isApiConfigured();
  const [apiBadge, setApiBadge] = useState<ApiBadgeState>(() =>
    configured ? "checking" : "off"
  );

  useEffect(() => {
    if (!isApiConfigured()) {
      setApiBadge("off");
      return;
    }
    const ac = new AbortController();
    const timer = window.setTimeout(() => ac.abort(), 8000);
    setApiBadge("checking");
    void probeBackendReachable(ac.signal)
      .then((ok) => setApiBadge(ok ? "live" : "down"))
      .catch(() => setApiBadge("down"));

    return () => {
      window.clearTimeout(timer);
      ac.abort();
    };
  }, []);

  return (
    <div className="app-shell">
      <a href="#main-content" className="skip-link">
        Skip to content
      </a>
      {!configured ? (
        <aside className="app-shell__api-banner app-shell__api-banner--warn" role="status">
          <strong>Backend not connected.</strong> Start the backend server, then refresh this page.
        </aside>
      ) : null}
      {configured && apiBadge === "down" ? (
        <aside className="app-shell__api-banner app-shell__api-banner--error" role="alert">
          <strong>API unreachable.</strong> The backend did not respond. Make sure it is running, then refresh this page.
        </aside>
      ) : null}
      <header className="app-shell__header">
        <div className="app-shell__brand">
          <span className="app-shell__logo" aria-hidden>
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M4 14c4.5-5 10.5-6.5 17-4.5-2 2.5-2.5 6-1 9-4.5-2.5-10-3-14 2a18 18 0 0 1-2-6.5Z"
                fill="currentColor"
                opacity="0.88"
              />
              <path
                d="M14.5 11c3-4 7.5-5.5 11-4"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                opacity="0.8"
              />
            </svg>
          </span>
          <div className="app-shell__brand-text">
            <span className="app-shell__title">Bird audio analysis</span>
            <span className="app-shell__subtitle">Spatiotemporal sound visualization &amp; classification</span>
          </div>
          <span
            className={`app-shell__api-badge app-shell__api-badge--${apiBadge}`}
            title={
              apiBadge === "off"
                ? "VITE_API_BASE_URL missing"
                : apiBadge === "checking"
                  ? "Checking API…"
                  : apiBadge === "live"
                    ? "Backend responded to probe"
                    : "Backend probe failed"
            }
          >
            {apiBadge === "off"
              ? "No API"
              : apiBadge === "checking"
                ? "API…"
                : apiBadge === "live"
                  ? "API OK"
                  : "API down"}
          </span>
        </div>
        <nav className="app-shell__nav" aria-label="Main">
          <NavLink
            to="/"
            end
            className={({ isActive }) => (isActive ? "nav-link nav-link--active" : "nav-link")}
          >
            Overview
          </NavLink>
          <NavLink
            to="/query"
            className={({ isActive }) => (isActive ? "nav-link nav-link--active" : "nav-link")}
          >
            Query
          </NavLink>
          <NavLink
            to="/saved"
            className={({ isActive }) => (isActive ? "nav-link nav-link--active" : "nav-link")}
          >
            Saved
          </NavLink>
          <NavLink
            to="/compare"
            className={({ isActive }) => (isActive ? "nav-link nav-link--active" : "nav-link")}
          >
            Compare
            {compareReady ? (
              <span className="nav-badge" title="Two recordings selected">
                2
              </span>
            ) : null}
          </NavLink>
        </nav>
      </header>
      <main className="app-shell__main" id="main-content">
        <Outlet />
      </main>
      <footer className="app-shell__footer">
        <span className="app-shell__footer-note">
          Each bird verse becomes a 3-D network of data points — a spatial score of frequency, amplitude, and time.
        </span>
      </footer>
    </div>
  );
}
