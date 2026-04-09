import { NavLink, Outlet } from "react-router-dom";

/** Top-level layout shell with the navigation header and a routed content outlet. */
export function AppShell() {
  return (
    <div className="app-root">
      <div className="bg-ambient" aria-hidden="true" />
      <header className="topbar">
        <div className="container topbar-inner">
          <NavLink to="/" className="brand">
            <span className="brand-mark" aria-hidden="true" />
            <span className="brand-text">
              <span className="brand-title">YouTube Intel</span>
              <span className="brand-subtitle">Made by Herman Dihtenko</span>
            </span>
          </NavLink>
          <nav className="topnav" aria-label="Навигация">
            <NavLink to="/" end>
              Главная
            </NavLink>
            <NavLink to="/videos">Видео</NavLink>
            <NavLink to="/budget">Dashboard</NavLink>
          </nav>
        </div>
      </header>

      <main className="page">
        <div className="container">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

