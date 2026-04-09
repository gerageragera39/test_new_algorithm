import { Link } from "react-router-dom";

/** 404 page rendered when no route matches. Shows a message and a link back to the home page. */
export function NotFoundPage() {
  return (
    <section className="card">
      <h1>Страница не найдена</h1>
      <p className="muted">Проверьте URL или перейдите на главную.</p>
      <Link to="/" className="btn primary inline-btn">
        На главную
      </Link>
    </section>
  );
}
