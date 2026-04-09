# `frontend/`

React SPA for operating the backend pipelines and reviewing results.

## Stack

- React 18
- TypeScript
- Vite
- React Router (`/ui` basename)

## Development

```bash
cd frontend
npm ci
npm run dev
```

## Production build

```bash
npm run build
```

Build output is generated in `frontend/dist/`.

In the deployed backend:
- the SPA shell is served under **`/ui`**;
- compiled static assets are served under **`/static/app`**.

## SPA routes

| Route | Page | Purpose |
|---|---|---|
| `/ui/` | `DashboardPage` | health, run controls, latest-report summary, appeal-trigger controls |
| `/ui/videos` | `VideosPage` | per-video status monitor and navigation |
| `/ui/budget` | `BudgetPage` | runtime settings and usage view |
| `/ui/reports/:videoId` | `ReportPage` | topic report detail |
| `/ui/appeal/:videoId` | `AppealPage` | appeal analytics, toxic review, guest workflow |
| `/ui/404` | `NotFoundPage` | fallback route |

For source-level details, see [`frontend/src/README.md`](./src/README.md).
