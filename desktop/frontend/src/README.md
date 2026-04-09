# `frontend/src/`

Source code for the **YouTube Intel** SPA.

> Last verified against the current router and API client on 2026-04-07.

## Layout

```text
src/
  App.tsx
  main.tsx
  styles.css
  components/
  lib/
  pages/
  types/
```

## Routing

| File | Purpose |
|---|---|
| `main.tsx` | React bootstrap |
| `App.tsx` | router tree with `/ui` basename |

## Components

| File | Purpose |
|---|---|
| `components/AppShell.tsx` | application shell, navigation, and first-run setup form |
| `components/StatusPill.tsx` | status badge for runs and pipeline states |
| `components/ToxicReviewPanel.tsx` | toxic-review queue and moderation actions |

## Pages

| File | Purpose |
|---|---|
| `pages/DashboardPage.tsx` | run controls, health, latest report snapshot, guest-name input |
| `pages/VideosPage.tsx` | video list and progress tracking |
| `pages/ReportPage.tsx` | report details with topic positions and comments |
| `pages/AppealPage.tsx` | appeal blocks, toxic review, manual ban + unban workflow |
| `pages/BudgetPage.tsx` | runtime settings, OAuth/playlist editing, and budget usage |
| `pages/NotFoundPage.tsx` | 404 screen |

## API layer

`lib/api.ts` currently exports:

- `getHealth`
- `runLatest`
- `runVideo`
- `getLatestReport`
- `getReportDetail`
- `getVideosStatuses`
- `getBudget`
- `getRuntimeSettings`
- `updateRuntimeSettings`
- `runAppealAnalytics`
- `getAppealAnalytics`
- `getAuthorComments`
- `getToxicReview`
- `banUser`
- `getVideoGuests`
- `updateVideoGuests`
- `ApiError`

## Types

`types/api.ts` mirrors the backend schemas used by the current UI:

- health and run responses
- report detail payloads
- budget and runtime settings
- appeal analytics blocks
- toxic moderation responses
- guest settings responses
