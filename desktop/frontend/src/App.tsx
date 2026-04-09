/**
 * @file Application router configuration.
 * Defines all client-side routes under the /ui basename using react-router-dom.
 */
import { Navigate, createBrowserRouter } from "react-router-dom";

import { AppShell } from "./components/AppShell";
import { AppealPage } from "./pages/AppealPage";
import { BudgetPage } from "./pages/BudgetPage";
import { DashboardPage } from "./pages/DashboardPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { ReportPage } from "./pages/ReportPage";
import { VideosPage } from "./pages/VideosPage";

export const router = createBrowserRouter(
  [
    {
      path: "/",
      element: <AppShell />,
      children: [
        {
          index: true,
          element: <DashboardPage />
        },
        {
          path: "videos",
          element: <VideosPage />
        },
        {
          path: "budget",
          element: <BudgetPage />
        },
        {
          path: "reports/:videoId",
          element: <ReportPage />
        },
        {
          path: "appeal/:videoId",
          element: <AppealPage />
        },
        {
          path: "404",
          element: <NotFoundPage />
        },
        {
          path: "*",
          element: <Navigate to="/404" replace />
        }
      ]
    }
  ],
  {
    basename: "/ui"
  }
);
