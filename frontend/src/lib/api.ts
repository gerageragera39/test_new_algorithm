import type {
  AppealAnalyticsResponse,
  AuthorCommentsResponse,
  BanUserResponse,
  BudgetUsageResponse,
  HealthResponse,
  ReportDetailResponse,
  ReportResponse,
  RuntimeSettingsResponse,
  RuntimeSettingsUpdateRequest,
  RunResponse,
  ToxicReviewResponse,
  VideoGuestsResponse,
  VideoStatusRow
} from "../types/api";

/** Options for controlling pipeline run behavior. */
interface RunOptions {
  skipFiltering?: boolean;
}

interface AppealRunOptions {
  guestNames?: string[];
}

/** Custom error class that captures the HTTP status code from a failed API response. */
class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** Parse a fetch Response as JSON, throwing an ApiError on non-OK status codes. */
async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload?.detail) {
        detail = payload.detail;
      }
    } catch {
      // Ignore parse errors and fallback to status text.
    }
    throw new ApiError(response.status, detail || "Request failed");
  }
  return (await response.json()) as T;
}

/** Fetch the backend health status from GET /health. */
export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch("/health");
  return parseResponse<HealthResponse>(response);
}

/** Trigger analysis for the channel's latest video via POST /run/latest. */
export async function runLatest(options?: RunOptions): Promise<RunResponse> {
  const payload: Record<string, unknown> = {};
  if (typeof options?.skipFiltering === "boolean") {
    payload.skip_filtering = options.skipFiltering;
  }
  const hasPayload = Object.keys(payload).length > 0;
  const response = await fetch("/run/latest", {
    method: "POST",
    headers: hasPayload
      ? {
          "Content-Type": "application/json"
        }
      : undefined,
    body: hasPayload ? JSON.stringify(payload) : undefined
  });
  return parseResponse<RunResponse>(response);
}

/** Trigger analysis for a specific video URL via POST /run/video. */
export async function runVideo(videoUrl: string, options?: RunOptions): Promise<RunResponse> {
  const payload: Record<string, unknown> = { video_url: videoUrl };
  if (typeof options?.skipFiltering === "boolean") {
    payload.skip_filtering = options.skipFiltering;
  }
  const response = await fetch("/run/video", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  return parseResponse<RunResponse>(response);
}

/** Fetch the most recent report from GET /reports/latest. */
export async function getLatestReport(): Promise<ReportResponse> {
  const response = await fetch("/reports/latest");
  return parseResponse<ReportResponse>(response);
}

/** Fetch a detailed report for a specific video from GET /reports/:videoId/detail. */
export async function getReportDetail(videoId: string): Promise<ReportDetailResponse> {
  const response = await fetch(`/reports/${videoId}/detail`);
  return parseResponse<ReportDetailResponse>(response);
}

/** Fetch processing statuses for all videos from GET /videos/statuses. */
export async function getVideosStatuses(): Promise<VideoStatusRow[]> {
  const response = await fetch("/videos/statuses");
  return parseResponse<VideoStatusRow[]>(response);
}

/** Fetch today's OpenAI budget usage from GET /budget. */
export async function getBudget(): Promise<BudgetUsageResponse> {
  const response = await fetch("/budget");
  return parseResponse<BudgetUsageResponse>(response);
}

/** Fetch current runtime settings from GET /settings/runtime. */
export async function getRuntimeSettings(): Promise<RuntimeSettingsResponse> {
  const response = await fetch("/settings/runtime");
  return parseResponse<RuntimeSettingsResponse>(response);
}

/** Update runtime settings via PUT /settings/runtime. */
export async function updateRuntimeSettings(
  payload: RuntimeSettingsUpdateRequest
): Promise<RuntimeSettingsResponse> {
  const response = await fetch("/settings/runtime", {
    method: "PUT",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  return parseResponse<RuntimeSettingsResponse>(response);
}

/** Trigger appeal analytics via POST /appeal/run. If no URL given, uses the latest video. */
export async function runAppealAnalytics(
  videoUrl?: string,
  options?: AppealRunOptions
): Promise<RunResponse> {
  const payload: Record<string, unknown> = {};
  if (videoUrl) {
    payload.video_url = videoUrl;
  }
  if (options?.guestNames && options.guestNames.length > 0) {
    payload.guest_names = options.guestNames;
  }
  const response = await fetch("/appeal/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return parseResponse<RunResponse>(response);
}

/** Fetch appeal analytics results for a video from GET /appeal/:videoId. */
export async function getAppealAnalytics(videoId: string): Promise<AppealAnalyticsResponse> {
  const response = await fetch(`/appeal/${videoId}`);
  return parseResponse<AppealAnalyticsResponse>(response);
}

/** Fetch all comments by a specific author under a video. */
export async function getAuthorComments(
  videoId: string,
  authorName: string
): Promise<AuthorCommentsResponse> {
  const response = await fetch(`/appeal/${videoId}/author/${encodeURIComponent(authorName)}`);
  return parseResponse<AuthorCommentsResponse>(response);
}

/** Fetch toxic comments requiring manual review. */
export async function getToxicReview(videoId: string): Promise<ToxicReviewResponse> {
  const response = await fetch(`/appeal/${videoId}/toxic-review`);
  return parseResponse<ToxicReviewResponse>(response);
}

/** Ban a user (manual or auto). */
export async function banUser(
  videoId: string,
  commentId: number,
  authorName: string,
  banReason?: string
): Promise<BanUserResponse> {
  const response = await fetch("/appeal/ban-user", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_id: videoId,
      comment_id: commentId,
      author_name: authorName,
      ban_reason: banReason
    })
  });
  return parseResponse<BanUserResponse>(response);
}

/** Get video guests list. */
export async function getVideoGuests(videoId: string): Promise<VideoGuestsResponse> {
  const response = await fetch(`/settings/video-guests/${videoId}`);
  return parseResponse<VideoGuestsResponse>(response);
}

/** Update video guests list. */
export async function updateVideoGuests(
  videoId: string,
  guestNames: string[]
): Promise<VideoGuestsResponse> {
  const response = await fetch(`/settings/video-guests/${videoId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ guest_names: guestNames })
  });
  return parseResponse<VideoGuestsResponse>(response);
}

export { ApiError };
