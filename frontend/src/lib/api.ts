import type {
  AppealAnalyticsResponse,
  AuthorCommentsResponse,
  BanUserResponse,
  BudgetUsageResponse,
  HealthResponse,
  QueueSnapshotResponse,
  ReportDetailResponse,
  ReportResponse,
  RuntimeSettingsResponse,
  RuntimeSettingsUpdateRequest,
  RunResponse,
  SetupRequest,
  SetupStatusResponse,
  SetupUpdateRequest,
  ToxicReviewResponse,
  UnbanUserResponse,
  VideoGuestsResponse,
  VideoStatusRow
} from "../types/api";

interface RunOptions {
  skipFiltering?: boolean;
}

interface AppealRunOptions {
  guestNames?: string[];
}

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

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

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch("/health");
  return parseResponse<HealthResponse>(response);
}

export async function getSetupStatus(): Promise<SetupStatusResponse> {
  const response = await fetch("/app/setup/status");
  return parseResponse<SetupStatusResponse>(response);
}

export async function completeSetup(payload: SetupRequest): Promise<SetupStatusResponse> {
  const response = await fetch("/app/setup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return parseResponse<SetupStatusResponse>(response);
}

export async function updateSetup(payload: SetupUpdateRequest): Promise<SetupStatusResponse> {
  const response = await fetch("/app/setup", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return parseResponse<SetupStatusResponse>(response);
}

export async function getQueueSnapshot(): Promise<QueueSnapshotResponse> {
  const response = await fetch("/queue");
  return parseResponse<QueueSnapshotResponse>(response);
}

export async function runLatest(options?: RunOptions): Promise<RunResponse> {
  const payload: Record<string, unknown> = {};
  if (typeof options?.skipFiltering === "boolean") {
    payload.skip_filtering = options.skipFiltering;
  }
  const hasPayload = Object.keys(payload).length > 0;
  const response = await fetch("/run/latest", {
    method: "POST",
    headers: hasPayload ? { "Content-Type": "application/json" } : undefined,
    body: hasPayload ? JSON.stringify(payload) : undefined
  });
  return parseResponse<RunResponse>(response);
}

export async function runVideo(videoUrl: string, options?: RunOptions): Promise<RunResponse> {
  const payload: Record<string, unknown> = { video_url: videoUrl };
  if (typeof options?.skipFiltering === "boolean") {
    payload.skip_filtering = options.skipFiltering;
  }
  const response = await fetch("/run/video", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return parseResponse<RunResponse>(response);
}

export async function getLatestReport(): Promise<ReportResponse> {
  const response = await fetch("/reports/latest");
  return parseResponse<ReportResponse>(response);
}

export async function getReportDetail(videoId: string): Promise<ReportDetailResponse> {
  const response = await fetch(`/reports/${videoId}/detail`);
  return parseResponse<ReportDetailResponse>(response);
}

export async function getVideosStatuses(): Promise<VideoStatusRow[]> {
  const response = await fetch("/videos/statuses");
  return parseResponse<VideoStatusRow[]>(response);
}

export async function getBudget(): Promise<BudgetUsageResponse> {
  const response = await fetch("/budget");
  return parseResponse<BudgetUsageResponse>(response);
}

export async function getRuntimeSettings(): Promise<RuntimeSettingsResponse> {
  const response = await fetch("/settings/runtime");
  return parseResponse<RuntimeSettingsResponse>(response);
}

export async function updateRuntimeSettings(
  payload: RuntimeSettingsUpdateRequest
): Promise<RuntimeSettingsResponse> {
  const response = await fetch("/settings/runtime", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return parseResponse<RuntimeSettingsResponse>(response);
}

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

export async function getAppealAnalytics(videoId: string): Promise<AppealAnalyticsResponse> {
  const response = await fetch(`/appeal/${videoId}`);
  return parseResponse<AppealAnalyticsResponse>(response);
}

export async function getAuthorComments(
  videoId: string,
  authorName: string
): Promise<AuthorCommentsResponse> {
  const response = await fetch(`/appeal/${videoId}/author/${encodeURIComponent(authorName)}`);
  return parseResponse<AuthorCommentsResponse>(response);
}

export async function getToxicReview(videoId: string): Promise<ToxicReviewResponse> {
  const response = await fetch(`/appeal/${videoId}/toxic-review`);
  return parseResponse<ToxicReviewResponse>(response);
}

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

export async function unbanUser(bannedUserId: number): Promise<UnbanUserResponse> {
  const response = await fetch("/appeal/unban-user", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ banned_user_id: bannedUserId })
  });
  return parseResponse<UnbanUserResponse>(response);
}

export async function getVideoGuests(videoId: string): Promise<VideoGuestsResponse> {
  const response = await fetch(`/settings/video-guests/${videoId}`);
  return parseResponse<VideoGuestsResponse>(response);
}

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
