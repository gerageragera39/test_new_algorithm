/** Props for the StatusPill component. */
interface StatusPillProps {
  status: string | null;
  label: string;
}

/** Renders a colored pill badge reflecting a pipeline run status (running, failed, completed, or neutral). */
export function StatusPill({ status, label }: StatusPillProps) {
  let variant = "neutral";
  if (status === "running") {
    variant = "running";
  } else if (status === "failed") {
    variant = "failed";
  } else if (status === "completed") {
    variant = "completed";
  }

  return <span className={`status-pill ${variant}`}>{label || "-"}</span>;
}
