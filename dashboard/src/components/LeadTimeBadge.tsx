import { Timer } from "lucide-react";

interface Props {
  minutes: number | null;
}

export default function LeadTimeBadge({ minutes }: Props) {
  if (minutes === null || minutes === undefined) {
    return (
      <div className="lead-time-badge">
        <Timer size={16} />
        Lead Time: <span className="value">--</span> min
      </div>
    );
  }

  return (
    <div className="lead-time-badge">
      <Timer size={16} />
      Lead Time: <span className="value">{minutes.toFixed(0)}</span> min
    </div>
  );
}
