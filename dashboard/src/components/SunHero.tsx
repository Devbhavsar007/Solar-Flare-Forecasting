import { Satellite } from "lucide-react";

interface Props {
  size?: number;
  showSatellite?: boolean;
}

/**
 * Illustrative animated sun — decorative only.
 *
 * This is deliberately NOT a real per-pixel solar disk image with
 * active-region detection boxes. JWALA's instruments (SoLEXS + HEL1OS)
 * are disk-integrated X-ray spectrometers — they measure total flux,
 * not spatially resolved imagery — so there is no real data pipeline
 * that could draw a box around a specific active region. Faking that
 * would misrepresent what the instruments actually see.
 */
export default function SunHero({ size = 420, showSatellite = true }: Props) {
  const spots = [
    { top: "28%", left: "34%", w: "10%" },
    { top: "52%", left: "58%", w: "7%" },
    { top: "62%", left: "30%", w: "6%" },
    { top: "38%", left: "60%", w: "5%" },
  ];

  return (
    <div className="sun-visual" style={{ maxWidth: size }}>
      <div className="sun-orbit-ring" />
      <div className="sun-orbit-ring inner" />
      {showSatellite && (
        <div className="sun-satellite">
          <Satellite size={16} className="sun-satellite-icon" />
        </div>
      )}
      <div className="sun-core">
        {spots.map((s, i) => (
          <div
            key={i}
            className="sun-spot"
            style={{ top: s.top, left: s.left, width: s.w, aspectRatio: 1 }}
          />
        ))}
      </div>
    </div>
  );
}