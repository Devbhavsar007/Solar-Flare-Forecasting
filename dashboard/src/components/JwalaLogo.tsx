
interface JwalaLogoProps {
  size?: number;
  className?: string;
  glow?: boolean;
}

export default function JwalaLogo({ size = 24, className = "", glow = true }: JwalaLogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      style={{ display: "inline-block", verticalAlign: "middle" }}
    >
      <defs>
        {/* Custom premium flame gradient */}
        <linearGradient id="jwala-logo-gradient" x1="0%" y1="100%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#ef4444" />
          <stop offset="50%" stopColor="#f59e0b" />
          <stop offset="100%" stopColor="#F6D337" />
        </linearGradient>

        {/* Outer ambient glow for a modern glassmorphic look */}
        {glow && (
          <filter id="jwala-logo-glow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="1.8" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        )}
      </defs>

      {/* Ambient background glow path */}
      {glow && (
        <path
          d="M12 2C7.5 7.5 4 11.5 4 15.5C4 19.5 7.5 22.5 12 22.5C16.5 22.5 20 19.5 20 15.5C20 11.5 16.5 7.5 12 2Z"
          fill="url(#jwala-logo-gradient)"
          opacity="0.25"
          filter="url(#jwala-logo-glow)"
        />
      )}

      {/* Outer Flame Ring / Solar Shield */}
      <path
        d="M12 5.5C14.5 9 17.5 12.5 17.5 16.5C17.5 19.5 15 21.5 12 21.5C9 21.5 6.5 19.5 6.5 16.5C6.5 12.5 9.5 9 12 5.5Z"
        stroke="url(#jwala-logo-gradient)"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity="0.85"
      />

      {/* Inner Flame Core (The Solar Flare) */}
      <path
        d="M12 2.5C12 2.5 15 7.5 15 11C15 13.5 13.5 15.2 12 15.2C10.5 15.2 9 13.5 9 11C9 7.5 12 2.5 12 2.5Z"
        fill="url(#jwala-logo-gradient)"
      />

      {/* Micro-spark/solar-burst highlight */}
      <circle cx="12" cy="11" r="1" fill="#FFFFFF" opacity="0.9" />
    </svg>
  );
}
