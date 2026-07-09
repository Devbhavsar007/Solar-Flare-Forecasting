import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
  Legend,
} from "recharts";
import type { FluxDataPoint } from "../types/api";

interface Props {
  data: FluxDataPoint[];
  showUncertainty?: boolean;
}

export default function FluxChart({ data, showUncertainty = true }: Props) {
  return (
    <div className="card chart-card">
      <div className="card-title">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" />
        </svg>
        Real-Time Flux Monitor
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="timestamp"
            tick={{ fill: "#5a6480", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
          />
          <YAxis
            yAxisId="sxr"
            tick={{ fill: "#5a6480", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => v.toExponential(0)}
            label={{
              value: "SXR (W/m²)",
              angle: -90,
              position: "insideLeft",
              fill: "#5a6480",
              fontSize: 10,
            }}
          />
          <YAxis
            yAxisId="hxr"
            orientation="right"
            tick={{ fill: "#5a6480", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            label={{
              value: "HXR (counts)",
              angle: 90,
              position: "insideRight",
              fill: "#5a6480",
              fontSize: 10,
            }}
          />
          <Tooltip
            contentStyle={{
              background: "#1a1f35",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 8,
              color: "#e8ecf4",
              fontSize: 12,
            }}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, color: "#8b95b0" }}
          />

          {showUncertainty && (
            <Area
              yAxisId="sxr"
              dataKey="q90"
              stroke="none"
              fill="rgba(246, 211, 55, 0.08)"
              name="q90 bound"
            />
          )}
          {showUncertainty && (
            <Area
              yAxisId="sxr"
              dataKey="q10"
              stroke="none"
              fill="rgba(246, 211, 55, 0.08)"
              name="q10 bound"
            />
          )}

          <Line
            yAxisId="sxr"
            type="monotone"
            dataKey="sxr"
            stroke="#F6D337"
            strokeWidth={2}
            dot={false}
            name="SXR Flux"
          />
          <Line
            yAxisId="hxr"
            type="monotone"
            dataKey="hxr"
            stroke="#f97316"
            strokeWidth={1.5}
            dot={false}
            name="HXR Counts"
            strokeDasharray="4 2"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
