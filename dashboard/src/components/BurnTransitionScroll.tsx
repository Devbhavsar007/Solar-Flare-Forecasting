import {
    startTransition,
    useEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import type { CSSProperties, RefObject } from "react";

/**
 * WebGL burn-transition overlay.
 *
 * `progress` (0→1) controls how far the burn has advanced:
 *   - 0 = fully transparent (nothing burned yet)
 *   - 1 = fully opaque fill color (everything burned/covered)
 *
 * The parent is responsible for deriving `progress` from scroll position
 * (or any other signal). This component does NOT use useScroll internally
 * because its layout position can be shifted by CSS transforms (e.g.
 * translateY(-100%)), which makes Framer Motion's layout-based scroll
 * tracking report wrong values.
 */

interface BurnTransitionProps {
    /** Ref holding 0→1 burn progress, written by the parent's scroll handler.
     *  Using a ref instead of a prop avoids React re-renders on every scroll
     *  frame — the WebGL loop reads it directly at 60fps with zero latency. */
    progressRef: RefObject<number>;
    fillColor: string;
    emberColor: string;
    glowColor: string;
    edgeWidth: number;
    noiseScale: number;
    flicker: number;
    style?: CSSProperties;
}

function parseColorToRGB(color: string): [number, number, number] {
    if (!color || color.trim() === "") return [0, 0, 0];
    const str = color.trim();

    const rgbaMatch = str.match(/rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)/i);
    if (rgbaMatch) {
        return [
            Math.max(0, Math.min(255, parseFloat(rgbaMatch[1]))) / 255,
            Math.max(0, Math.min(255, parseFloat(rgbaMatch[2]))) / 255,
            Math.max(0, Math.min(255, parseFloat(rgbaMatch[3]))) / 255
        ];
    }

    const hex = str.replace(/^#/, "");
    if (hex.length === 6 || hex.length === 8) {
        return [
            parseInt(hex.slice(0, 2), 16) / 255,
            parseInt(hex.slice(2, 4), 16) / 255,
            parseInt(hex.slice(4, 6), 16) / 255
        ];
    }
    if (hex.length === 3) {
        return [
            parseInt(hex[0] + hex[0], 16) / 255,
            parseInt(hex[1] + hex[1], 16) / 255,
            parseInt(hex[2] + hex[2], 16) / 255
        ];
    }

    if (typeof window !== "undefined" && typeof document !== "undefined") {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        if (ctx) {
            ctx.fillStyle = color;
            const normalized = ctx.fillStyle;
            const cleanHex = normalized.replace(/^#/, "");
            if (cleanHex.length === 6) {
                return [
                    parseInt(cleanHex.slice(0, 2), 16) / 255,
                    parseInt(cleanHex.slice(2, 4), 16) / 255,
                    parseInt(cleanHex.slice(4, 6), 16) / 255
                ];
            }
        }
    }
    return [1, 1, 1];
}

export default function BurnTransitionScroll(props: BurnTransitionProps) {
    const {
        progressRef: externalProgressRef,
        fillColor,
        emberColor,
        glowColor,
        edgeWidth,
        noiseScale,
        flicker,
        style,
    } = props;

    const containerRef = useRef<HTMLDivElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const frameRef = useRef<number>(0);
    const observerRef = useRef<ResizeObserver | null>(null);
    const [hasWebGL, setHasWebGL] = useState(true);

    const fillRGB = useMemo(() => parseColorToRGB(fillColor), [fillColor]);
    const emberRGB = useMemo(() => parseColorToRGB(emberColor), [emberColor]);
    const glowRGB = useMemo(() => parseColorToRGB(glowColor), [glowColor]);

    const fragmentShader = useMemo(
        () => `
precision mediump float;

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_progress;
uniform float u_edgeWidth;
uniform float u_noiseScale;
uniform float u_flicker;
uniform vec3 u_fillColor;
uniform vec3 u_emberColor;
uniform vec3 u_glowColor;

// Hash function used by noise.
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// Generate a random gradient vector on the unit circle
vec2 grad(vec2 p) {
    float h = hash(p) * 6.2831853; // 2 * pi
    return vec2(cos(h), sin(h));
}

// 2D Perlin (gradient) noise
float perlinNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0); // quintic curve

    float a = dot(grad(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0));
    float b = dot(grad(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
    float c = dot(grad(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
    float d = dot(grad(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0));

    float n = mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    return n * 0.5 + 0.5; // Map from [-0.5, 0.5] to [0.0, 1.0]
}

// Fractal Brownian Motion (fbm): layered Perlin noise with coordinate rotation
float fbm(vec2 p) {
    float total = 0.0;
    float amplitude = 0.5;
    mat2 m2 = mat2(0.8, 0.6, -0.6, 0.8);
    p = m2 * p; // rotate initial coordinate to prevent axis alignment
    for (int i = 0; i < 5; i++) {
        total += perlinNoise(p) * amplitude;
        p = m2 * p * 2.02;
        amplitude *= 0.5;
    }
    return total;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    vec2 centered = (uv - 0.5) * vec2(u_resolution.x / u_resolution.y, 1.0);

    // Animated noise field used as the burn map.
    float burnNoise = fbm(centered * u_noiseScale + vec2(0.0, u_time * 0.12));

    // Core burn logic:
    // - if burnNoise < progress: pixel is burned away => transparent.
    // - if burnNoise just above progress: render ember band (glowing edge).
    // - otherwise: keep un-burned fill color.
    float threshBottom = u_progress * 4.2 - 3.1;
    float threshTop = u_progress * 1.5 - 0.2;
    float threshold = mix(threshBottom, threshTop, uv.y);
    float distToEdge = burnNoise - threshold;

    vec3 color = u_fillColor;
    float alpha = 1.0;

    if (distToEdge < 0.0) {
        alpha = 0.0;
    } else if (distToEdge < u_edgeWidth) {
        float edgeT = 1.0 - clamp(distToEdge / u_edgeWidth, 0.0, 1.0);
        float spark = (sin(u_time * 35.0 + uv.y * 80.0) + sin(u_time * 22.0 + uv.x * 55.0)) * 0.5;
        float flicker = 1.0 + spark * u_flicker * 0.35;
        vec3 hot = u_emberColor * (1.1 * flicker);
        vec3 warm = u_glowColor * (0.9 + 0.2 * flicker);
        color = mix(warm, hot, pow(edgeT, 0.45));
        alpha = mix(0.7, 1.0, edgeT);
    }

    gl_FragColor = vec4(color, alpha);
}
`,
        []
    );

    const vertexShader = useMemo(
        () => `
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`,
        []
    );

    useEffect(() => {
        if (typeof window === "undefined" || typeof document === "undefined")
            return;
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container) return;

        const gl = canvas.getContext("webgl", {
            alpha: true,
            premultipliedAlpha: false,
        });
        if (!gl) {
            startTransition(() => setHasWebGL(false));
            return;
        }
        startTransition(() => setHasWebGL(true));

        const compile = (type: number, source: string): WebGLShader | null => {
            const shader = gl.createShader(type);
            if (!shader) return null;
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                console.warn("Shader compile error:", gl.getShaderInfoLog(shader));
                gl.deleteShader(shader);
                return null;
            }
            return shader;
        };

        const vs = compile(gl.VERTEX_SHADER, vertexShader);
        const fs = compile(gl.FRAGMENT_SHADER, fragmentShader);
        if (!vs || !fs) {
            startTransition(() => setHasWebGL(false));
            return;
        }

        const program = gl.createProgram();
        if (!program) {
            startTransition(() => setHasWebGL(false));
            return;
        }
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            startTransition(() => setHasWebGL(false));
            return;
        }

        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(
            gl.ARRAY_BUFFER,
            new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
            gl.STATIC_DRAW
        );

        gl.useProgram(program);
        const position = gl.getAttribLocation(program, "a_position");
        gl.enableVertexAttribArray(position);
        gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

        const uResolution = gl.getUniformLocation(program, "u_resolution");
        const uTime = gl.getUniformLocation(program, "u_time");
        const uProgress = gl.getUniformLocation(program, "u_progress");
        const uEdgeWidth = gl.getUniformLocation(program, "u_edgeWidth");
        const uNoiseScale = gl.getUniformLocation(program, "u_noiseScale");
        const uFlicker = gl.getUniformLocation(program, "u_flicker");
        const uFillColor = gl.getUniformLocation(program, "u_fillColor");
        const uEmberColor = gl.getUniformLocation(program, "u_emberColor");
        const uGlowColor = gl.getUniformLocation(program, "u_glowColor");

        const resize = () => {
            const rect = container.getBoundingClientRect();
            const dpr = Math.max(1, window.devicePixelRatio || 1);
            const width = Math.max(1, Math.round(rect.width * dpr));
            const height = Math.max(1, Math.round(rect.height * dpr));
            if (canvas.width !== width || canvas.height !== height) {
                canvas.width = width;
                canvas.height = height;
            }
            gl.viewport(0, 0, canvas.width, canvas.height);
        };

        resize();
        observerRef.current = new ResizeObserver(() => resize());
        observerRef.current.observe(container);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        const startTime = performance.now();
        const loop = () => {
            frameRef.current = window.requestAnimationFrame(loop);

            const t = (performance.now() - startTime) / 1000;
            const p = externalProgressRef.current ?? 0;

            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.useProgram(program);
            gl.uniform2f(uResolution, canvas.width, canvas.height);
            gl.uniform1f(uTime, t);
            // The shader treats progress=0 as "fully solid fill" and
            // progress=1 as "fully burned away (transparent)".
            // Our prop is the opposite: 0 = transparent, 1 = opaque.
            // Invert it so the visual behavior matches expectations.
            gl.uniform1f(uProgress, 1.0 - p);
            gl.uniform1f(uEdgeWidth, edgeWidth);
            gl.uniform1f(uNoiseScale, noiseScale);
            gl.uniform1f(uFlicker, flicker);
            gl.uniform3f(uFillColor, fillRGB[0], fillRGB[1], fillRGB[2]);
            gl.uniform3f(uEmberColor, emberRGB[0], emberRGB[1], emberRGB[2]);
            gl.uniform3f(uGlowColor, glowRGB[0], glowRGB[1], glowRGB[2]);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        };

        loop();

        return () => {
            if (frameRef.current) window.cancelAnimationFrame(frameRef.current);
            observerRef.current?.disconnect();
            gl.deleteBuffer(positionBuffer);
            gl.deleteProgram(program);
            gl.deleteShader(vs);
            gl.deleteShader(fs);
        };
    }, [
        fragmentShader,
        vertexShader,
        edgeWidth,
        noiseScale,
        flicker,
        fillRGB,
        emberRGB,
        glowRGB,
    ]);

    return (
        <div
            ref={containerRef}
            style={{
                position: "relative",
                width: style?.width ?? 400,
                height: style?.height ?? 400,
                inset: 0,
                pointerEvents: "none",
                ...style,
            }}
        >
            <canvas
                ref={canvasRef}
                style={{
                    position: "absolute",
                    inset: 0,
                    width: "100%",
                    height: "100%",
                    display: "block",
                    zIndex: 1,
                }}
            />
            <div
                aria-hidden
                style={{
                    position: "absolute",
                    inset: 0,
                    background: fillColor,
                    opacity: hasWebGL ? 0 : (externalProgressRef.current ?? 0),
                    zIndex: 0,
                }}
            />
        </div>
    );
}