import { useEffect, useRef } from "react";

// WebGL Color parsing and helper functions
const cssVariableRegex = /var\s*\(\s*(--[\w-]+)(?:\s*,\s*((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*))?\s*\)/;

function extractDefaultValue(cssVar: string): string {
  if (!cssVar || !cssVar.startsWith("var(")) return cssVar;
  const match = cssVariableRegex.exec(cssVar);
  if (!match) return cssVar;
  const fallback = (match[2] || "").trim();
  if (fallback.startsWith("var(")) return extractDefaultValue(fallback);
  return fallback || cssVar;
}

function resolveTokenColor(input: string): string {
  if (typeof input !== "string") return input;
  if (!input.startsWith("var(")) return input;
  return extractDefaultValue(input);
}

function parseColorToRgba(input: string): { r: number; g: number; b: number; a: number } {
  if (!input || input.trim() === "") return { r: 0, g: 0, b: 0, a: 1 };
  const str = input.trim();
  
  const rgbaMatch = str.match(/rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)/i);
  if (rgbaMatch) {
    const r = Math.max(0, Math.min(255, parseFloat(rgbaMatch[1]))) / 255;
    const g = Math.max(0, Math.min(255, parseFloat(rgbaMatch[2]))) / 255;
    const b = Math.max(0, Math.min(255, parseFloat(rgbaMatch[3]))) / 255;
    const a = rgbaMatch[4] !== undefined ? Math.max(0, Math.min(1, parseFloat(rgbaMatch[4]))) : 1;
    return { r, g, b, a };
  }
  
  const hex = str.replace(/^#/, "");
  if (hex.length === 8) {
    return {
      r: parseInt(hex.slice(0, 2), 16) / 255,
      g: parseInt(hex.slice(2, 4), 16) / 255,
      b: parseInt(hex.slice(4, 6), 16) / 255,
      a: parseInt(hex.slice(6, 8), 16) / 255,
    };
  }
  if (hex.length === 6) {
    return {
      r: parseInt(hex.slice(0, 2), 16) / 255,
      g: parseInt(hex.slice(2, 4), 16) / 255,
      b: parseInt(hex.slice(4, 6), 16) / 255,
      a: 1,
    };
  }
  if (hex.length === 3) {
    return {
      r: parseInt(hex[0] + hex[0], 16) / 255,
      g: parseInt(hex[1] + hex[1], 16) / 255,
      b: parseInt(hex[2] + hex[2], 16) / 255,
      a: 1,
    };
  }
  return { r: 0, g: 0, b: 0, a: 1 };
}

function mapLinear(value: number, inMin: number, inMax: number, outMin: number, outMax: number): number {
  if (inMax === inMin) return outMin;
  const t = (value - inMin) / (inMax - inMin);
  return outMin + t * (outMax - outMin);
}

function mapNoiseScale(ui: number) { return mapLinear(Math.max(0, Math.min(1, ui)), 0, 1, 1, 20); }
function mapNoiseIntensity(ui: number) { return mapLinear(Math.max(0, Math.min(1, ui)), 0, 1, 0, 0.5); }
function mapScrollSensitivity(ui: number) { return mapLinear(Math.max(0, Math.min(1, ui)), 0, 1, 0, 0.01); }
function mapBaseAnimationSpeed(ui: number) { return mapLinear(Math.max(0, Math.min(1, ui)), 0, 1, 0, 0.1); }
function mapEdgeSoftness(ui: number) { return mapLinear(Math.max(0, Math.min(1, ui)), 0, 1, 0.01, 0.2); }
function mapBloomRadius(ui: number) { return mapLinear(Math.max(0, Math.min(1, ui)), 0, 1, 0, 0.3); }

interface WebGLBurnTransitionProps {
  color?: string;
  transitionColor?: string;
  noiseScale?: number;
  noiseIntensity?: number;
  scrollSensitivity?: number;
  baseAnimationSpeed?: number;
  edgeSoftness?: number;
  bloomIntensity?: number;
  bloomRadius?: number;
  parallaxEnabled?: boolean;
  style?: React.CSSProperties;
  movement?: {
    horizontal: "left" | "center" | "right";
    vertical: number;
  };
}

export default function WebGLBurnTransition({
  color = "#0d0f12",
  transitionColor = "#F6D337",
  noiseScale = 0.37,
  noiseIntensity = 0.3,
  scrollSensitivity = 0.4,
  baseAnimationSpeed = 0.1,
  edgeSoftness = 0.4,
  bloomIntensity = 0.5,
  bloomRadius = 0.1,
  parallaxEnabled = true,
  style,
  movement = { horizontal: "center", vertical: 0.5 }
}: WebGLBurnTransitionProps) {
  
  const internalNoiseScale = mapNoiseScale(noiseScale);
  const internalNoiseIntensity = mapNoiseIntensity(noiseIntensity);
  const internalScrollSensitivity = mapScrollSensitivity(scrollSensitivity);
  const internalBaseAnimationSpeed = mapBaseAnimationSpeed(baseAnimationSpeed);
  const internalEdgeSoftness = mapEdgeSoftness(edgeSoftness);
  const internalGrainScale = 0;
  const internalBloomRadius = mapBloomRadius(bloomRadius);

  const horizontalMovementValue = movement?.horizontal === "left" ? 1 : movement?.horizontal === "right" ? -1 : 0;

  const resolvedColor = resolveTokenColor(color);
  const colorRgba = parseColorToRgba(resolvedColor);

  const resolvedTransitionColor = transitionColor ? resolveTokenColor(transitionColor) : resolvedColor;
  const transitionColorRgba = parseColorToRgba(resolvedTransitionColor);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const bufferRef = useRef<WebGLBuffer | null>(null);
  
  const colorRef = useRef([colorRgba.r, colorRgba.g, colorRgba.b]);
  const transitionColorRef = useRef([transitionColorRgba.r, transitionColorRgba.g, transitionColorRgba.b]);
  const noiseScaleRef = useRef(internalNoiseScale);
  const noiseIntensityRef = useRef(internalNoiseIntensity);
  const scrollSensitivityRef = useRef(internalScrollSensitivity);
  const baseAnimationSpeedRef = useRef(internalBaseAnimationSpeed);
  const edgeSoftnessRef = useRef(internalEdgeSoftness);
  const grainScaleRef = useRef(internalGrainScale);
  const movementHorizontalRef = useRef(horizontalMovementValue);
  const movementVerticalRef = useRef(movement?.vertical ?? 0.5);
  
  const scrollOffsetRef = useRef(0);
  const lastScrollYRef = useRef(0);
  const lastScrollTimeRef = useRef(0);
  const scrollVelocityRef = useRef(0);
  const animationFrameRef = useRef<number | null>(null);
  const baseTimeRef = useRef(0);
  const startTimeRef = useRef(0);
  const parallaxEnabledRef = useRef(parallaxEnabled);
  
  const parallaxStartRef = useRef(0);
  const parallaxEndRef = useRef(100);
  const parallaxOffsetRef = useRef(0);
  const canvasSizeRef = useRef({ width: 0, height: 0 });

  // Bloom refs
  const bloomIntensityRef = useRef(bloomIntensity);
  const bloomRadiusRef = useRef(internalBloomRadius);
  const extractProgramRef = useRef<WebGLProgram | null>(null);
  const blurProgramRef = useRef<WebGLProgram | null>(null);
  const compositeProgramRef = useRef<WebGLProgram | null>(null);
  
  const framebufferRef = useRef<WebGLFramebuffer | null>(null);
  const sceneTextureRef = useRef<WebGLTexture | null>(null);
  const extractFramebufferRef = useRef<WebGLFramebuffer | null>(null);
  const extractTextureRef = useRef<WebGLTexture | null>(null);
  const blurFramebuffer1Ref = useRef<WebGLFramebuffer | null>(null);
  const blurTexture1Ref = useRef<WebGLTexture | null>(null);
  const blurFramebuffer2Ref = useRef<WebGLFramebuffer | null>(null);
  const blurTexture2Ref = useRef<WebGLTexture | null>(null);
  const bloomDownsampleRef = useRef(2);

  const createShader = (gl: WebGLRenderingContext, type: number, source: string) => {
    const shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error("Shader compilation error:", gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  };

  const createProgram = (gl: WebGLRenderingContext, vertexShader: WebGLShader, fragmentShader: WebGLShader) => {
    const program = gl.createProgram();
    if (!program) return null;
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("Program linking error:", gl.getProgramInfoLog(program));
      gl.deleteProgram(program);
      return null;
    }
    return program;
  };

  const createFramebufferTexture = (gl: WebGLRenderingContext, width: number, height: number) => {
    const texture = gl.createTexture();
    if (!texture) return { framebuffer: null, texture: null };
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    
    const framebuffer = gl.createFramebuffer();
    if (!framebuffer) return { framebuffer: null, texture };
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { framebuffer, texture };
  };

  const resizeCanvas = () => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const gl = glRef.current;
    if (!canvas || !container || !gl) return;
    const rect = container.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const newWidth = Math.floor(rect.width * dpr);
    const newHeight = Math.floor(rect.height * dpr);

    if (canvas.width === newWidth && canvas.height === newHeight) return;
    canvas.width = newWidth;
    canvas.height = newHeight;
    canvasSizeRef.current = { width: newWidth, height: newHeight };
    
    gl.viewport(0, 0, canvas.width, canvas.height);
    const downsample = bloomDownsampleRef.current;
    const bloomWidth = Math.floor(newWidth / downsample);
    const bloomHeight = Math.floor(newHeight / downsample);

    if (sceneTextureRef.current) {
      gl.bindTexture(gl.TEXTURE_2D, sceneTextureRef.current);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, newWidth, newHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    if (extractTextureRef.current) {
      gl.bindTexture(gl.TEXTURE_2D, extractTextureRef.current);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, bloomWidth, bloomHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    if (blurTexture1Ref.current) {
      gl.bindTexture(gl.TEXTURE_2D, blurTexture1Ref.current);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, bloomWidth, bloomHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    if (blurTexture2Ref.current) {
      gl.bindTexture(gl.TEXTURE_2D, blurTexture2Ref.current);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, bloomWidth, bloomHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
  };

  const renderScene = (targetFramebuffer: WebGLFramebuffer | null) => {
    const gl = glRef.current;
    const program = programRef.current;
    const buffer = bufferRef.current;
    if (!gl || !program || !buffer) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFramebuffer);
    gl.viewport(0, 0, canvasSizeRef.current.width, canvasSizeRef.current.height);
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    
    const positionLocation = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    const colorLocation = gl.getUniformLocation(program, "u_color");
    const [r, g, b] = colorRef.current;
    gl.uniform3f(colorLocation, r, g, b);

    const transitionColorLocation = gl.getUniformLocation(program, "u_transition_color");
    if (transitionColorLocation) {
      const [tr, tg, tb] = transitionColorRef.current;
      gl.uniform3f(transitionColorLocation, tr, tg, tb);
    }

    const noiseScaleLocation = gl.getUniformLocation(program, "u_noise_scale");
    if (noiseScaleLocation) {
      gl.uniform1f(noiseScaleLocation, noiseScaleRef.current);
    }
    const noiseIntensityLocation = gl.getUniformLocation(program, "u_noise_intensity");
    if (noiseIntensityLocation) {
      gl.uniform1f(noiseIntensityLocation, noiseIntensityRef.current);
    }

    const currentTime = performance.now();
    if (startTimeRef.current === 0) {
      startTimeRef.current = currentTime;
    }
    
    const elapsedSeconds = (currentTime - startTimeRef.current) / 1000;
    baseTimeRef.current = elapsedSeconds * baseAnimationSpeedRef.current;

    const scrollOffsetLocation = gl.getUniformLocation(program, "u_scroll_offset");
    if (scrollOffsetLocation) {
      gl.uniform1f(scrollOffsetLocation, baseTimeRef.current + scrollOffsetRef.current);
    }

    const edgeSoftnessLocation = gl.getUniformLocation(program, "u_edge_softness");
    if (edgeSoftnessLocation) {
      gl.uniform1f(edgeSoftnessLocation, edgeSoftnessRef.current);
    }

    const grainScaleLocation = gl.getUniformLocation(program, "u_grain_scale");
    if (grainScaleLocation) {
      gl.uniform1f(grainScaleLocation, grainScaleRef.current);
    }

    const movementHorizontalLocation = gl.getUniformLocation(program, "u_movement_horizontal");
    if (movementHorizontalLocation) {
      gl.uniform1f(movementHorizontalLocation, movementHorizontalRef.current);
    }
    const movementVerticalLocation = gl.getUniformLocation(program, "u_movement_vertical");
    if (movementVerticalLocation) {
      gl.uniform1f(movementVerticalLocation, movementVerticalRef.current);
    }

    const parallaxOffsetLocation = gl.getUniformLocation(program, "u_parallax_offset");
    if (parallaxOffsetLocation) {
      gl.uniform1f(parallaxOffsetLocation, parallaxOffsetRef.current);
    }

    const aspectRatioLocation = gl.getUniformLocation(program, "u_aspect_ratio");
    if (aspectRatioLocation) {
      const width = canvasSizeRef.current.width;
      const height = canvasSizeRef.current.height;
      const aspectRatio = height > 0 ? width / height : 1;
      gl.uniform1f(aspectRatioLocation, aspectRatio);
    }

    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  const renderExtract = (sourceTexture: WebGLTexture, targetFramebuffer: WebGLFramebuffer | null) => {
    const gl = glRef.current;
    const extractProgram = extractProgramRef.current;
    const buffer = bufferRef.current;
    if (!gl || !extractProgram || !buffer) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFramebuffer);
    
    const downsample = bloomDownsampleRef.current;
    const bloomWidth = Math.floor(canvasSizeRef.current.width / downsample);
    const bloomHeight = Math.floor(canvasSizeRef.current.height / downsample);
    gl.viewport(0, 0, bloomWidth, bloomHeight);
    gl.useProgram(extractProgram);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    
    const positionLocation = gl.getAttribLocation(extractProgram, "a_position");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
    const textureLocation = gl.getUniformLocation(extractProgram, "u_texture");
    gl.uniform1i(textureLocation, 0);

    const transitionColorLocation = gl.getUniformLocation(extractProgram, "u_transition_color");
    if (transitionColorLocation) {
      const [tr, tg, tb] = transitionColorRef.current;
      gl.uniform3f(transitionColorLocation, tr, tg, tb);
    }

    const baseColorLocation = gl.getUniformLocation(extractProgram, "u_base_color");
    if (baseColorLocation) {
      const [r, g, b] = colorRef.current;
      gl.uniform3f(baseColorLocation, r, g, b);
    }
    
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.BLEND);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  const renderBlur = (sourceTexture: WebGLTexture, targetFramebuffer: WebGLFramebuffer | null, direction: number[]) => {
    const gl = glRef.current;
    const blurProgram = blurProgramRef.current;
    const buffer = bufferRef.current;
    if (!gl || !blurProgram || !buffer) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFramebuffer);
    
    const downsample = bloomDownsampleRef.current;
    const bloomWidth = Math.floor(canvasSizeRef.current.width / downsample);
    const bloomHeight = Math.floor(canvasSizeRef.current.height / downsample);
    gl.viewport(0, 0, bloomWidth, bloomHeight);
    gl.useProgram(blurProgram);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    
    const positionLocation = gl.getAttribLocation(blurProgram, "a_position");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
    const textureLocation = gl.getUniformLocation(blurProgram, "u_texture");
    gl.uniform1i(textureLocation, 0);

    const directionLocation = gl.getUniformLocation(blurProgram, "u_direction");
    gl.uniform2f(directionLocation, direction[0], direction[1]);

    const resolutionLocation = gl.getUniformLocation(blurProgram, "u_resolution");
    gl.uniform2f(resolutionLocation, bloomWidth, bloomHeight);

    const radiusLocation = gl.getUniformLocation(blurProgram, "u_radius");
    gl.uniform1f(radiusLocation, bloomRadiusRef.current);
    
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.BLEND);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  const renderComposite = (sceneTexture: WebGLTexture, bloomTexture: WebGLTexture) => {
    const gl = glRef.current;
    const compositeProgram = compositeProgramRef.current;
    const buffer = bufferRef.current;
    if (!gl || !compositeProgram || !buffer) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasSizeRef.current.width, canvasSizeRef.current.height);
    gl.useProgram(compositeProgram);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    
    const positionLocation = gl.getAttribLocation(compositeProgram, "a_position");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
    const sceneLocation = gl.getUniformLocation(compositeProgram, "u_scene");
    gl.uniform1i(sceneLocation, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, bloomTexture);
    const bloomLocation = gl.getUniformLocation(compositeProgram, "u_bloom");
    gl.uniform1i(bloomLocation, 1);

    const intensityLocation = gl.getUniformLocation(compositeProgram, "u_bloom_intensity");
    gl.uniform1f(intensityLocation, bloomIntensityRef.current);

    const transitionColorLocation = gl.getUniformLocation(compositeProgram, "u_transition_color");
    if (transitionColorLocation) {
      const [tr, tg, tb] = transitionColorRef.current;
      gl.uniform3f(transitionColorLocation, tr, tg, tb);
    }
    
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.BLEND);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  };

  const render = () => {
    const gl = glRef.current;
    if (!gl || !programRef.current) return;
    
    const hasBloom = bloomIntensityRef.current > 0 && 
                     framebufferRef.current && 
                     sceneTextureRef.current && 
                     blurFramebuffer1Ref.current && 
                     blurTexture1Ref.current && 
                     blurFramebuffer2Ref.current && 
                     blurTexture2Ref.current && 
                     blurProgramRef.current && 
                     compositeProgramRef.current;
                     
    if (hasBloom && extractProgramRef.current && extractFramebufferRef.current && extractTextureRef.current) {
      renderScene(framebufferRef.current);
      renderExtract(sceneTextureRef.current, extractFramebufferRef.current);
      renderBlur(extractTextureRef.current, blurFramebuffer1Ref.current, [1, 0]);
      renderBlur(blurTexture1Ref.current, blurFramebuffer2Ref.current, [0, 1]);
      renderComposite(sceneTextureRef.current, blurTexture2Ref.current);
    } else {
      renderScene(null);
    }
  };

  const updateParallaxOffset = () => {
    if (!parallaxEnabledRef.current) {
      parallaxOffsetRef.current = 0;
      return;
    }
    const currentScrollY = window.scrollY || window.pageYOffset;
    const viewportHeight = window.innerHeight || 800;
    
    // progress is 0.0 at scrollY = 0, and 1.0 at scrollY = viewportHeight
    const progress = Math.max(0, Math.min(1, currentScrollY / viewportHeight));
    
    parallaxOffsetRef.current = progress - 0.5;
  };

  useEffect(() => {
    const resolved = resolveTokenColor(color);
    const rgba = parseColorToRgba(resolved);
    colorRef.current = [rgba.r, rgba.g, rgba.b];
    if (glRef.current && programRef.current) {
      render();
    }
  }, [color]);

  useEffect(() => {
    const resolved = transitionColor ? resolveTokenColor(transitionColor) : resolveTokenColor(color);
    const rgba = parseColorToRgba(resolved);
    transitionColorRef.current = [rgba.r, rgba.g, rgba.b];
    if (glRef.current && programRef.current) {
      render();
    }
  }, [transitionColor, color]);

  useEffect(() => {
    noiseScaleRef.current = mapNoiseScale(noiseScale);
    if (glRef.current && programRef.current) {
      render();
    }
  }, [noiseScale]);

  useEffect(() => {
    noiseIntensityRef.current = mapNoiseIntensity(noiseIntensity);
    if (glRef.current && programRef.current) {
      render();
    }
  }, [noiseIntensity]);

  useEffect(() => {
    scrollSensitivityRef.current = mapScrollSensitivity(scrollSensitivity);
  }, [scrollSensitivity]);

  useEffect(() => {
    baseAnimationSpeedRef.current = mapBaseAnimationSpeed(baseAnimationSpeed);
  }, [baseAnimationSpeed]);

  useEffect(() => {
    edgeSoftnessRef.current = mapEdgeSoftness(edgeSoftness);
    if (glRef.current && programRef.current) {
      render();
    }
  }, [edgeSoftness]);

  useEffect(() => {
    const horizontalValue = movement?.horizontal === "left" ? 1 : movement?.horizontal === "right" ? -1 : 0;
    movementHorizontalRef.current = horizontalValue;
    if (glRef.current && programRef.current) {
      render();
    }
  }, [movement?.horizontal]);

  useEffect(() => {
    movementVerticalRef.current = movement?.vertical ?? 0.5;
    if (glRef.current && programRef.current) {
      render();
    }
  }, [movement?.vertical]);

  useEffect(() => {
    parallaxEnabledRef.current = parallaxEnabled;
    parallaxStartRef.current = 0;
    parallaxEndRef.current = 100;
    updateParallaxOffset();
  }, [parallaxEnabled]);

  useEffect(() => {
    bloomIntensityRef.current = bloomIntensity;
  }, [bloomIntensity]);

  useEffect(() => {
    bloomRadiusRef.current = mapBloomRadius(bloomRadius);
    if (glRef.current && blurProgramRef.current) {
      render();
    }
  }, [bloomRadius]);

  const vertexShader = `
    attribute vec2 a_position;
    varying vec2 v_uv;

    void main() {
        v_uv = 0.5 * (a_position + 1.0);
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
  `;

  const fragmentShader = `
    precision mediump float;
    varying vec2 v_uv;
    uniform vec3 u_color;
    uniform vec3 u_transition_color;
    uniform float u_noise_scale;
    uniform float u_noise_intensity;
    uniform float u_scroll_offset;
    uniform float u_edge_softness;
    uniform float u_grain_scale;
    uniform float u_movement_horizontal;
    uniform float u_movement_vertical;
    uniform float u_parallax_offset;
    uniform float u_aspect_ratio;

    float random(vec2 st) {
        return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
    }

    float noise(vec2 st) {
        vec2 i = floor(st);
        vec2 f = fract(st);

        float a = random(i);
        float b = random(i + vec2(1.0, 0.0));
        float c = random(i + vec2(0.0, 1.0));
        float d = random(i + vec2(1.0, 1.0));

        vec2 u = f * f * (3.0 - 2.0 * f);

        return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
    }

    float fbm(vec2 st) {
        float value = 0.0;
        float amplitude = 0.5;

        for (int i = 0; i < 4; i++) {
            value += amplitude * noise(st);
            st *= 2.0;
            amplitude *= 0.5;
        }
        return value;
    }

    float detailedNoise(vec2 st) {
        float value = 0.0;
        float amplitude = 0.5;

        for (int i = 0; i < 6; i++) {
            value += amplitude * noise(st);
            st *= 2.2;
            amplitude *= 0.45;
        }
        return value;
    }

    void main() {
        float baseLine = 0.5 + u_parallax_offset;
        float horizontalOffset = u_scroll_offset * u_movement_horizontal;
        float verticalOffset = u_scroll_offset * u_movement_vertical;

        vec2 noiseCoord = vec2(
            v_uv.x * u_aspect_ratio * u_noise_scale + horizontalOffset,
            v_uv.y * 3.0 + verticalOffset * 0.6
        );
        float edgeNoise = fbm(noiseCoord);
        float mainEdge = baseLine + (edgeNoise - 0.5) * u_noise_intensity;

        vec2 thicknessNoiseCoord = vec2(
            v_uv.x * u_aspect_ratio * u_noise_scale * 2.3 + horizontalOffset * 0.7,
            v_uv.y * 2.0 + verticalOffset * 0.4 + 100.0
        );
        float thicknessNoise = fbm(thicknessNoiseCoord);
        float minThickness = u_edge_softness * 0.1;
        float maxThickness = u_edge_softness;
        float localThickness = mix(minThickness, maxThickness, thicknessNoise);

        float lowerBound = mainEdge - localThickness * 0.4;
        float upperBound = mainEdge + localThickness * 0.6;

        vec2 grainCoord = vec2(
            v_uv.x * u_aspect_ratio * u_grain_scale * 3.0 + horizontalOffset * 0.5,
            v_uv.y * u_grain_scale * 3.0 + verticalOffset * 0.3
        );
        float grain = detailedNoise(grainCoord);

        vec2 fiberCoord = vec2(
            v_uv.x * u_aspect_ratio * u_grain_scale * 8.0 + horizontalOffset * 0.3,
            v_uv.y * u_grain_scale * 2.0 + verticalOffset * 0.2
        );
        float fiberNoise = noise(fiberCoord);
        float combinedGrain = grain * 0.6 + fiberNoise * 0.4;

        if (v_uv.y < lowerBound) {
            gl_FragColor = vec4(u_color, 1.0);
        }
        else if (v_uv.y < mainEdge) {
            float t = (v_uv.y - lowerBound) / max(mainEdge - lowerBound, 0.001);
            float grainThreshold = 1.0 - pow(t, 1.5);
            grainThreshold -= thicknessNoise * 0.2;

            if (combinedGrain > grainThreshold) {
                gl_FragColor = vec4(u_transition_color, 1.0);
            } else {
                gl_FragColor = vec4(u_color, 1.0);
            }
        }
        else if (v_uv.y < upperBound) {
            float t = (v_uv.y - mainEdge) / max(upperBound - mainEdge, 0.001);
            float grainThreshold = pow(t, 1.2);
            grainThreshold += thicknessNoise * 0.15;

            if (combinedGrain > grainThreshold) {
                gl_FragColor = vec4(u_transition_color, 1.0);
            } else {
                discard;
            }
        }
        else {
            discard;
        }
    }
  `;

  const extractFragmentShader = `
    precision mediump float;
    varying vec2 v_uv;
    uniform sampler2D u_texture;
    uniform vec3 u_transition_color;
    uniform vec3 u_base_color;

    void main() {
        vec4 pixel = texture2D(u_texture, v_uv);
        float distToTransition = length(pixel.rgb - u_transition_color);
        float distToBase = length(pixel.rgb - u_base_color);

        float isTransition = 1.0 - smoothstep(0.0, 0.5, distToTransition);
        float notBase = smoothstep(0.0, 0.3, distToBase);
        float mask = isTransition * notBase * pixel.a;
        mask = pow(mask, 0.8);
        gl_FragColor = vec4(1.0, 1.0, 1.0, mask);
    }
  `;

  const blurFragmentShader = `
    precision mediump float;
    varying vec2 v_uv;
    uniform sampler2D u_texture;
    uniform vec2 u_direction;
    uniform vec2 u_resolution;
    uniform float u_radius;

    void main() {
        float blur_size = u_radius * 12.0;
        float alpha = 0.0;
        float totalWeight = 0.0;

        for (int i = -6; i <= 6; i++) {
            float offset = float(i);
            float weight = exp(-0.5 * (offset * offset) / 4.0);
            vec2 sampleOffset = u_direction * (offset * blur_size) / u_resolution;
            float sampleAlpha = texture2D(u_texture, v_uv + sampleOffset).a;

            alpha += sampleAlpha * weight;
            totalWeight += weight;
        }

        alpha = totalWeight > 0.0 ? alpha / totalWeight : 0.0;
        gl_FragColor = vec4(1.0, 1.0, 1.0, alpha);
    }
  `;

  const compositeFragmentShader = `
    precision mediump float;
    varying vec2 v_uv;
    uniform sampler2D u_scene;
    uniform sampler2D u_bloom;
    uniform float u_bloom_intensity;
    uniform vec3 u_transition_color;

    void main() {
        vec4 scene = texture2D(u_scene, v_uv);
        vec4 bloom = texture2D(u_bloom, v_uv);
        float bloomStrength = bloom.a * u_bloom_intensity;
        vec3 bloomColor = u_transition_color * bloomStrength * 2.0;

        if (scene.a < 0.001) {
            float glowAlpha = bloomStrength * 1.5;
            gl_FragColor = vec4(u_transition_color, glowAlpha);
        } else {
            vec3 result = scene.rgb + bloomColor;
            result = min(result, vec3(1.0));
            gl_FragColor = vec4(result, scene.a);
        }
    }
  `;

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const gl = canvas.getContext("webgl", { alpha: true, premultipliedAlpha: false });
    if (!gl) {
      console.error("WebGL not supported");
      return;
    }
    glRef.current = gl;

    const vertexShaderObj = createShader(gl, gl.VERTEX_SHADER, vertexShader);
    const fragmentShaderObj = createShader(gl, gl.FRAGMENT_SHADER, fragmentShader);
    if (!vertexShaderObj || !fragmentShaderObj) return;

    const program = createProgram(gl, vertexShaderObj, fragmentShaderObj);
    if (!program) return;
    programRef.current = program;

    const extractFragmentShaderObj = createShader(gl, gl.FRAGMENT_SHADER, extractFragmentShader);
    if (extractFragmentShaderObj) {
      const extractProgram = createProgram(gl, vertexShaderObj, extractFragmentShaderObj);
      if (extractProgram) {
        extractProgramRef.current = extractProgram;
      }
    }

    const blurFragmentShaderObj = createShader(gl, gl.FRAGMENT_SHADER, blurFragmentShader);
    if (blurFragmentShaderObj) {
      const blurProgram = createProgram(gl, vertexShaderObj, blurFragmentShaderObj);
      if (blurProgram) {
        blurProgramRef.current = blurProgram;
      }
    }

    const compositeFragmentShaderObj = createShader(gl, gl.FRAGMENT_SHADER, compositeFragmentShader);
    if (compositeFragmentShaderObj) {
      const compositeProgram = createProgram(gl, vertexShaderObj, compositeFragmentShaderObj);
      if (compositeProgram) {
        compositeProgramRef.current = compositeProgram;
      }
    }

    const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const buffer = gl.createBuffer();
    if (!buffer) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    bufferRef.current = buffer;

    const initialWidth = 256;
    const initialHeight = 256;
    const { framebuffer: fb1, texture: tex1 } = createFramebufferTexture(gl, initialWidth, initialHeight);
    framebufferRef.current = fb1;
    sceneTextureRef.current = tex1;

    const { framebuffer: fbExtract, texture: texExtract } = createFramebufferTexture(gl, initialWidth, initialHeight);
    extractFramebufferRef.current = fbExtract;
    extractTextureRef.current = texExtract;

    const { framebuffer: fb2, texture: tex2 } = createFramebufferTexture(gl, initialWidth, initialHeight);
    blurFramebuffer1Ref.current = fb2;
    blurTexture1Ref.current = tex2;

    const { framebuffer: fb3, texture: tex3 } = createFramebufferTexture(gl, initialWidth, initialHeight);
    blurFramebuffer2Ref.current = fb3;
    blurTexture2Ref.current = tex3;

    startTimeRef.current = performance.now();

    const resizeObserver = new ResizeObserver(() => {
      resizeCanvas();
    });
    resizeObserver.observe(container);

    resizeCanvas();
    updateParallaxOffset();
    render();

    const animate = () => {
      if (glRef.current && programRef.current) {
        if (parallaxEnabledRef.current) {
          updateParallaxOffset();
        }
        render();
      }
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    animationFrameRef.current = requestAnimationFrame(animate);

    const scrollHandler = () => {
      const currentScrollY = window.scrollY || window.pageYOffset;
      const currentTime = performance.now();
      if (lastScrollTimeRef.current > 0) {
        const deltaY = currentScrollY - lastScrollYRef.current;
        const deltaTime = currentTime - lastScrollTimeRef.current;
        if (deltaTime > 0 && Math.abs(deltaY) > 0) {
          scrollVelocityRef.current = (deltaY / deltaTime) * 1000;
          scrollOffsetRef.current += deltaY * scrollSensitivityRef.current;
        }
      }
      lastScrollYRef.current = currentScrollY;
      lastScrollTimeRef.current = currentTime;
      
      updateParallaxOffset();
    };

    lastScrollYRef.current = window.scrollY || window.pageYOffset;
    lastScrollTimeRef.current = performance.now();
    
    window.addEventListener("scroll", scrollHandler, { passive: true });

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("scroll", scrollHandler);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (glRef.current) {
        const gl = glRef.current;
        if (bufferRef.current) gl.deleteBuffer(bufferRef.current);
        if (programRef.current) gl.deleteProgram(programRef.current);
        if (extractProgramRef.current) gl.deleteProgram(extractProgramRef.current);
        if (blurProgramRef.current) gl.deleteProgram(blurProgramRef.current);
        if (compositeProgramRef.current) gl.deleteProgram(compositeProgramRef.current);
        
        if (framebufferRef.current) gl.deleteFramebuffer(framebufferRef.current);
        if (sceneTextureRef.current) gl.deleteTexture(sceneTextureRef.current);
        if (extractFramebufferRef.current) gl.deleteFramebuffer(extractFramebufferRef.current);
        if (extractTextureRef.current) gl.deleteTexture(extractTextureRef.current);
        if (blurFramebuffer1Ref.current) gl.deleteFramebuffer(blurFramebuffer1Ref.current);
        if (blurTexture1Ref.current) gl.deleteTexture(blurTexture1Ref.current);
        if (blurFramebuffer2Ref.current) gl.deleteFramebuffer(blurFramebuffer2Ref.current);
        if (blurTexture2Ref.current) gl.deleteTexture(blurTexture2Ref.current);
      }
    };
  }, []);

  const containerStyle = {
    ...style,
    position: "relative" as const,
    width: "100%",
    height: "100%",
    overflow: "hidden"
  };

  const canvasStyle = {
    position: "absolute" as const,
    inset: 0,
    width: "100%",
    height: "100%",
    display: "block"
  };

  return (
    <div ref={containerRef} style={containerStyle}>
      <canvas ref={canvasRef} style={canvasStyle} />
    </div>
  );
}
