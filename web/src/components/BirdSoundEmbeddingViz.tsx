import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import {
  buildAudioDrivenPoints,
  enrichChirpRgb,
  extractChirpChains,
  freqToColor,
  syntheticAudioToWavBlob,
} from "@/lib/audioDrivenPointCloud";

interface FrequencyChirpVizProps {
  seed: string;
  audioFile?: File | null;
}

const FPS = 60;
const MIN_FREQ = 2000;
const MAX_FREQ = 8000;

// ── Custom ShaderMaterial: renders each point as a colored square with border + center dot ──

const SQUARE_VERT = /* glsl */ `
  attribute vec3 aColor;
  attribute float aAmplitude;
  attribute float aLitness;
  attribute float aDormancy;

  varying vec3 vColor;
  varying float vAmplitude;
  varying float vLitness;
  varying float vDormancy;

  void main() {
    vColor = aColor;
    vAmplitude = aAmplitude;
    vLitness = aLitness;
    vDormancy = aDormancy;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    float dist = max(-mvPosition.z, 0.5);

    float baseSize;
    if (aLitness >= 1.95) {
      baseSize = 17.0;
    } else if (aLitness > 0.08) {
      baseSize = 10.0 + aAmplitude * 7.0;
    } else {
      // Ghost: shrink toward just the center dot as dormancy grows (0=fresh square, 1=dot only)
      float shrink = 1.0 - aDormancy * 0.72;
      baseSize = (6.0 + aAmplitude * 4.5) * max(shrink, 0.28);
    }

    gl_PointSize = clamp(baseSize * (160.0 / dist), 1.0, 30.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const SQUARE_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAmplitude;
  varying float vLitness;
  varying float vDormancy;

  void main() {
    vec2 uv = gl_PointCoord - vec2(0.5);
    float sq = max(abs(uv.x), abs(uv.y));
    float r  = length(uv);

    bool isActive = vLitness >= 1.95;
    bool isLit    = vLitness > 0.08 && !isActive;
    float litW    = clamp(vLitness, 0.0, 1.0);

    bool isBorder = sq > 0.355;
    bool isDot    = r  < 0.090;

    // Ghost points: only border ring + tiny dot visible
    if (!isActive && !isLit && !isBorder && !isDot) discard;

    vec3  col;
    float alpha;

    if (isActive) {
      if (isBorder)     { col = vec3(1.0, 0.07, 0.03); alpha = 1.00; }
      else if (isDot)   { col = vec3(1.0, 1.0,  1.0);  alpha = 1.00; }
      else              { col = vColor * 0.12;           alpha = 0.40; }
    } else if (isLit) {
      if (isBorder)     { col = vColor * (1.9 + litW * 2.1); alpha = 0.93; }
      else if (isDot)   { col = vec3(1.0);                    alpha = 0.88; }
      else              { col = vColor * (0.2 + litW * 0.3);  alpha = 0.32; }
    } else {
      // Ghost: border fades out as dormancy → 1; center dot persists and brightens
      float borderFade = max(0.0, 1.0 - vDormancy);
      float g = 0.045 + vAmplitude * 0.04;
      if (isBorder) {
        if (borderFade < 0.015) discard;
        col = vColor * g;
        alpha = (0.28 + vAmplitude * 0.12) * borderFade;
      } else {
        // Dot: always visible; gets slightly brighter as the border fully disappears
        col = vColor * (g * 0.4 + vDormancy * g * 0.35);
        alpha = 0.09 + vDormancy * 0.09;
      }
    }

    gl_FragColor = vec4(col, alpha);
  }
`;

// ── 3-D spatial layout (unchanged from original) ──────────────────────────────────────────

function spectrogramSampleToPosition(
  emissionTimeSec: number,
  totalDurSec: number,
  freqHz: number,
  amplitudeRaw: number,
  maxAmp: number,
  normWithinSegment: number | null
): [number, number, number] {
  const normT = emissionTimeSec / totalDurSec;
  const normF = Math.max(0, Math.min(1, (freqHz - MIN_FREQ) / (MAX_FREQ - MIN_FREQ)));
  const normA = maxAmp > 1e-8 ? amplitudeRaw / maxAmp : amplitudeRaw;

  let x: number;
  if (normWithinSegment != null) {
    x = (normWithinSegment - 0.5) * 5.45;
  } else {
    x =
      Math.sin(normT * Math.PI * 2 * 11) * 0.65 +
      Math.cos(normT * Math.PI * 2 * 19 + 1.3) * 0.42;
  }

  const y = (normF - 0.5) * 3.15;
  const phaseRibbon = normWithinSegment != null ? normWithinSegment : normT;
  const wobble =
    0.5 * Math.sin(phaseRibbon * Math.PI * 18 + normF * Math.PI * 9) +
    0.4 * Math.sin(phaseRibbon * Math.PI * 31 - normF * Math.PI * 13) +
    0.28 * Math.cos(phaseRibbon * Math.PI * 8 + normA * Math.PI * 6);
  const z = (normA - 0.38) * 3.05 + wobble * (0.3 + normA * 0.95);

  return [x, y, z];
}

// ── CSS color string helper ───────────────────────────────────────────────────────────────

function colorToCss(c: { r: number; g: number; b: number }): string {
  return `rgb(${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)})`;
}

// ── Main component ────────────────────────────────────────────────────────────────────────

export function BirdSoundEmbeddingViz({ seed, audioFile }: FrequencyChirpVizProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let alive = true;
    let raf = 0;
    let resizeObserver: ResizeObserver | null = null;
    let renderer: THREE.WebGLRenderer | null = null;
    let labelRenderer: CSS2DRenderer | null = null;
    let controls: OrbitControls | null = null;
    let pointsGeometry: THREE.BufferGeometry | null = null;
    let pointsMaterial: THREE.ShaderMaterial | null = null;
    let litLineGeometry: THREE.BufferGeometry | null = null;
    let litLineMaterial: THREE.LineBasicMaterial | null = null;
    let skelGeometry: THREE.BufferGeometry | null = null;
    let skelMaterial: THREE.LineBasicMaterial | null = null;
    let composer: EffectComposer | null = null;
    let bloomPass: UnrealBloomPass | null = null;
    let playbackObjectUrl: string | null = null;

    const setup = async () => {
      const points = await buildAudioDrivenPoints(seed, audioFile);
      if (!alive || points.length === 0) return;

      // ── Scene ─────────────────────────────────────────────────────────────
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x050608);
      scene.fog = new THREE.Fog(0x050608, 6, 26);

      const camera = new THREE.PerspectiveCamera(46, 1, 0.08, 90);
      camera.position.set(3.15, 1.95, 4.35);

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 0.88;

      labelRenderer = new CSS2DRenderer();
      labelRenderer.domElement.className = "embedding-viz__labels";
      labelRenderer.domElement.style.position = "absolute";
      labelRenderer.domElement.style.inset = "0";
      labelRenderer.domElement.style.pointerEvents = "none";

      mount.appendChild(renderer.domElement);
      mount.appendChild(labelRenderer.domElement);

      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.07;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 0.3;
      controls.target.set(0, 0.05, 0);
      controls.minDistance = 1.2;
      controls.maxDistance = 8;

      // ── Audio-driven data arrays ───────────────────────────────────────────
      const n = points.length;
      const basePos = new Float32Array(n * 3);
      const positions = new Float32Array(n * 3);
      const aColors = new Float32Array(n * 3);
      const aAmplitudes = new Float32Array(n);
      const aLitness = new Float32Array(n);
      const amplitudes = new Float32Array(n);
      const emissions = new Float32Array(n);
      const freqs = new Float32Array(n);
      const totalDur = Math.max(1 / FPS, points[n - 1]?.emissionTime ?? 1 / FPS);

      let maxAmpAll = 0;
      for (let i = 0; i < n; i++) {
        const p = points[i]!;
        amplitudes[i] = p.amplitude;
        emissions[i] = p.emissionTime;
        freqs[i] = p.freqHz;
        maxAmpAll = Math.max(maxAmpAll, p.amplitude);
      }

      // Chirp chains for spatial layout
      const chirpOpts = { gapSeconds: 0.068, minChirpSeconds: 0.048, maxNodes: 200 } as const;
      const chains = extractChirpChains(amplitudes, FPS, chirpOpts);
      const chainOfFrame = new Int32Array(n).fill(-1);
      const segT0: number[] = [];
      const segT1: number[] = [];
      chains.forEach((ch, cid) => {
        segT0[cid] = emissions[ch[0]!]!;
        segT1[cid] = emissions[ch[ch.length - 1]!]!;
        for (const idx of ch) chainOfFrame[idx] = cid;
      });

      // Position of each frame within its chain — enables full-chain accumulation lighting
      const posInChain = new Int32Array(n).fill(-1);
      chains.forEach((ch) => {
        ch.forEach((frameIdx, pos) => {
          posInChain[frameIdx] = pos;
        });
      });

      // Per-frame lit intensity that persists across animation frames (allows slow verse fade-out)
      const litIntensities = new Float32Array(n);
      const CHIRP_FADE_SECS = 0.9; // how long a completed verse lingers before fading

      // Dormancy: 0 = freshly-lit square, 1 = fully dormant (just the center dot visible).
      // All points start fully dormant so the initial view is a cloud of tiny dots.
      // When a point gets lit its dormancy resets to 0; when dark again it slowly climbs back to 1.
      const dormancy = new Float32Array(n).fill(1.0);
      const DORMANCY_SECS = 12; // seconds for an un-lit square to shrink down to its center dot

      // Build 3-D positions and per-point color
      for (let i = 0; i < n; i++) {
        const p = points[i]!;
        const cid = chainOfFrame[i]!;
        let normWithin: number | null = null;
        if (cid >= 0) {
          const t0 = segT0[cid]!;
          const t1 = segT1[cid]!;
          normWithin = Math.max(0, Math.min(1, (emissions[i]! - t0) / Math.max(t1 - t0, 1 / FPS)));
        }
        const [x, y, z] = spectrogramSampleToPosition(
          p.emissionTime,
          totalDur,
          p.freqHz,
          p.amplitude,
          maxAmpAll,
          normWithin
        );
        basePos[i * 3] = x;
        basePos[i * 3 + 1] = y;
        basePos[i * 3 + 2] = z;
        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;

        const c = enrichChirpRgb(freqToColor(p.freqHz));
        aColors[i * 3] = c.r;
        aColors[i * 3 + 1] = c.g;
        aColors[i * 3 + 2] = c.b;
        aAmplitudes[i] = maxAmpAll > 1e-8 ? p.amplitude / maxAmpAll : p.amplitude;
        aLitness[i] = 0;
      }

      // Adaptive noise floor for playhead gating
      const sortedAmp = Array.from(amplitudes).sort((a, b) => a - b);
      const q15 = sortedAmp[Math.min(sortedAmp.length - 1, Math.floor(sortedAmp.length * 0.15))] ?? 0.018;
      const ampLitFloor = Math.max(0.009, Math.min(0.08, q15 * 0.92));

      // ── Static chirp-chain skeleton lines ─────────────────────────────────
      const MAX_SKEL = 2500;
      const skelEdgeA: number[] = [];
      const skelEdgeB: number[] = [];
      for (const ch of chains) {
        for (let k = 0; k + 1 < ch.length && skelEdgeA.length < MAX_SKEL; k++) {
          skelEdgeA.push(ch[k]!);
          skelEdgeB.push(ch[k + 1]!);
        }
      }
      if (skelEdgeA.length > 0) {
        const E = skelEdgeA.length;
        const skelPos = new Float32Array(E * 6);
        const skelCol = new Float32Array(E * 6);
        for (let k = 0; k < E; k++) {
          const ia = skelEdgeA[k]!;
          const ib = skelEdgeB[k]!;
          const ca = freqToColor(freqs[ia]!);
          const cb = freqToColor(freqs[ib]!);
          const dim = 0.03;
          skelPos[k * 6]     = basePos[ia * 3]!;
          skelPos[k * 6 + 1] = basePos[ia * 3 + 1]!;
          skelPos[k * 6 + 2] = basePos[ia * 3 + 2]!;
          skelPos[k * 6 + 3] = basePos[ib * 3]!;
          skelPos[k * 6 + 4] = basePos[ib * 3 + 1]!;
          skelPos[k * 6 + 5] = basePos[ib * 3 + 2]!;
          skelCol[k * 6]     = ca.r * dim;
          skelCol[k * 6 + 1] = ca.g * dim;
          skelCol[k * 6 + 2] = ca.b * dim;
          skelCol[k * 6 + 3] = cb.r * dim;
          skelCol[k * 6 + 4] = cb.g * dim;
          skelCol[k * 6 + 5] = cb.b * dim;
        }
        skelGeometry = new THREE.BufferGeometry();
        skelGeometry.setAttribute("position", new THREE.BufferAttribute(skelPos, 3));
        skelGeometry.setAttribute("color", new THREE.BufferAttribute(skelCol, 3));
        skelMaterial = new THREE.LineBasicMaterial({
          vertexColors: true,
          transparent: true,
          opacity: 0.9,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        });
        scene.add(new THREE.LineSegments(skelGeometry, skelMaterial));
      }

      // ── Point cloud (ShaderMaterial) ──────────────────────────────────────
      pointsGeometry = new THREE.BufferGeometry();
      pointsGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      pointsGeometry.setAttribute("aColor", new THREE.BufferAttribute(aColors, 3));
      pointsGeometry.setAttribute("aAmplitude", new THREE.BufferAttribute(aAmplitudes, 1));
      pointsGeometry.setAttribute("aLitness", new THREE.BufferAttribute(aLitness, 1));
      // aDormancy starts at 1 (fully dormant = dot only); reset to 0 when lit
      pointsGeometry.setAttribute("aDormancy", new THREE.BufferAttribute(new Float32Array(n).fill(1.0), 1));

      pointsMaterial = new THREE.ShaderMaterial({
        vertexShader: SQUARE_VERT,
        fragmentShader: SQUARE_FRAG,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      scene.add(new THREE.Points(pointsGeometry, pointsMaterial));

      // ── Dynamic lit-line segments (updated each frame) ────────────────────
      let chainEdgeCount = 0;
      for (const ch of chains) chainEdgeCount += Math.max(0, ch.length - 1);
      const allocSegs = Math.min(4096, Math.max(chainEdgeCount, 32));

      litLineGeometry = new THREE.BufferGeometry();
      litLineGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(allocSegs * 6), 3));
      litLineGeometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(allocSegs * 6), 3));
      litLineGeometry.setDrawRange(0, 0);
      litLineMaterial = new THREE.LineBasicMaterial({
        transparent: true,
        opacity: 0.7,
        vertexColors: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      scene.add(new THREE.LineSegments(litLineGeometry, litLineMaterial));

      // ── Labels ────────────────────────────────────────────────────────────
      const labelRoot = new THREE.Group();
      scene.add(labelRoot);

      // Dynamic labels pool: shown on lit / active chirp-chain points with animated lifetime
      const DYN_SLOTS = 8;
      type DynSlot = {
        wrap: HTMLDivElement;
        ampEl: HTMLElement;
        lifeEl: HTMLElement;
        timeEl: HTMLElement;
        obj: CSS2DObject;
      };
      const dynSlots: DynSlot[] = [];
      for (let j = 0; j < DYN_SLOTS; j++) {
        const wrap = document.createElement("div");
        wrap.className = "embedding-viz__label evl-dyn";
        wrap.style.opacity = "0";
        const ampEl = document.createElement("div");
        ampEl.className = "evl-amp";
        ampEl.textContent = "—";
        const lifeEl = document.createElement("div");
        lifeEl.className = "evl-life";
        lifeEl.textContent = "—";
        const timeEl = document.createElement("div");
        timeEl.className = "evl-time";
        timeEl.textContent = "—";
        wrap.appendChild(ampEl);
        wrap.appendChild(lifeEl);
        wrap.appendChild(timeEl);
        const obj = new CSS2DObject(wrap);
        obj.position.set(0, 0, 0);
        dynSlots.push({ wrap, ampEl, lifeEl, timeEl, obj });
        labelRoot.add(obj);
      }

      scene.add(new THREE.AmbientLight(0xffffff, 0.28));

      // ── Post-processing ───────────────────────────────────────────────────
      const dpr = () => Math.min(window.devicePixelRatio, 2);
      const bloomRes = () => {
        const w = Math.max(320, mount.clientWidth);
        const h = Math.max(420, mount.clientHeight);
        const p = dpr();
        return new THREE.Vector2(Math.max(256, Math.floor(w * p)), Math.max(256, Math.floor(h * p)));
      };
      composer = new EffectComposer(renderer);
      const renderScene = new RenderPass(scene, camera);
      // threshold 0.22 → ghost points (~0.1 luminance) won't bloom; lit points (>0.4) will
      bloomPass = new UnrealBloomPass(bloomRes(), 1.45, 0.68, 0.22);
      composer.addPass(renderScene);
      composer.addPass(bloomPass);

      const setSize = () => {
        if (!renderer || !labelRenderer || !composer) return;
        const w = Math.max(320, mount.clientWidth);
        const h = Math.max(420, mount.clientHeight);
        const p = dpr();
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setPixelRatio(p);
        renderer.setSize(w, h);
        composer.setPixelRatio(p);
        composer.setSize(w, h);
        labelRenderer.setSize(w, h);
        if (bloomPass) {
          const br = bloomRes();
          bloomPass.resolution.set(br.x, br.y);
        }
      };
      setSize();
      resizeObserver = new ResizeObserver(() => setSize());
      resizeObserver.observe(mount);

      // ── Animation loop ────────────────────────────────────────────────────

      const nearestFrameIndex = (t: number): number => {
        const probe = Math.min(n - 1, Math.max(0, Math.floor(t * FPS + 1e-6)));
        let best = probe;
        let bestAbs = Math.abs(emissions[probe]! - t);
        const lo = Math.max(0, probe - 2);
        const hi = Math.min(n - 1, probe + 2);
        for (let j = lo; j <= hi; j++) {
          const d = Math.abs(emissions[j]! - t);
          if (d < bestAbs) {
            bestAbs = d;
            best = j;
          }
        }
        return best;
      };

      const animate = () => {
        if (
          !alive ||
          !renderer ||
          !labelRenderer ||
          !composer ||
          !controls ||
          !pointsGeometry ||
          !litLineGeometry ||
          !litLineMaterial
        )
          return;
        raf = requestAnimationFrame(animate);

        const au = audioRef.current;
        const rawClock = au != null ? au.currentTime : (performance.now() / 1000) % totalDur;
        const lastEmission = emissions[n - 1]!;
        const audioTime = Math.min(
          Math.max(0, rawClock),
          Math.max(lastEmission, totalDur) - 0.25 / FPS
        );

        const iPlay = nearestFrameIndex(audioTime);
        const ampNow = amplitudes[iPlay]!;

        // ── Decay all lit intensities each frame (verse fades after chirp ends) ──
        const decayRate = 1 / (FPS * CHIRP_FADE_SECS);
        for (let i = 0; i < n; i++) {
          if (litIntensities[i] > 0) litIntensities[i] = Math.max(0, litIntensities[i] - decayRate);
        }

        // ── Accumulate the full chirp chain up to the current playhead ────────
        const cid = chainOfFrame[iPlay]!;
        if (ampNow > ampLitFloor * 0.85) {
          if (cid >= 0) {
            // Inside a chirp: build the entire verse sculpture from chain start → now
            const chain = chains[cid]!;
            const playPos = posInChain[iPlay]!;
            litIntensities[iPlay] = 1.0;
            for (let k = playPos - 1; k >= 0; k--) {
              const idx = chain[k]!;
              // Gradient: start of chain is dim, just behind playhead is bright
              const progress = playPos > 0 ? k / playPos : 1;
              const w = 0.20 + progress * 0.80;
              litIntensities[idx] = Math.max(litIntensities[idx], w);
            }
          } else {
            // Gap / silence frame — light just the immediate playhead
            litIntensities[iPlay] = Math.max(litIntensities[iPlay], 0.7);
            for (let back = 1; back <= 6; back++) {
              const j = iPlay - back;
              if (j < 0) break;
              const age = audioTime - emissions[j]!;
              if (age > 0.15) break;
              litIntensities[j] = Math.max(litIntensities[j], 0.35 * (1 - age / 0.15));
            }
          }
        }

        // ── Derive master brightness and lit set from accumulated intensities ──
        const litSet = new Set<number>();
        let maxLitIntensity = 0;
        for (let i = 0; i < n; i++) {
          if (litIntensities[i] > 0.02) {
            litSet.add(i);
            if (litIntensities[i] > maxLitIntensity) maxLitIntensity = litIntensities[i];
          }
        }
        const masterFade = Math.min(1, maxLitIntensity * 1.8);

        // Update aLitness attribute (drives shader square state per point)
        const litnessAttr = pointsGeometry.getAttribute("aLitness") as THREE.BufferAttribute;
        for (let i = 0; i < n; i++) {
          if (i === iPlay && ampNow > ampLitFloor * 0.85) {
            litnessAttr.setX(i, 2.0);                              // red border playhead
          } else if (litIntensities[i] > 0.02) {
            litnessAttr.setX(i, litIntensities[i]);               // decay chain
          } else {
            litnessAttr.setX(i, 0.0);                              // ghost
          }
        }
        litnessAttr.needsUpdate = true;

        // ── Dormancy: lit points snap to 0 (full square); dark points slowly age toward 1 (dot only) ──
        const dormancyRate = 1 / (FPS * DORMANCY_SECS);
        const dormancyAttr = pointsGeometry.getAttribute("aDormancy") as THREE.BufferAttribute;
        for (let i = 0; i < n; i++) {
          if (litIntensities[i] > 0.02) {
            dormancy[i] = 0; // lit → instantly materialise as a full square
          } else {
            dormancy[i] = Math.min(1.0, dormancy[i] + dormancyRate);
          }
          dormancyAttr.setX(i, dormancy[i]);
        }
        dormancyAttr.needsUpdate = true;

        // Dynamic lit line segments connecting the active verse network
        const lineAttr = litLineGeometry.getAttribute("position") as THREE.BufferAttribute;
        const lineColorAttr = litLineGeometry.getAttribute("color") as THREE.BufferAttribute;
        let segCount = 0;
        for (let i = 0; i < n - 1 && segCount < allocSegs; i++) {
          if (!litSet.has(i) || !litSet.has(i + 1)) continue;
          const v0 = segCount * 2;
          const v1 = segCount * 2 + 1;
          lineAttr.setXYZ(v0, basePos[i * 3]!, basePos[i * 3 + 1]!, basePos[i * 3 + 2]!);
          lineAttr.setXYZ(v1, basePos[(i + 1) * 3]!, basePos[(i + 1) * 3 + 1]!, basePos[(i + 1) * 3 + 2]!);
          const litA = litIntensities[i]!;
          const litB = litIntensities[i + 1]!;
          const ca = freqToColor(freqs[i]!);
          const cb = freqToColor(freqs[i + 1]!);
          lineColorAttr.setXYZ(v0, ca.r * litA * 2.0, ca.g * litA * 2.0, ca.b * litA * 2.0);
          lineColorAttr.setXYZ(v1, cb.r * litB * 2.0, cb.g * litB * 2.0, cb.b * litB * 2.0);
          segCount++;
        }
        litLineGeometry.setDrawRange(0, segCount * 2);
        lineAttr.needsUpdate = true;
        lineColorAttr.needsUpdate = true;
        litLineMaterial.opacity = 0.8 * Math.max(0.1, masterFade);

        // Dynamic labels — pick highest-intensity lit frames (playhead first, then spread across verse)
        const litArray = Array.from(litSet).sort((a, b) => {
          if (a === iPlay) return -1;
          if (b === iPlay) return 1;
          return litIntensities[b]! - litIntensities[a]!;  // brightest first
        });
        for (let j = 0; j < dynSlots.length; j++) {
          const slot = dynSlots[j]!;
          if (j < litArray.length && masterFade > 0.02) {
            const idx = litArray[j]!;
            slot.obj.position.set(
              basePos[idx * 3]! + 0.10,
              basePos[idx * 3 + 1]! + 0.20,
              basePos[idx * 3 + 2]!
            );
            const amp = amplitudes[idx]!;
            const ampNorm = maxAmpAll > 1e-8 ? amp / maxAmpAll : amp;
            const life = Math.max(0, audioTime - emissions[idx]!);
            slot.ampEl.textContent = ampNorm.toFixed(4);
            slot.lifeEl.textContent = life.toFixed(2);
            slot.timeEl.textContent = emissions[idx]!.toFixed(2);
            const c = freqToColor(freqs[idx]!);
            slot.wrap.style.setProperty("--lc", colorToCss(c));
            slot.wrap.className =
              "embedding-viz__label evl-dyn" + (idx === iPlay ? " evl-active" : "");
            slot.wrap.style.opacity = String(0.70 + masterFade * 0.30);
          } else {
            slot.wrap.style.opacity = "0.004";
          }
        }

        controls.update();
        composer.render();
        labelRenderer.render(scene, camera);
      };
      animate();

      const statsTag = mount.parentElement?.querySelector(".embedding-viz-stats") as HTMLElement | null;
      if (statsTag) {
        statsTag.textContent = `${n} frames · ≤${totalDur.toFixed(1)}s · ${chains.length} chirp chain${chains.length !== 1 ? "s" : ""}`;
      }

      // ── Audio playback ────────────────────────────────────────────────────
      const audio = new Audio();
      audioRef.current = audio;
      audio.loop = true;
      audio.volume = 0.7;
      playbackObjectUrl = URL.createObjectURL(audioFile ?? syntheticAudioToWavBlob(seed));
      if (!alive) {
        URL.revokeObjectURL(playbackObjectUrl);
        playbackObjectUrl = null;
        return;
      }
      audio.src = playbackObjectUrl;
      const tryPlay = () => {
        void audio.play().catch(() => {
          /* autoplay policy — first click on the stage starts audio */
        });
      };
      tryPlay();
      mount.addEventListener("click", () => { tryPlay(); }, { once: true });
    };

    setup();

    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      resizeObserver?.disconnect();
      controls?.dispose();
      bloomPass?.dispose();
      composer?.dispose();
      pointsGeometry?.dispose();
      pointsMaterial?.dispose();
      litLineGeometry?.dispose();
      litLineMaterial?.dispose();
      skelGeometry?.dispose();
      skelMaterial?.dispose();
      renderer?.dispose();
      if (renderer?.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
      if (labelRenderer?.domElement.parentNode === mount) mount.removeChild(labelRenderer.domElement);
      if (playbackObjectUrl) {
        URL.revokeObjectURL(playbackObjectUrl);
        playbackObjectUrl = null;
      }
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = "";
        audioRef.current.removeAttribute("src");
        audioRef.current = null;
      }
    };
  }, [seed, audioFile]);

  return (
    <div
      ref={mountRef}
      className="embedding-viz"
      role="img"
      aria-label="3-D audio visualization: each audio frame becomes a frequency-colored square data point. The playhead lights the current moment in red; decay trail fades back. Drag to orbit, scroll to zoom."
    />
  );
}
