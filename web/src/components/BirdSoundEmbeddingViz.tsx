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
  /** Drives deterministic fallback when no real audio file is available. */
  seed: string;
  audioFile?: File | null;
}

const FPS = 60;
const GHOST_GREY_R = 0.004;
const GHOST_GREY_G = 0.005;
const GHOST_GREY_B = 0.006;

const MIN_FREQ = 2000;
const MAX_FREQ = 8000;

/**
 * 3‑D layout: amplitude + band on Y/Z plus stable “gesture phase” ribbons.
 *
 * Inside a chirp segment, **horizontal spread is progress through that call only** (`normWithin 0→1`).
 * Repeated identical chirps reuse the **same** X‑wiggle corridor (overlay), not an endless L→R
 * marquee from global clip time. Silence / gaps use a compact wiggle so ghosts don’t scan sideways.
 */
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
    x = Math.sin(normT * Math.PI * 2 * 11) * 0.65 + Math.cos(normT * Math.PI * 2 * 19 + 1.3) * 0.42;
  }

  const y = (normF - 0.5) * 3.15;

  /** Local gesture phase; gaps fall back to `normT` only for leftover depth texture. */
  const phaseRibbon = normWithinSegment != null ? normWithinSegment : normT;

  const wobble =
    0.5 * Math.sin(phaseRibbon * Math.PI * 18 + normF * Math.PI * 9) +
    0.4 * Math.sin(phaseRibbon * Math.PI * 31 - normF * Math.PI * 13) +
    0.28 * Math.cos(phaseRibbon * Math.PI * 8 + normA * Math.PI * 6);

  const z = (normA - 0.38) * 3.05 + wobble * (0.3 + normA * 0.95);

  return [x, y, z];
}

/**
 * Frame-level audio viz: spectrogram-ish 3‑D placement; highlights follow **detected chirp chains**
 * only — the playhead marches forward along each segment in lockstep with `audio.currentTime`,
 * not a blanket time window over the whole clip (that read as an out‑of‑sync slideshow).
 */
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
    let pointsMaterial: THREE.PointsMaterial | null = null;
    let lineGeometry: THREE.BufferGeometry | null = null;
    let lineMaterial: THREE.LineBasicMaterial | null = null;
    let composer: EffectComposer | null = null;
    let bloomPass: UnrealBloomPass | null = null;
    let labelEls: HTMLDivElement[] = [];
    let labels: CSS2DObject[] = [];
    let playbackObjectUrl: string | null = null;

    const setup = async () => {
      const points = await buildAudioDrivenPoints(seed, audioFile);
      if (!alive || points.length === 0) return;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x050608);
      scene.fog = new THREE.Fog(0x050608, 5.6, 24);

      const camera = new THREE.PerspectiveCamera(46, 1, 0.08, 90);
      camera.position.set(3.15, 1.95, 4.35);

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 0.92;

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
      controls.autoRotateSpeed = 0.28;
      controls.target.set(0, 0.05, 0);
      controls.minDistance = 1.2;
      controls.maxDistance = 8;

      const grid = new THREE.GridHelper(15, 60, 0x1c2638, 0x121820);
      grid.position.y = -1.6;
      grid.visible = false;
      scene.add(grid);

      const basePos = new Float32Array(points.length * 3);
      const positions = new Float32Array(points.length * 3);
      const colors = new Float32Array(points.length * 3);
      const amplitudes = new Float32Array(points.length);
      const emissions = new Float32Array(points.length);
      const freqs = new Float32Array(points.length);
      const totalDur = Math.max(1 / FPS, points[points.length - 1]?.emissionTime ?? 1 / FPS);

      let maxAmpAll = 0;
      for (let i = 0; i < points.length; i++) {
        const p = points[i]!;
        amplitudes[i] = p.amplitude;
        emissions[i] = p.emissionTime;
        freqs[i] = p.freqHz;
        maxAmpAll = Math.max(maxAmpAll, p.amplitude);
      }

      const chirpOpts = {
        gapSeconds: 0.068,
        minChirpSeconds: 0.048,
        maxNodes: 200,
      } as const;
      const chains = extractChirpChains(amplitudes, FPS, chirpOpts);

      const chainOfFrame = new Int32Array(points.length).fill(-1);
      const segT0: number[] = [];
      const segT1: number[] = [];
      chains.forEach((ch, cid) => {
        const i0 = ch[0]!;
        const i1 = ch[ch.length - 1]!;
        segT0[cid] = emissions[i0]!;
        segT1[cid] = emissions[i1]!;
        for (const idx of ch) chainOfFrame[idx] = cid;
      });

      for (let i = 0; i < points.length; i++) {
        const p = points[i]!;
        const cid = chainOfFrame[i]!;
        let normWithin: number | null = null;
        if (cid >= 0) {
          const t0 = segT0[cid]!;
          const t1 = segT1[cid]!;
          const denom = Math.max(t1 - t0, 1 / FPS);
          normWithin = Math.max(0, Math.min(1, (emissions[i]! - t0) / denom));
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
      }

      const sortedAmp = Array.from(amplitudes).sort((a, b) => a - b);
      const q15Idx = Math.min(sortedAmp.length - 1, Math.floor(sortedAmp.length * 0.15));
      const q15 = sortedAmp[q15Idx] ?? 0.018;
      const ampLitFloor = Math.max(0.009, Math.min(0.08, q15 * 0.92));
      let maxChain = 0;
      for (const c of chains) {
        maxChain = Math.max(maxChain, c.length);
      }
      const labelSlots = Math.min(96, Math.max(maxChain, 32));
      let chainEdgeCount = 0;
      for (const c of chains) chainEdgeCount += Math.max(0, c.length - 1);
      const allocSegs = Math.min(4096, Math.max(chainEdgeCount, 32));

      for (let i = 0; i < points.length; i++) {
        colors[i * 3] = 0;
        colors[i * 3 + 1] = 0;
        colors[i * 3 + 2] = 0;
      }

      pointsGeometry = new THREE.BufferGeometry();
      pointsGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      pointsGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      pointsMaterial = new THREE.PointsMaterial({
        size: 0.15,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 1,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const pointCloud = new THREE.Points(pointsGeometry, pointsMaterial);
      scene.add(pointCloud);

      const linePositions = new Float32Array(allocSegs * 6);
      const lineColors = new Float32Array(allocSegs * 6);
      lineGeometry = new THREE.BufferGeometry();
      lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
      lineGeometry.setAttribute("color", new THREE.BufferAttribute(lineColors, 3));
      lineGeometry.setDrawRange(0, 0);
      lineMaterial = new THREE.LineBasicMaterial({
        transparent: true,
        opacity: 0.55,
        vertexColors: true,
        blending: THREE.AdditiveBlending,
      });
      const lineSegments = new THREE.LineSegments(lineGeometry, lineMaterial);
      scene.add(lineSegments);

      const labelRoot = new THREE.Group();
      for (let j = 0; j < labelSlots; j++) {
        const el = document.createElement("div");
        el.className = "embedding-viz__label";
        const ampRow = document.createElement("div");
        ampRow.className = "embedding-viz__label__amp";
        ampRow.textContent = "—";
        const timeRow = document.createElement("div");
        timeRow.className = "embedding-viz__label__t";
        timeRow.textContent = "—";
        el.appendChild(ampRow);
        el.appendChild(timeRow);
        el.style.opacity = "0";
        labelEls.push(el);
        const obj = new CSS2DObject(el);
        obj.position.set(0, 0, 0);
        labels.push(obj);
        labelRoot.add(obj);
      }
      scene.add(labelRoot);

      scene.add(new THREE.AmbientLight(0xffffff, 0.32));

      const dpr = () => Math.min(window.devicePixelRatio, 2);
      const bloomResolution = () => {
        const w = Math.max(320, mount.clientWidth);
        const h = Math.max(420, mount.clientHeight);
        const p = dpr();
        return new THREE.Vector2(Math.max(256, Math.floor(w * p)), Math.max(256, Math.floor(h * p)));
      };
      composer = new EffectComposer(renderer);
      const renderScene = new RenderPass(scene, camera);
      bloomPass = new UnrealBloomPass(bloomResolution(), 1.82, 0.88, 0.022);
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
          const br = bloomResolution();
          bloomPass.resolution.set(br.x, br.y);
        }
      };
      setSize();
      resizeObserver = new ResizeObserver(() => setSize());
      resizeObserver.observe(mount);

      const animate = () => {
        if (
          !alive ||
          !renderer ||
          !labelRenderer ||
          !composer ||
          !controls ||
          !pointsGeometry ||
          !lineGeometry ||
          !lineMaterial
        )
          return;
        raf = requestAnimationFrame(animate);

        // Drive the playhead from the HTMLAudio clock whenever we have an element (paused or not).
        // Using wall-clock when paused broke sync after autoplay denial and made scrubbing meaningless.
        const au = audioRef.current;
        const rawClock = au != null ? au.currentTime : (performance.now() / 1000) % totalDur;
        /** No `% totalDur`: that wrapped long uploads onto the wrong part of the track. Clamp to analyzed span. */
        const lastEmission = emissions[points.length - 1]!;
        const audioTime = Math.min(
          Math.max(0, rawClock),
          Math.max(lastEmission, totalDur) - 0.25 / FPS
        );

        /** One playhead sample + tight decay — gate on RMS only so every similar chirp is treated evenly. */
        const DECAY_SEC = 0.075;

        const nearestFrameIndex = (t: number): number => {
          const probe = Math.min(points.length - 1, Math.max(0, Math.floor(t * FPS + 1e-6)));
          let best = probe;
          let bestAbs = Math.abs(emissions[probe]! - t);
          const lo = Math.max(0, probe - 2);
          const hi = Math.min(points.length - 1, probe + 2);
          for (let j = lo; j <= hi; j++) {
            const d = Math.abs(emissions[j]! - t);
            if (d < bestAbs) {
              bestAbs = d;
              best = j;
            }
          }
          return best;
        };

        const iPlay = nearestFrameIndex(audioTime);
        const ampNow = amplitudes[iPlay]!;

        const litSet = new Set<number>();
        const pointIntensities = new Map<number, number>();
        let maxIntensity = 0;

        const considerLit = (idx: number, weight: number) => {
          const a = amplitudes[idx]! * weight;
          if (a <= ampLitFloor * 0.45) return;
          litSet.add(idx);
          pointIntensities.set(idx, Math.max(pointIntensities.get(idx) ?? 0, a));
          maxIntensity = Math.max(maxIntensity, pointIntensities.get(idx)!);
        };

        if (ampNow > ampLitFloor * 0.85) {
          considerLit(iPlay, 1);
          for (let back = 1; back <= 6; back++) {
            const j = iPlay - back;
            if (j < 0) break;
            const age = audioTime - emissions[j]!;
            if (age <= 0) break;
            const w = Math.pow(1 - Math.min(age / DECAY_SEC, 1), 2.4);
            if (w < 0.06) break;
            considerLit(j, w);
          }
        }

        const masterFade = Math.min(1, maxIntensity * 2.6);

        const posAttr = pointsGeometry.getAttribute("position") as THREE.BufferAttribute;
        const colorAttr = pointsGeometry.getAttribute("color") as THREE.BufferAttribute;
        const lineAttr = lineGeometry.getAttribute("position") as THREE.BufferAttribute;
        const lineColorAttr = lineGeometry.getAttribute("color") as THREE.BufferAttribute;

        for (let i = 0; i < points.length; i++) {
          const bx = basePos[i * 3]!;
          const by = basePos[i * 3 + 1]!;
          const bz = basePos[i * 3 + 2]!;
          posAttr.setXYZ(i, bx, by, bz);

          const c = enrichChirpRgb(freqToColor(freqs[i]!));
          const amp = amplitudes[i]!;
          const isLit = litSet.has(i);
          if (isLit) {
            const bright = 1.45 + amp * 1.85;
            const activeR = c.r * bright;
            const activeG = c.g * bright;
            const activeB = c.b * bright;
            const mix = Math.max(0, Math.min(1, masterFade));
            colorAttr.setXYZ(
              i,
              GHOST_GREY_R * (1 - mix) + activeR * mix,
              GHOST_GREY_G * (1 - mix) + activeG * mix,
              GHOST_GREY_B * (1 - mix) + activeB * mix
            );
          } else {
            colorAttr.setXYZ(i, GHOST_GREY_R, GHOST_GREY_G, GHOST_GREY_B);
          }
        }
        posAttr.needsUpdate = true;
        colorAttr.needsUpdate = true;

        let segCount = 0;
        for (let i = 0; i < points.length - 1 && segCount < allocSegs; i++) {
          const ia = i;
          const ib = i + 1;
          if (!litSet.has(ia) || !litSet.has(ib)) continue;
          const oa = ia * 3;
          const ob = ib * 3;
          const v0 = segCount * 2;
          const v1 = segCount * 2 + 1;
          lineAttr.setXYZ(
            v0,
            posAttr.array[oa] as number,
            posAttr.array[oa + 1] as number,
            posAttr.array[oa + 2] as number
          );
          lineAttr.setXYZ(
            v1,
            posAttr.array[ob] as number,
            posAttr.array[ob + 1] as number,
            posAttr.array[ob + 2] as number
          );
          const cA = enrichChirpRgb(freqToColor(freqs[ia]!));
          const cB = enrichChirpRgb(freqToColor(freqs[ib]!));
          const e = masterFade * 1.55;
          lineColorAttr.setXYZ(v0, cA.r * e, cA.g * e, cA.b * e);
          lineColorAttr.setXYZ(v1, cB.r * e, cB.g * e, cB.b * e);
          segCount++;
        }
        lineGeometry.setDrawRange(0, segCount > 0 ? segCount * 2 : 0);
        lineAttr.needsUpdate = true;
        lineColorAttr.needsUpdate = true;
        lineMaterial.opacity = 0.7 * Math.max(0.1, masterFade);

        const litArray = Array.from(litSet).sort((a, b) => {
          if (a === iPlay) return -1;
          if (b === iPlay) return 1;
          return emissions[a]! - emissions[b]!;
        });
        for (let j = 0; j < labels.length; j++) {
          const el = labelEls[j]!;
          const obj = labels[j]!;
          if (j < litArray.length && masterFade > 0.02) {
            const idx = litArray[j]!;
            const oa = idx * 3;
            const amp = amplitudes[idx]!;
            const tEm = emissions[idx]!;
            const freq = freqs[idx]!;
            obj.position.set(
              (posAttr.array[oa] as number) + 0.08,
              (posAttr.array[oa + 1] as number) + 0.15,
              posAttr.array[oa + 2] as number
            );
            const ampNorm = maxAmpAll > 1e-8 ? amp / maxAmpAll : amp;
            const life = Math.max(0, audioTime - tEm);
            el.querySelector(".embedding-viz__label__amp")!.textContent = ampNorm.toFixed(4);
            el.querySelector(".embedding-viz__label__t")!.textContent =
              `${tEm.toFixed(2)}s · life ${life.toFixed(2)}s`;
            el.setAttribute("title", `${(freq / 1000).toFixed(2)} kHz`);
            const vis = masterFade;
            el.style.opacity = String(0.25 + vis * 0.75);
            el.style.setProperty("--label-glow", String(0.4 + vis * 0.8));
            el.style.transform = `scale(${0.92 + vis * 0.12})`;
          } else {
            el.style.opacity = String(0.004);
            el.style.transform = "scale(0.86)";
          }
        }

        controls.update();
        composer.render();
        labelRenderer.render(scene, camera);
      };
      animate();

      const rootTag = mount.parentElement?.querySelector(".embedding-viz-stats") as HTMLElement | null;
      if (rootTag) {
        rootTag.textContent = `${points.length} frames · ≤${totalDur.toFixed(1)}s analyzed · RMS-gated playhead`;
      }

      const audio = new Audio();
      audioRef.current = audio;
      audio.loop = true;
      audio.volume = 0.7;
      playbackObjectUrl = URL.createObjectURL(
        audioFile ?? syntheticAudioToWavBlob(seed)
      );
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
      mount.addEventListener(
        "click",
        () => {
          tryPlay();
        },
        { once: true }
      );
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
      lineGeometry?.dispose();
      lineMaterial?.dispose();
      renderer?.dispose();
      if (renderer?.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
      if (labelRenderer?.domElement.parentNode === mount) {
        mount.removeChild(labelRenderer.domElement);
      }
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
      aria-label="Bird audio visualization: playback lights one analysis frame near the clock—the color and glow match that frames frequency and amplitude during chirps."
    />
  );
}
