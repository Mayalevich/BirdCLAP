/** Decode a File into an AudioBuffer using the Web Audio API. */
export async function decodeAudioFile(file: File): Promise<AudioBuffer> {
  const arrayBuf = await file.arrayBuffer();
  const ctx = new AudioContext();
  const buffer = await ctx.decodeAudioData(arrayBuf);
  await ctx.close();
  return buffer;
}

/**
 * Slice an AudioBuffer to a mono segment starting at `startS` seconds
 * with duration `durationS` seconds (clamped to the buffer length).
 */
export function sliceAudioBuffer(
  buffer: AudioBuffer,
  startS: number,
  durationS: number,
): AudioBuffer {
  const sr = buffer.sampleRate;
  const startSample = Math.floor(startS * sr);
  const length = Math.min(
    Math.ceil(durationS * sr),
    buffer.length - startSample,
  );

  const sliced = new AudioBuffer({ length, sampleRate: sr, numberOfChannels: 1 });
  const out = sliced.getChannelData(0);

  if (buffer.numberOfChannels === 1) {
    out.set(buffer.getChannelData(0).subarray(startSample, startSample + length));
  } else {
    // mix down to mono
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      const ch = buffer.getChannelData(c).subarray(startSample, startSample + length);
      for (let i = 0; i < length; i++) {
        out[i] += ch[i] / buffer.numberOfChannels;
      }
    }
  }

  return sliced;
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

/**
 * Encode an AudioBuffer as a 32-bit float PCM WAV File.
 * The backend accepts this format and resamples if needed.
 */
export function audioBufferToWavFile(buffer: AudioBuffer, name: string): File {
  const numSamples = buffer.length;
  const sampleRate = buffer.sampleRate;
  const bytesPerSample = 4; // float32

  // Mix to mono for the WAV (backend expects mono)
  const samples = new Float32Array(numSamples);
  if (buffer.numberOfChannels === 1) {
    samples.set(buffer.getChannelData(0));
  } else {
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      const ch = buffer.getChannelData(c);
      for (let i = 0; i < numSamples; i++) {
        samples[i] += ch[i] / buffer.numberOfChannels;
      }
    }
  }

  const dataSize = numSamples * bytesPerSample;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);           // fmt chunk size
  view.setUint16(20, 3, true);            // audio format: IEEE float
  view.setUint16(22, 1, true);            // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true);              // block align
  view.setUint16(34, 32, true);           // bits per sample
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  new Float32Array(buf, 44).set(samples);

  return new File([buf], name, { type: "audio/wav" });
}

/** Format seconds as m:ss */
export function fmtSeconds(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}
