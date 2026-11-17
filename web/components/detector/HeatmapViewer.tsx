"use client";

import Image from "next/image";
import { useEffect, useId, useRef, useState } from "react";
import type { Analysis } from "@/lib/api";

type Target = "fake" | "real";

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new window.Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

/**
 * Black-and-white "spotlight" heatmap: the photo is shown in greyscale, and areas
 * with little evidence are darkened, so the areas the model relied on stay bright.
 */
export function HeatmapViewer({ result }: { result: Analysis }) {
  const [target, setTarget] = useState<Target>(result.verdict === "real" ? "real" : "fake");
  const [strength, setStrength] = useState(0.8);
  const [images, setImages] = useState<{ base: HTMLImageElement; fake: HTMLImageElement; real: HTMLImageElement } | null>(null);
  const [peak, setPeak] = useState(1);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sliderId = useId();

  useEffect(() => {
    let cancelled = false;
    Promise.all([loadImage(result.image), loadImage(result.heat_fake), loadImage(result.heat_real)]).then(
      ([base, fake, real]) => !cancelled && setImages({ base, fake, real }),
    );
    return () => {
      cancelled = true;
    };
  }, [result.image, result.heat_fake, result.heat_real]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !images) return;
    const { base } = images;
    const heat = target === "fake" ? images.fake : images.real;
    const w = base.naturalWidth;
    const h = base.naturalHeight;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    ctx.drawImage(heat, 0, 0, w, h);
    const heatData = ctx.getImageData(0, 0, w, h).data;
    ctx.drawImage(base, 0, 0, w, h);
    const frame = ctx.getImageData(0, 0, w, h);
    const px = frame.data;
    let max = 0;
    for (let i = 0; i < px.length; i += 4) {
      const e = heatData[i] / 255;
      if (e > max) max = e;
      const lum = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
      const v = lum * (1 - strength * (1 - e) * 0.92);
      px[i] = px[i + 1] = px[i + 2] = v;
    }
    ctx.putImageData(frame, 0, 0);
    setPeak(max);
  }, [images, target, strength]);

  const word = target.toUpperCase();
  return (
    <div>
      <h2 className="text-xs font-semibold uppercase tracking-[0.14em]">4 · Where the evidence is</h2>

      <div className="mt-4 flex flex-wrap items-end justify-between gap-4">
        <div role="radiogroup" aria-label="Show evidence for" className="inline-flex border border-fg">
          {(["fake", "real"] as const).map((t) => (
            <button
              key={t}
              type="button"
              role="radio"
              aria-checked={target === t}
              onClick={() => setTarget(t)}
              className={`h-11 cursor-pointer px-4 text-sm font-medium transition-colors duration-150 ${
                target === t ? "bg-inverse text-on-inverse" : "hover:bg-surface"
              }`}
            >
              Evidence for {t === "fake" ? "FAKE" : "REAL"}
            </button>
          ))}
        </div>
        <div className="w-full max-w-60">
          <label htmlFor={sliderId} className="flex justify-between text-xs text-muted">
            <span>Highlight strength</span>
            <span className="font-mono tabular-nums">{Math.round(strength * 100)}%</span>
          </label>
          <input
            id={sliderId}
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={strength}
            onChange={(e) => setStrength(Number(e.target.value))}
            className="mt-2 h-6 w-full cursor-pointer accent-[var(--fg)]"
          />
        </div>
      </div>

      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <figure>
          <div className="relative aspect-square overflow-hidden bg-surface">
            <Image src={result.image} alt="The image the model analysed" fill unoptimized className="object-contain" />
          </div>
          <figcaption className="mt-2 text-xs text-muted">{result.image_caption}</figcaption>
        </figure>
        <figure>
          <div className="relative aspect-square overflow-hidden bg-surface">
            <canvas
              ref={canvasRef}
              role="img"
              aria-label={`${result.heatmap} for ${word}: brighter areas carry more evidence for ${word}.`}
              className="absolute inset-0 size-full object-contain"
            />
          </div>
          <figcaption className="mt-2 text-xs leading-relaxed text-muted">
            {result.heatmap}: bright = strongest evidence for {word}; dark = little or none.
            {peak < 0.05 && <strong className="block text-fg">The model found almost no evidence for {word} here.</strong>}
          </figcaption>
        </figure>
      </div>

      <details className="mt-4 text-sm text-muted">
        <summary className="cursor-pointer py-2 font-medium text-fg">How this map is made</summary>
        <p className="mt-1 max-w-prose leading-relaxed">
          {result.heatmap === "Grad-CAM"
            ? "Grad-CAM reads the network's last convolutional layer and weights each feature map by how strongly it pushed the score towards FAKE or REAL."
            : "The image is split into an 8 × 8 grid. Each block is blurred in turn and the model is run again; blocks whose blurring moves the score most hold the evidence."}{" "}
          It shows where the evidence is, not what it is.
        </p>
      </details>
    </div>
  );
}
