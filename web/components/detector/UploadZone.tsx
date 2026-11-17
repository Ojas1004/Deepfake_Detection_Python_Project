"use client";

import Image from "next/image";
import { useEffect, useId, useRef, useState } from "react";
import { ImageSquare, UploadSimple } from "@phosphor-icons/react";
import { fileSize } from "@/lib/format";

const ACCEPT = ["image/jpeg", "image/png", "image/webp"];
const MAX_BYTES = 20 * 1024 * 1024;

type Props = {
  file: File | null;
  previewUrl: string | null;
  onFile: (file: File) => void;
  busy: boolean;
};

export function UploadZone({ file, previewUrl, onFile, busy }: Props) {
  const inputId = useId();
  const helpId = useId();
  const errorId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function accept(f: File | undefined | null) {
    if (!f) return;
    if (!ACCEPT.includes(f.type)) {
      setError(`“${f.name}” is not a JPEG, PNG or WebP image. Choose a photo in one of those formats.`);
      return;
    }
    if (f.size > MAX_BYTES) {
      setError(`“${f.name}” is larger than 20 MB. Choose a smaller photo.`);
      return;
    }
    setError(null);
    onFile(f);
  }

  // Paste an image from the clipboard anywhere on the page.
  useEffect(() => {
    function onPaste(e: ClipboardEvent) {
      const item = Array.from(e.clipboardData?.files ?? []).find((f) => f.type.startsWith("image/"));
      if (item) accept(item);
    }
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  });

  return (
    <div>
      <h2 className="text-xs font-semibold uppercase tracking-[0.14em]">2 · Photo</h2>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          accept(e.dataTransfer.files?.[0]);
        }}
        className={`mt-4 border transition-colors duration-150 has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-fg ${
          dragging ? "border-fg bg-surface" : file ? "border-line" : "border-dashed border-line-strong"
        }`}
      >
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={ACCEPT.join(",")}
          aria-describedby={`${helpId}${error ? ` ${errorId}` : ""}`}
          aria-invalid={error ? true : undefined}
          className="sr-only"
          onChange={(e) => {
            accept(e.target.files?.[0]);
            e.target.value = "";
          }}
        />

        {file && previewUrl ? (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-3 p-3">
            <div className="relative size-16 shrink-0 overflow-hidden bg-surface">
              <Image src={previewUrl} alt="The uploaded photo" fill unoptimized className="object-cover" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium" title={file.name}>
                {file.name}
              </p>
              <p id={helpId} className="mt-0.5 text-xs text-muted">
                {fileSize(file.size)} · drop, paste or choose another photo to replace it
              </p>
            </div>
            <label
              htmlFor={inputId}
              aria-disabled={busy || undefined}
              className="inline-flex h-11 w-full shrink-0 cursor-pointer items-center justify-center gap-2 border border-fg px-4 text-sm font-medium sm:w-auto transition-colors duration-150 hover:bg-inverse hover:text-on-inverse"
            >
              <UploadSimple aria-hidden="true" className="size-4" />
              Replace
            </label>
          </div>
        ) : (
          <label htmlFor={inputId} className="flex cursor-pointer flex-col items-center px-6 py-14 text-center">
            <ImageSquare aria-hidden="true" weight="thin" className="size-12" />
            <span className="mt-4 text-base font-semibold">Drop a photo here, or choose one</span>
            <span id={helpId} className="mt-1.5 text-sm text-muted">
              JPEG, PNG or WebP, up to 20 MB. You can also paste an image.
            </span>
            <span className="mt-6 inline-flex h-11 items-center gap-2 bg-inverse px-5 text-sm font-medium text-on-inverse">
              <UploadSimple aria-hidden="true" className="size-4" />
              Choose photo
            </span>
          </label>
        )}
      </div>
      {error && (
        <p id={errorId} role="alert" className="mt-2 text-sm font-medium">
          {error}
        </p>
      )}
    </div>
  );
}
