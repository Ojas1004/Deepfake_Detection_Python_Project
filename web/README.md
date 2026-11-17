# Web front-end (Next.js)

A black-and-white web UI for the detector: upload a face photo, pick a model, and see the verdict,
P(fake), a heatmap of the evidence and a plain-language explanation. A second page shows the benchmark results.

It talks to the Python API in `../src/api.py`, which runs the models.

## Run

Two terminals, both started from the project folder:

```fish
# 1. the Python API (models run here, on the GPU)
.venv/bin/python -m src.api                  # http://127.0.0.1:8000

# 2. the web UI
cd web
npm install                                  # first time only
npm run dev                                  # http://localhost:3000
```

For a production build: `npm run build`, then `npm run start`.

If the API runs somewhere else, set `NEXT_PUBLIC_API_URL` (see `.env.example`). The API only accepts
requests from `http://localhost:3000` and `http://127.0.0.1:3000`; set `WEB_ORIGINS` (comma-separated)
when starting it to allow other addresses.

## Structure

```
app/layout.tsx                  fonts (Inter, JetBrains Mono), header, skip link, footer
app/page.tsx                    Detect page
app/benchmarks/page.tsx         Benchmarks page (server-rendered from /api/benchmarks)
components/detector/            model picker, upload, verdict, heatmap, region table
components/benchmarks/          results table
lib/api.ts                      API types and calls
app/globals.css                 black-and-white design tokens (light and dark)
```

## Design

- **Black and white only.** Greys are used for hierarchy and meet WCAG AA contrast. Dark mode inverts the same tokens.
- **Verdicts never rely on colour:** FAKE is a solid black block, REAL an outlined box, "leans fake – uncertain" a
  hatched box. Each also has its own icon and word.
- **Heatmaps in greyscale:** areas with little evidence are darkened, so the areas the model relied on stay bright.
  The highlight strength can be adjusted.
- Keyboard accessible, with visible focus rings, `prefers-reduced-motion` support and a phone layout
  (a native select replaces the model list).
