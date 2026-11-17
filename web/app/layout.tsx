import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { SiteHeader } from "@/components/SiteHeader";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains", display: "swap" });

export const metadata: Metadata = {
  title: { default: "Deepfake Face Detector", template: "%s · Deepfake Face Detector" },
  description: "Upload a face photo and see whether it is a real photograph or AI-generated, and why.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0a0a" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable}`}>
      <body className="min-h-dvh">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:bg-inverse focus:px-4 focus:py-2 focus:text-on-inverse"
        >
          Skip to main content
        </a>
        <SiteHeader />
        <main id="main" className="mx-auto max-w-6xl px-4 pb-24 pt-10 sm:px-6">
          {children}
        </main>
        <footer className="border-t border-line">
          <div className="mx-auto max-w-6xl px-4 py-6 text-xs leading-relaxed text-muted sm:px-6">
            A student project. No detector is reliable on the most convincing modern fakes: treat every
            verdict as one piece of evidence, not proof.
          </div>
        </footer>
      </body>
    </html>
  );
}
