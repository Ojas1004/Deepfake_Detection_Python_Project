"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Detect" },
  { href: "/benchmarks", label: "Benchmarks" },
];

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="border-b border-line">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-3 text-sm font-semibold tracking-tight">
          <span aria-hidden="true" className="grid size-7 place-items-center bg-inverse text-on-inverse">
            <span className="block size-2.5 rounded-full border-2 border-current" />
          </span>
          <span>Deepfake Face Detector</span>
        </Link>
        <nav aria-label="Main">
          <ul className="flex gap-1">
            {NAV.map((item) => {
              const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    aria-current={active ? "page" : undefined}
                    className={`inline-flex h-11 items-center px-3 text-sm transition-colors duration-150 ${
                      active
                        ? "font-semibold underline decoration-2 underline-offset-[10px]"
                        : "text-muted hover:text-fg"
                    }`}
                  >
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
      </div>
    </header>
  );
}
