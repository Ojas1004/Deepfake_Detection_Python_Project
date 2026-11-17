import { Detector } from "@/components/detector/Detector";

export default function HomePage() {
  return (
    <>
      <div className="max-w-2xl">
        <h1 className="text-4xl font-bold tracking-tight text-balance sm:text-5xl">Is this face real?</h1>
        <p className="mt-4 text-base leading-relaxed text-muted">
          Upload a photo with a face. The detector finds and aligns the face, scores it, and shows which
          areas the decision rests on.
        </p>
      </div>
      <Detector />
    </>
  );
}
