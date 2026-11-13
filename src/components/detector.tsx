"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import {
  decodeGrid,
  imageToTensor,
  MARK_COLOR,
  toImagePixels,
  type Detection,
  type Spec,
  type TrainingReport,
} from "@/lib/detector";

const SAMPLES = [
  { src: "/samples/cat.jpg", label: "Cat" },
  { src: "/samples/dog.jpg", label: "Dog" },
  { src: "/samples/horse.jpg", label: "Horse" },
  { src: "/samples/cow.jpg", label: "Cow" },
  { src: "/samples/sheep.jpg", label: "Sheep" },
  { src: "/samples/bird.jpg", label: "Bird" },
  { src: "/samples/cup.jpg", label: "No animal" },
];

type Photo = {
  src: string;
  name: string;
  width: number;
  height: number;
};

type ModelState =
  | { status: "loading" }
  | { status: "ready" }
  | { status: "error"; message: string };

export function Detector() {
  const fileRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const sessionRef = useRef<OrtSession | null>(null);
  const ortRef = useRef<OrtModule | null>(null);
  const rawRef = useRef<Float32Array | null>(null);
  const revokeRef = useRef<string | null>(null);

  const [spec, setSpec] = useState<Spec | null>(null);
  const [report, setReport] = useState<TrainingReport | null>(null);
  const [model, setModel] = useState<ModelState>({ status: "loading" });
  const [photo, setPhoto] = useState<Photo | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selected, setSelected] = useState(0);
  const [threshold, setThreshold] = useState(55);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [hasGrid, setHasGrid] = useState(false);
  const requestRef = useRef(0);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const specResponse = await fetch("/models/spec.json");
        if (!specResponse.ok) {
          throw new Error("The trained network is not in this build yet.");
        }
        const nextSpec = (await specResponse.json()) as Spec;
        // The packaged runtime is loaded as a static file. Importing it
        // through the app bundler never finishes, so the page stays on
        // "Loading network".
        const ort = (await withTimeout(
          // Served from public/ort. The bundler must not compile this file.
          import(
            /* webpackIgnore: true */
            /* turbopackIgnore: true */
            // @ts-expect-error The runtime is a static browser file, not a TypeScript module.
            "/ort/ort.wasm.min.mjs"
          ),
          15000,
          "The runtime did not load.",
        )) as OrtModule;
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.wasmPaths = "/ort/";
        const session = await withTimeout(
          ort.InferenceSession.create("/models/animal-detector.onnx", {
            executionProviders: ["wasm"],
          }),
          25000,
          "The network took too long to load.",
        );
        if (cancelled) return;
        ortRef.current = ort;
        sessionRef.current = session;
        setSpec(nextSpec);
        setModel({ status: "ready" });
      } catch (error) {
        if (cancelled) return;
        setModel({
          status: "error",
          message: error instanceof Error ? error.message : "The network failed to load.",
        });
      }

      try {
        const reportResponse = await fetch("/models/report.json");
        if (reportResponse.ok && !cancelled) {
          setReport((await reportResponse.json()) as TrainingReport);
        }
      } catch {
        // The architecture panel is optional if the report is missing.
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (revokeRef.current) URL.revokeObjectURL(revokeRef.current);
    };
  }, []);

  const pixels = useMemo(() => {
    if (!photo || !spec) return [];
    return detections.map((detection) =>
      toImagePixels(detection, photo.width, photo.height, spec.imageSize),
    );
  }, [detections, photo, spec]);

  async function runPhoto(source: CanvasImageSource, width: number, height: number) {
    const session = sessionRef.current;
    const ort = ortRef.current;
    if (!session || !ort || !spec) return;
    const request = requestRef.current + 1;
    requestRef.current = request;
    rawRef.current = null;
    setHasGrid(false);
    setBusy(true);
    setNote(null);
    try {
      const data = imageToTensor(source, width, height, spec);
      const tensor = new ort.Tensor("float32", data, [1, 3, spec.imageSize, spec.imageSize]);
      const output = await session.run({ image: tensor });
      if (request !== requestRef.current) return;
      const prediction = output.prediction;
      if (!prediction) {
        throw new Error("The network did not return a prediction.");
      }
      const raw = Float32Array.from(prediction.data as ArrayLike<number>);
      rawRef.current = raw;
      setHasGrid(true);
      const found = decodeGrid(raw, spec, threshold / 100);
      setDetections(found);
      setSelected(0);
    } catch (error) {
      if (request !== requestRef.current) return;
      rawRef.current = null;
      setHasGrid(false);
      setDetections([]);
      setNote(error instanceof Error ? error.message : "The photograph could not be read.");
    } finally {
      if (request === requestRef.current) setBusy(false);
    }
  }

  function applyThreshold(next: number) {
    setThreshold(next);
    if (!rawRef.current || !spec) return;
    const found = decodeGrid(rawRef.current, spec, next / 100);
    setDetections(found);
    setSelected(0);
  }

  function rememberPhoto(next: Photo) {
    if (revokeRef.current && revokeRef.current !== next.src) {
      URL.revokeObjectURL(revokeRef.current);
      revokeRef.current = null;
    }
    setPhoto(next);
    setDetections([]);
    setNote(null);
  }

  function openFile(file: File) {
    if (!file.type.startsWith("image/")) {
      setNote("Choose a JPEG, PNG, or WebP photograph.");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setNote("That file is over 15 MB. Try a smaller photograph.");
      return;
    }
    const url = URL.createObjectURL(file);
    revokeRef.current = url;
    const probe = new Image();
    probe.onload = () => {
      rememberPhoto({
        src: url,
        name: file.name,
        width: probe.naturalWidth,
        height: probe.naturalHeight,
      });
      void runPhoto(probe, probe.naturalWidth, probe.naturalHeight);
    };
    probe.onerror = () => {
      setNote("This file could not be read. Try a JPEG or PNG.");
    };
    probe.src = url;
  }

  function openSample(src: string, label: string) {
    const probe = new Image();
    probe.onload = () => {
      rememberPhoto({
        src,
        name: label,
        width: probe.naturalWidth,
        height: probe.naturalHeight,
      });
      void runPhoto(probe, probe.naturalWidth, probe.naturalHeight);
    };
    probe.onerror = () => setNote("That sample photograph did not load.");
    probe.src = src;
  }

  const classList = spec?.classes ?? ["bird", "cat", "dog", "horse", "sheep", "cow"];

  return (
    <div className="flex flex-1 flex-col">
      <header className="border-b border-border/80 bg-card/70">
        <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs font-medium tracking-[0.22em] text-muted-foreground uppercase">
              On-device convolutional detector
            </p>
            <h1 className="font-heading text-4xl leading-none font-medium tracking-tight text-foreground sm:text-5xl">
              Fieldmark
            </h1>
            <p className="mt-2 max-w-xl text-sm leading-6 text-muted-foreground">
              Marks birds, cats, dogs, horses, sheep, and cows. The photograph stays in
              this browser. A detection head trained in this project sits on a ResNet-18
              trunk.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={model.status === "error" ? "destructive" : "secondary"}>
              {model.status === "loading" && "Loading network"}
              {model.status === "ready" && "Network ready"}
              {model.status === "error" && "Network unavailable"}
            </Badge>
            {report ? (
              <Badge variant="outline">
                {report.parameters.toLocaleString()} weights
              </Badge>
            ) : null}
          </div>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-6xl flex-1 gap-6 px-4 py-6 lg:grid-cols-[minmax(0,1.6fr)_22rem]">
        <section
          className={`relative flex min-h-[24rem] flex-col rounded-2xl bg-card p-3 ring-1 ring-foreground/10 ${
            dragOver ? "ring-2 ring-primary" : ""
          }`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragOver(false);
            const file = event.dataTransfer.files?.[0];
            if (file) openFile(file);
          }}
        >
          {photo ? (
            <div className="flex flex-1 items-center justify-center overflow-hidden rounded-xl bg-[#d7cbb6]">
              <div className="relative inline-block max-w-full">
                <img
                  ref={imageRef}
                  src={photo.src}
                  alt={photo.name}
                  className="block max-h-[68vh] max-w-full"
                />
                <svg
                  className="pointer-events-none absolute inset-0 h-full w-full"
                  viewBox={`0 0 ${photo.width} ${photo.height}`}
                >
                  {pixels.map((box, index) => {
                    const color = MARK_COLOR[box.name] ?? "#1b3a2f";
                    const active = index === selected;
                    return (
                      <g key={`${box.name}-${index}`}>
                        <rect
                          x={box.x1}
                          y={box.y1}
                          width={Math.max(0, box.x2 - box.x1)}
                          height={Math.max(0, box.y2 - box.y1)}
                          fill={color}
                          fillOpacity={active ? 0.16 : 0.05}
                          stroke={color}
                          strokeWidth={active ? photo.width / 140 : photo.width / 220}
                        />
                        <MarkLabel
                          box={box}
                          color={color}
                          imageWidth={photo.width}
                          imageHeight={photo.height}
                        />
                      </g>
                    );
                  })}
                </svg>
              </div>
            </div>
          ) : (
            <div className="flex flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-foreground/20 bg-[#f6f0e4] px-6 text-center">
              <p className="font-heading text-3xl text-foreground">Drop a photograph</p>
              <p className="mt-2 max-w-md text-sm leading-6 text-muted-foreground">
                Fieldmark looks for six animals and draws a box around each one it is
                willing to name. A coffee cup, a car, or a lion stays unmarked.
              </p>
              <Button
                className="mt-5"
                size="lg"
                disabled={model.status !== "ready"}
                onClick={() => fileRef.current?.click()}
              >
                Choose a photograph
              </Button>
              {model.status === "loading" ? (
                <p className="mt-3 text-xs text-muted-foreground">
                  The weights are still loading into the page.
                </p>
              ) : null}
            </div>
          )}

          {photo ? (
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3 px-1">
              <p className="truncate text-sm text-muted-foreground">
                {busy ? "Reading the frame…" : photo.name}
              </p>
              <Button
                variant="outline"
                disabled={model.status !== "ready" || busy}
                onClick={() => fileRef.current?.click()}
              >
                Replace photograph
              </Button>
            </div>
          ) : null}

          <input
            ref={fileRef}
            type="file"
            accept="image/jpeg,image/png,image/webp,image/gif"
            className="sr-only"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) openFile(file);
              event.target.value = "";
            }}
          />
        </section>

        <aside className="flex flex-col gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Findings</CardTitle>
              <CardDescription>
                {detections.length === 0
                  ? "Marks above the confidence line appear here."
                  : `${detections.length} mark${detections.length === 1 ? "" : "s"} in this frame.`}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              {model.status === "error" ? (
                <p className="text-sm leading-6 text-destructive">{model.message}</p>
              ) : null}
              {note ? <p className="text-sm leading-6 text-destructive">{note}</p> : null}
              {photo && !busy && detections.length === 0 && !note ? (
                <p className="text-sm leading-6 text-muted-foreground">
                  No {classList.join(", ")} cleared {threshold}%. Lower the line if a
                  faint mark should still count.
                </p>
              ) : null}
              <ul className="flex flex-col gap-2">
                {detections.map((detection, index) => (
                  <li key={`${detection.name}-${index}`}>
                    <button
                      type="button"
                      onClick={() => setSelected(index)}
                      className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-left ring-1 ${
                        index === selected
                          ? "bg-secondary ring-foreground/20"
                          : "ring-transparent hover:bg-muted"
                      }`}
                    >
                      <span className="flex items-center gap-2">
                        <span
                          className="size-2.5 rounded-full"
                          style={{ background: MARK_COLOR[detection.name] ?? "#1b3a2f" }}
                        />
                        <span className="capitalize">{detection.name}</span>
                      </span>
                      <span className="tabular-nums text-muted-foreground">
                        {Math.round(detection.score * 100)}%
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
              <div className="pt-1">
                <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
                  <span>Confidence line</span>
                  <span className="tabular-nums">{threshold}%</span>
                </div>
                <Slider
                  min={10}
                  max={80}
                  step={1}
                  value={[threshold]}
                  disabled={!hasGrid}
                  onValueChange={(value) => {
                    const next = Array.isArray(value) ? value[0] : value;
                    applyThreshold(next);
                  }}
                />
              </div>
            </CardContent>
          </Card>

          <Card size="sm">
            <CardHeader>
              <CardTitle>What it can name</CardTitle>
              <CardDescription>
                Six classes from Pascal VOC. Other animals are outside this network.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {classList.map((name) => (
                <Badge key={name} variant="outline" className="capitalize">
                  <span
                    className="size-1.5 rounded-full"
                    style={{ background: MARK_COLOR[name] ?? "#1b3a2f" }}
                  />
                  {name}
                  {report?.perClassRecall?.[name] != null
                    ? ` ${Math.round(report.perClassRecall[name] * 100)}%`
                    : ""}
                </Badge>
              ))}
            </CardContent>
          </Card>
        </aside>
      </main>

      <section className="mx-auto w-full max-w-6xl px-4 pb-6">
        <div className="mb-3 flex items-baseline justify-between gap-3">
          <h2 className="font-heading text-xl">Try a known frame</h2>
          <p className="text-xs text-muted-foreground">Wikimedia Commons</p>
        </div>
        <div className="flex gap-3 overflow-x-auto pb-2">
          {SAMPLES.map((sample) => (
            <button
              key={sample.src}
              type="button"
              disabled={model.status !== "ready" || busy}
              onClick={() => openSample(sample.src, sample.label)}
              className="w-28 shrink-0 overflow-hidden rounded-xl bg-card text-left ring-1 ring-foreground/10 disabled:opacity-50"
            >
              <img src={sample.src} alt="" className="h-20 w-full object-cover" />
              <span className="block px-2 py-1.5 text-xs">{sample.label}</span>
            </button>
          ))}
        </div>
      </section>

      {report ? (
        <section className="border-t border-border/80 bg-card/50">
          <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-6">
            <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h2 className="font-heading text-xl">The network</h2>
                <p className="max-w-2xl text-sm leading-6 text-muted-foreground">
                  A ResNet-18 trunk, started from ImageNet, reduces a {report.imageSize}px
                  frame to a 7×7 grid. The 1×1 detection head on that grid was trained
                  here. Each cell has two predictors: a box, an objectness score, and a
                  class. On held-out photos the best checkpoint reached{" "}
                  {Math.round(report.bestRecall * 100)}% recall
                  {report.bestPrecision != null
                    ? ` and ${Math.round(report.bestPrecision * 100)}% precision`
                    : ""}
                  .
                </p>
              </div>
              <p className="text-xs text-muted-foreground">{report.dataset}</p>
            </div>
            <ol className="flex gap-2 overflow-x-auto pb-1">
              {report.layers
                .filter((layer) => layer.kind === "conv")
                .map((layer) => (
                  <li
                    key={layer.name}
                    className="min-w-36 rounded-lg bg-background px-3 py-2 ring-1 ring-foreground/10"
                  >
                    <p className="text-xs text-muted-foreground">{layer.name}</p>
                    <p className="mt-1 text-sm">
                      {layer.kernel}×{layer.kernel}
                      {layer.stride && layer.stride > 1 ? ` / ${layer.stride}` : ""}
                    </p>
                    <p className="text-xs text-muted-foreground tabular-nums">
                      {layer.inChannels} → {layer.outChannels}
                    </p>
                  </li>
                ))}
            </ol>
          </div>
        </section>
      ) : null}
    </div>
  );
}

function MarkLabel({
  box,
  color,
  imageWidth,
  imageHeight,
}: {
  box: Detection;
  color: string;
  imageWidth: number;
  imageHeight: number;
}) {
  const label = `${box.name} ${Math.round(box.score * 100)}%`;
  const fontSize = Math.max(imageWidth, imageHeight) / 42;
  const pad = fontSize * 0.35;
  const width = label.length * fontSize * 0.56 + pad * 2;
  const height = fontSize + pad * 1.4;
  const x = box.x1;
  const y = Math.max(0, box.y1 - height);
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill={color} />
      <text
        x={x + pad}
        y={y + height - pad * 0.85}
        fill="#f6f1e6"
        fontSize={fontSize}
        fontFamily="Outfit, sans-serif"
      >
        {label}
      </text>
    </g>
  );
}

function withTimeout<T>(work: Promise<T>, milliseconds: number, message: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), milliseconds);
    work.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error instanceof Error ? error : new Error(message));
      },
    );
  });
}

type OrtTensor = {
  data: Float32Array | number[];
};

type OrtSession = {
  run: (feeds: Record<string, unknown>) => Promise<Record<string, OrtTensor | undefined>>;
};

type OrtModule = {
  env: { wasm: { numThreads: number; wasmPaths: string } };
  Tensor: new (type: "float32", data: Float32Array, dims: number[]) => unknown;
  InferenceSession: {
    create: (
      url: string,
      options: { executionProviders: string[] },
    ) => Promise<OrtSession>;
  };
};
