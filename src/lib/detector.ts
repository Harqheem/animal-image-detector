export type Spec = {
  classes: string[];
  imageSize: number;
  grid: number;
  boxesPerCell: number;
  fill: number;
  channels: number;
  mean?: number[];
  std?: number[];
};

export type Detection = {
  classId: number;
  name: string;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type LayerRow = {
  name: string;
  kind: string;
  inChannels?: number;
  outChannels?: number;
  kernel?: number;
  stride?: number;
  channels?: number;
};

export type TrainingReport = {
  dataset: string;
  classes: string[];
  parameters: number;
  epochs: number;
  bestRecall: number;
  bestPrecision?: number;
  bestF1?: number;
  perClassRecall: Record<string, number>;
  layers: LayerRow[];
  imageSize: number;
};

export const MARK_COLOR: Record<string, string> = {
  bird: "#1d6f8a",
  cat: "#c45c26",
  dog: "#246246",
  horse: "#8a4b2a",
  sheep: "#31557d",
  cow: "#6d3d6a",
};

export function letterboxGeometry(width: number, height: number, size: number) {
  const scale = size / Math.max(width, height);
  const resizedW = Math.max(1, Math.min(size, Math.floor(width * scale)));
  const resizedH = Math.max(1, Math.min(size, Math.floor(height * scale)));
  const padX = Math.floor((size - resizedW) / 2);
  const padY = Math.floor((size - resizedH) / 2);
  return { scale, resizedW, resizedH, padX, padY };
}

export function imageToTensor(
  image: CanvasImageSource,
  width: number,
  height: number,
  spec: Spec,
): Float32Array {
  const size = spec.imageSize;
  const { resizedW, resizedH, padX, padY } = letterboxGeometry(width, height, size);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("This browser could not read the photograph.");
  }
  context.fillStyle = `rgb(${spec.fill}, ${spec.fill}, ${spec.fill})`;
  context.fillRect(0, 0, size, size);
  context.drawImage(image, padX, padY, resizedW, resizedH);
  const pixels = context.getImageData(0, 0, size, size).data;
  const plane = size * size;
  const data = new Float32Array(3 * plane);
  const mean = spec.mean ?? [0.485, 0.456, 0.406];
  const std = spec.std ?? [0.229, 0.224, 0.225];
  for (let index = 0; index < plane; index += 1) {
    data[index] = (pixels[index * 4] / 255 - mean[0]) / std[0];
    data[plane + index] = (pixels[index * 4 + 1] / 255 - mean[1]) / std[1];
    data[plane * 2 + index] = (pixels[index * 4 + 2] / 255 - mean[2]) / std[2];
  }
  return data;
}

export function decodeGrid(
  data: Float32Array,
  spec: Spec,
  scoreThreshold: number,
  iouThreshold = 0.3,
): Detection[] {
  const { grid, boxesPerCell, channels, classes } = spec;
  const expected = grid * grid * boxesPerCell * channels;
  if (data.length < expected) {
    throw new Error("The network returned a grid this page does not understand.");
  }

  const candidates: Detection[] = [];
  for (let cellY = 0; cellY < grid; cellY += 1) {
    for (let cellX = 0; cellX < grid; cellX += 1) {
      for (let slot = 0; slot < boxesPerCell; slot += 1) {
        const base = ((cellY * grid + cellX) * boxesPerCell + slot) * channels;
        const centerX = (sigmoid(data[base]) + cellX) / grid;
        const centerY = (sigmoid(data[base + 1]) + cellY) / grid;
        const width = sigmoid(data[base + 2]);
        const height = sigmoid(data[base + 3]);
        const objectness = sigmoid(data[base + 4]);
        const logits: number[] = [];
        for (let classIndex = 0; classIndex < classes.length; classIndex += 1) {
          logits.push(data[base + 5 + classIndex]);
        }
        const probabilities = softmax(logits);
        let classId = 0;
        for (let classIndex = 1; classIndex < probabilities.length; classIndex += 1) {
          if (probabilities[classIndex] > probabilities[classId]) {
            classId = classIndex;
          }
        }
        const score = objectness * probabilities[classId];
        if (score < scoreThreshold) {
          continue;
        }
        const x1 = clamp(centerX - width / 2, 0, 1);
        const y1 = clamp(centerY - height / 2, 0, 1);
        const x2 = clamp(centerX + width / 2, 0, 1);
        const y2 = clamp(centerY + height / 2, 0, 1);
        if (x2 - x1 < 0.001 || y2 - y1 < 0.001) {
          continue;
        }
        candidates.push({
          classId,
          name: classes[classId],
          score,
          x1,
          y1,
          x2,
          y2,
        });
      }
    }
  }
  return nms(candidates, iouThreshold);
}

export function toImagePixels(
  detection: Detection,
  width: number,
  height: number,
  size: number,
): Detection {
  const { scale, padX, padY } = letterboxGeometry(width, height, size);
  const x = (value: number) => clamp((value * size - padX) / scale, 0, width);
  const y = (value: number) => clamp((value * size - padY) / scale, 0, height);
  return {
    ...detection,
    x1: x(detection.x1),
    y1: y(detection.y1),
    x2: x(detection.x2),
    y2: y(detection.y2),
  };
}

function nms(detections: Detection[], iouThreshold: number): Detection[] {
  const pool = [...detections].sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];
  while (pool.length > 0) {
    const best = pool.shift();
    if (!best) break;
    kept.push(best);
    for (let index = pool.length - 1; index >= 0; index -= 1) {
      if (iou(best, pool[index]) >= iouThreshold) {
        pool.splice(index, 1);
      }
    }
  }
  return kept;
}

function iou(a: Detection, b: Detection): number {
  const left = Math.max(a.x1, b.x1);
  const top = Math.max(a.y1, b.y1);
  const right = Math.min(a.x2, b.x2);
  const bottom = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  if (intersection <= 0) return 0;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  const union = areaA + areaB - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function softmax(logits: number[]): number[] {
  const peak = Math.max(...logits);
  const elevated = logits.map((value) => Math.exp(value - peak));
  const sum = elevated.reduce((total, value) => total + value, 0);
  return elevated.map((value) => value / sum);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
