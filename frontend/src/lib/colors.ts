type RGB = [number, number, number];

// Global bounding box stroke color (change here to update all tabs)
export const BBOX_COLOR = "#2CFF05";
// Semi-transparent version used for track lines
export const BBOX_TRACK_COLOR = "rgba(59,130,246,0.5)";

const CLASS_COLORS: Record<string, RGB> = {
  person: [34, 197, 94],
  car: [59, 130, 246],
  truck: [249, 115, 22],
  bus: [168, 85, 247],
  motorcycle: [236, 72, 153],
  bicycle: [20, 184, 166],
  dog: [234, 179, 8],
  cat: [239, 68, 68],
  default: [156, 163, 175],
};

export function getClassColor(className: string): RGB {
  return CLASS_COLORS[className.toLowerCase()] ?? CLASS_COLORS.default;
}

export function getClassColorHex(className: string): string {
  const [r, g, b] = getClassColor(className);
  return `rgb(${r},${g},${b})`;
}

export function speedToColor(speed: number, maxSpeed: number): RGB {
  const t = maxSpeed > 0 ? Math.min(1, speed / maxSpeed) : 0;
  if (t < 0.25) {
    const s = t / 0.25;
    return [
      Math.round(59 * (1 - s) + 6 * s),
      Math.round(130 * (1 - s) + 182 * s),
      Math.round(246 * (1 - s) + 212 * s),
    ];
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25;
    return [
      Math.round(6 * (1 - s) + 34 * s),
      Math.round(182 * (1 - s) + 197 * s),
      Math.round(212 * (1 - s) + 94 * s),
    ];
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25;
    return [
      Math.round(34 * (1 - s) + 234 * s),
      Math.round(197 * (1 - s) + 179 * s),
      Math.round(94 * (1 - s) + 8 * s),
    ];
  } else {
    const s = (t - 0.75) / 0.25;
    return [
      Math.round(234 * (1 - s) + 239 * s),
      Math.round(179 * (1 - s) + 68 * s),
      Math.round(8 * (1 - s) + 68 * s),
    ];
  }
}

export function speedToColorHex(speed: number, maxSpeed: number): string {
  const [r, g, b] = speedToColor(speed, maxSpeed);
  return `rgb(${r},${g},${b})`;
}

const CLUSTER_PALETTE: RGB[] = [
  [99, 102, 241],
  [14, 165, 233],
  [16, 185, 129],
  [245, 158, 11],
  [239, 68, 68],
  [217, 70, 239],
  [20, 184, 166],
  [251, 146, 60],
  [52, 211, 153],
  [167, 139, 250],
];

const NOISE_COLOR: RGB = [120, 120, 120];

export function getClusterColor(clusterId: number): RGB {
  if (clusterId < 0) return NOISE_COLOR;
  return CLUSTER_PALETTE[clusterId % CLUSTER_PALETTE.length];
}

export function getClusterColorHex(clusterId: number): string {
  const [r, g, b] = getClusterColor(clusterId);
  return `rgb(${r},${g},${b})`;
}

// Time gradient: Plasma colormap — dark violet (oldest) → bright yellow (most recent)
const TIME_STOPS: RGB[] = [
  [13,   8, 135],   // t=0.00 dark violet  (#0D0887)
  [126,  3, 168],   // t=0.33 purple       (#7E03A8)
  [240, 100,  61],  // t=0.67 orange-red   (#F0643D)
  [240, 249,  33],  // t=1.00 bright yellow (#F0F921)
];

export function timeToColor(timestamp: number, minTime: number, maxTime: number): RGB {
  const range = maxTime - minTime;
  const t = range > 0 ? Math.min(1, Math.max(0, (timestamp - minTime) / range)) : 0;
  const segments = TIME_STOPS.length - 1;
  const scaled = t * segments;
  const i = Math.min(segments - 1, Math.floor(scaled));
  const s = scaled - i;
  const [r1, g1, b1] = TIME_STOPS[i];
  const [r2, g2, b2] = TIME_STOPS[i + 1];
  return [
    Math.round(r1 * (1 - s) + r2 * s),
    Math.round(g1 * (1 - s) + g2 * s),
    Math.round(b1 * (1 - s) + b2 * s),
  ];
}

export function timeToColorHex(timestamp: number, minTime: number, maxTime: number): string {
  const [r, g, b] = timeToColor(timestamp, minTime, maxTime);
  return `rgb(${r},${g},${b})`;
}
