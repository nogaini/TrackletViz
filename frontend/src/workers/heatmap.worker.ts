/**
 * Web Worker: heatmap grid accumulation.
 *
 * Input message:
 *   { bboxes: Array<{x1,y1,x2,y2}>, vw: number, vh: number, gridW: number, gridH: number }
 *
 * Output message:
 *   { grid: Float32Array, maxVal: number }
 */

interface BBoxInput {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface WorkerInput {
  bboxes: BBoxInput[];
  vw: number;
  vh: number;
  gridW: number;
  gridH: number;
}

self.onmessage = (e: MessageEvent<WorkerInput>) => {
  const { bboxes, vw, vh, gridW, gridH } = e.data;
  const grid = new Float32Array(gridW * gridH);

  for (const box of bboxes) {
    const gx1 = Math.max(0, Math.floor((box.x1 / vw) * gridW));
    const gy1 = Math.max(0, Math.floor((box.y1 / vh) * gridH));
    const gx2 = Math.min(gridW, Math.ceil((box.x2 / vw) * gridW));
    const gy2 = Math.min(gridH, Math.ceil((box.y2 / vh) * gridH));
    for (let gy = gy1; gy < gy2; gy++) {
      for (let gx = gx1; gx < gx2; gx++) {
        grid[gy * gridW + gx] += 1;
      }
    }
  }

  let maxVal = 1;
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] > maxVal) maxVal = grid[i];
  }

  // Transfer the buffer to avoid copying
  self.postMessage({ grid, maxVal }, [grid.buffer]);
};
