export type Point = { x: number; y: number };
export type Quad = [Point, Point, Point, Point]; // TL, TR, BR, BL order

function gaussianElimination(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];

    const pivot = M[col][col];
    if (Math.abs(pivot) < 1e-12) continue;

    for (let row = col + 1; row < n; row++) {
      const f = M[row][col] / pivot;
      for (let k = col; k <= n; k++) M[row][k] -= f * M[col][k];
    }
  }

  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i];
  }
  return x;
}

/**
 * Compute a 3×3 homography (9-element row-major array, h8=1) that maps
 * each src[i] to dst[i].
 * src = corners of recording element in element space (TL,TR,BR,BL)
 * dst = destination corners in canvas/display space
 */
export function computeHomography(src: Quad, dst: Quad): number[] {
  const A: number[][] = [];
  const b: number[] = [];

  for (let i = 0; i < 4; i++) {
    const { x: sx, y: sy } = src[i];
    const { x: dx, y: dy } = dst[i];

    A.push([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy]);
    b.push(dx);

    A.push([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy]);
    b.push(dy);
  }

  const h = gaussianElimination(A, b);
  return [...h, 1]; // normalize: h8 = 1
}

/**
 * Convert a 9-element row-major homography [h0..h8] to a CSS matrix3d string.
 * Apply with transform-origin: 0 0 on the target element.
 *
 * Row-major H:
 *   [h0 h1 h2]
 *   [h3 h4 h5]
 *   [h6 h7 h8]
 *
 * CSS matrix3d (column-major 4×4 that encodes the projective map):
 *   matrix3d(h0, h3, 0, h6,
 *            h1, h4, 0, h7,
 *            0,  0,  1, 0,
 *            h2, h5, 0, h8)
 */
export function homographyToCSS(h: number[]): string {
  return (
    `matrix3d(` +
    `${h[0]},${h[3]},0,${h[6]},` +
    `${h[1]},${h[4]},0,${h[7]},` +
    `0,0,1,0,` +
    `${h[2]},${h[5]},0,${h[8]})`
  );
}

/**
 * Compute the inverse of a 3×3 homography (to map display coords → element coords).
 * Used for image export pixel-sampling.
 */
export function invertHomography(h: number[]): number[] {
  const [a, b, c, d, e, f, g, hh, i] = h;
  const det = a * (e * i - f * hh) - b * (d * i - f * g) + c * (d * hh - e * g);
  if (Math.abs(det) < 1e-12) return h;
  const inv = [
    (e * i - f * hh) / det,
    (c * hh - b * i) / det,
    (b * f - c * e) / det,
    (f * g - d * i) / det,
    (a * i - c * g) / det,
    (c * d - a * f) / det,
    (d * hh - e * g) / det,
    (b * g - a * hh) / det,
    (a * e - b * d) / det,
  ];
  return inv;
}

/** Apply homography to a single point. */
export function applyHomography(h: number[], p: Point): Point {
  const w = h[6] * p.x + h[7] * p.y + h[8];
  return {
    x: (h[0] * p.x + h[1] * p.y + h[2]) / w,
    y: (h[3] * p.x + h[4] * p.y + h[5]) / w,
  };
}

/** Default corners: centered quad at given padding fraction. */
export function defaultCorners(w: number, h: number, pad = 0.15): Quad {
  const px = w * pad, py = h * pad;
  return [
    { x: px, y: py },
    { x: w - px, y: py },
    { x: w - px, y: h - py },
    { x: px, y: h - py },
  ];
}
