import type { Quad } from './homography';

export interface DevicePreset {
  id: string;
  label: string;
  svgWidth: number;
  svgHeight: number;
  /** Exact screen corners in SVG coordinate space: TL,TR,BR,BL */
  screenCorners: Quad;
  buildSVG: () => string;
}

// ─── MacBook Pro ──────────────────────────────────────────────────────────────
const MB_W = 1400, MB_H = 880;
// Screen area inside bezel
const MB_SX = 168, MB_SY = 52, MB_SW = 1064, MB_SH = 660;

function macbookSVG(): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${MB_W}" height="${MB_H}" viewBox="0 0 ${MB_W} ${MB_H}">
  <defs>
    <linearGradient id="lid" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#D4D4D8"/>
      <stop offset="100%" stop-color="#A1A1AA"/>
    </linearGradient>
    <linearGradient id="base" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#E4E4E7"/>
      <stop offset="100%" stop-color="#C4C4C7"/>
    </linearGradient>
  </defs>
  <!-- Lid body -->
  <rect x="80" y="20" width="1240" height="760" rx="14" fill="url(#lid)" stroke="#8A8A90" stroke-width="1.5"/>
  <!-- Inner bezel -->
  <rect x="140" y="42" width="1120" height="700" rx="4" fill="#1A1A1A"/>
  <!-- Screen area (black placeholder — recording fills here) -->
  <rect x="${MB_SX}" y="${MB_SY}" width="${MB_SW}" height="${MB_SH}" fill="#000"/>
  <!-- Camera dot -->
  <circle cx="700" cy="46" r="4" fill="#3A3A3A"/>
  <!-- Hinge -->
  <rect x="80" y="776" width="1240" height="12" rx="2" fill="#8A8A90"/>
  <!-- Base -->
  <rect x="30" y="788" width="1340" height="70" rx="6" fill="url(#base)" stroke="#A1A1AA" stroke-width="1"/>
  <!-- Keyboard area (simplified) -->
  <rect x="220" y="800" width="960" height="42" rx="4" fill="#D4D4D8" opacity="0.5"/>
  <!-- Trackpad -->
  <rect x="570" y="808" width="260" height="28" rx="5" fill="#C4C4C7" stroke="#A1A1AA" stroke-width="0.5"/>
  <!-- Apple logo (outline) -->
  <text x="700" y="432" text-anchor="middle" fill="#2A2A2A" font-family="system-ui" font-size="52" opacity="0.2"></text>
</svg>`;
}

// ─── iPhone 15 ────────────────────────────────────────────────────────────────
const IP_W = 430, IP_H = 940;
const IP_SX = 18, IP_SY = 52, IP_SW = 394, IP_SH = 836;

function iphoneSVG(): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${IP_W}" height="${IP_H}" viewBox="0 0 ${IP_W} ${IP_H}">
  <defs>
    <linearGradient id="body" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#2A2A2E"/>
      <stop offset="100%" stop-color="#1A1A1E"/>
    </linearGradient>
    <clipPath id="screen-clip">
      <rect x="${IP_SX}" y="${IP_SY}" width="${IP_SW}" height="${IP_SH}" rx="40"/>
    </clipPath>
  </defs>
  <!-- Body -->
  <rect x="0" y="0" width="${IP_W}" height="${IP_H}" rx="50" fill="url(#body)" stroke="#444" stroke-width="1"/>
  <!-- Side buttons -->
  <rect x="-2" y="180" width="4" height="60" rx="2" fill="#555"/>
  <rect x="-2" y="260" width="4" height="80" rx="2" fill="#555"/>
  <rect x="-2" y="360" width="4" height="80" rx="2" fill="#555"/>
  <rect x="${IP_W - 2}" y="220" width="4" height="120" rx="2" fill="#555"/>
  <!-- Screen -->
  <rect x="${IP_SX}" y="${IP_SY}" width="${IP_SW}" height="${IP_SH}" rx="40" fill="#000"/>
  <!-- Dynamic Island -->
  <rect x="162" y="60" width="106" height="34" rx="17" fill="#000"/>
  <!-- Home indicator -->
  <rect x="165" y="900" width="100" height="5" rx="3" fill="#555" opacity="0.8"/>
</svg>`;
}

// ─── iPad Pro ─────────────────────────────────────────────────────────────────
const IPAD_W = 820, IPAD_H = 1100;
const IPAD_SX = 50, IPAD_SY = 78, IPAD_SW = 720, IPAD_SH = 944;

function ipadSVG(): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${IPAD_W}" height="${IPAD_H}" viewBox="0 0 ${IPAD_W} ${IPAD_H}">
  <defs>
    <linearGradient id="ipad-body" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#2C2C30"/>
      <stop offset="100%" stop-color="#1C1C20"/>
    </linearGradient>
  </defs>
  <!-- Body -->
  <rect x="0" y="0" width="${IPAD_W}" height="${IPAD_H}" rx="24" fill="url(#ipad-body)" stroke="#444" stroke-width="1"/>
  <!-- Side button -->
  <rect x="${IPAD_W - 2}" y="120" width="3" height="60" rx="2" fill="#555"/>
  <!-- Volume -->
  <rect x="-1" y="200" width="3" height="50" rx="2" fill="#555"/>
  <rect x="-1" y="270" width="3" height="50" rx="2" fill="#555"/>
  <!-- Screen -->
  <rect x="${IPAD_SX}" y="${IPAD_SY}" width="${IPAD_SW}" height="${IPAD_SH}" fill="#000"/>
  <!-- Front camera -->
  <circle cx="${IPAD_W / 2}" cy="42" r="5" fill="#333"/>
  <!-- Home bar -->
  <rect x="${IPAD_W / 2 - 50}" y="${IPAD_H - 14}" width="100" height="5" rx="3" fill="#555" opacity="0.6"/>
</svg>`;
}

// ─── Browser Window ───────────────────────────────────────────────────────────
const BR_W = 1400, BR_H = 900;
const BR_SX = 0, BR_SY = 84, BR_SW = 1400, BR_SH = 816;

function browserSVG(): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${BR_W}" height="${BR_H}" viewBox="0 0 ${BR_W} ${BR_H}">
  <!-- Window chrome -->
  <rect x="0" y="0" width="${BR_W}" height="${BR_H}" rx="10" fill="#1E1E26" stroke="#333" stroke-width="1"/>
  <!-- Title bar -->
  <rect x="0" y="0" width="${BR_W}" height="44" rx="10" fill="#252530"/>
  <rect x="0" y="34" width="${BR_W}" height="10" fill="#252530"/>
  <!-- Traffic lights -->
  <circle cx="20" cy="22" r="7" fill="#FF5F57"/>
  <circle cx="40" cy="22" r="7" fill="#FEBC2E"/>
  <circle cx="60" cy="22" r="7" fill="#28C840"/>
  <!-- Tab bar -->
  <rect x="0" y="44" width="${BR_W}" height="40" fill="#1C1C24"/>
  <!-- Active tab -->
  <rect x="88" y="48" width="200" height="32" rx="6" fill="#252530"/>
  <text x="140" y="68" fill="#AAAACC" font-family="system-ui" font-size="12">localhost:5173</text>
  <!-- Address bar -->
  <rect x="0" y="50" width="${BR_W}" height="34" fill="#1C1C24"/>
  <rect x="300" y="55" width="800" height="24" rx="12" fill="#12121A" stroke="#333" stroke-width="1"/>
  <text x="700" y="72" text-anchor="middle" fill="#7070A0" font-family="system-ui" font-size="12">https://your-app.vercel.app</text>
  <!-- Content area -->
  <rect x="${BR_SX}" y="${BR_SY}" width="${BR_SW}" height="${BR_SH}" fill="#0A0A14"/>
</svg>`;
}

// ─── Exports ──────────────────────────────────────────────────────────────────

export const DEVICES: DevicePreset[] = [
  {
    id: 'macbook',
    label: 'MacBook Pro',
    svgWidth: MB_W, svgHeight: MB_H,
    screenCorners: [
      { x: MB_SX,         y: MB_SY },
      { x: MB_SX + MB_SW, y: MB_SY },
      { x: MB_SX + MB_SW, y: MB_SY + MB_SH },
      { x: MB_SX,         y: MB_SY + MB_SH },
    ],
    buildSVG: macbookSVG,
  },
  {
    id: 'iphone',
    label: 'iPhone 15',
    svgWidth: IP_W, svgHeight: IP_H,
    screenCorners: [
      { x: IP_SX,         y: IP_SY },
      { x: IP_SX + IP_SW, y: IP_SY },
      { x: IP_SX + IP_SW, y: IP_SY + IP_SH },
      { x: IP_SX,         y: IP_SY + IP_SH },
    ],
    buildSVG: iphoneSVG,
  },
  {
    id: 'ipad',
    label: 'iPad Pro',
    svgWidth: IPAD_W, svgHeight: IPAD_H,
    screenCorners: [
      { x: IPAD_SX,           y: IPAD_SY },
      { x: IPAD_SX + IPAD_SW, y: IPAD_SY },
      { x: IPAD_SX + IPAD_SW, y: IPAD_SY + IPAD_SH },
      { x: IPAD_SX,           y: IPAD_SY + IPAD_SH },
    ],
    buildSVG: ipadSVG,
  },
  {
    id: 'browser',
    label: 'Browser',
    svgWidth: BR_W, svgHeight: BR_H,
    screenCorners: [
      { x: BR_SX,         y: BR_SY },
      { x: BR_SX + BR_SW, y: BR_SY },
      { x: BR_SX + BR_SW, y: BR_SY + BR_SH },
      { x: BR_SX,         y: BR_SY + BR_SH },
    ],
    buildSVG: browserSVG,
  },
];

export function svgToDataURL(svg: string): string {
  return 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svg)));
}
