// Typed client for the Pool Vision backend.

export interface Ball {
  id: number;
  x: number;
  y: number;
  radius: number;
  is_cue: boolean;
  color_hsv?: number[] | null;
}

export interface Pocket {
  name: string;
  x: number;
  y: number;
}

export interface Spin {
  hit_x: number; // -1 left .. +1 right (fraction of cue-ball radius)
  hit_y: number; // -1 top  .. +1 bottom
  zone: string; // e.g. "Top/Follow", "Stun/Center", "Draw"
  tip: string;
}

export interface Shot {
  target_ball: number;
  target_pocket: string;
  difficulty: number;
  success_rate: number;
  cue: [number, number];
  ghost: [number, number];
  contact: [number, number];
  object_center: [number, number];
  pocket: [number, number];
  spin: Spin | null;
}

export interface AnalyzeResult {
  table_detected: boolean;
  width: number;
  height: number;
  cue_ball: Ball | null;
  balls: Ball[];
  pockets: Pocket[];
  shot: Shot | null;
  candidates: Shot[];
  overlay_png: string; // base64 PNG
  warped_png: string; // base64 PNG of the clean warped table
}

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8090";

// Corners are TL, TR, BR, BL, normalized to [0,1] (values may fall outside
// when a corner is off-frame). Omit to let the server auto-detect the table.
export type Corners = [number, number][];

export async function analyzeImage(
  file: File,
  corners?: Corners,
  targetBall?: number,
): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append("image", file);
  if (corners && corners.length === 4) {
    form.append("corners", JSON.stringify(corners));
  }
  if (targetBall != null) {
    form.append("target_ball", String(targetBall));
  }
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`analyze failed (${res.status}): ${text}`);
  }
  return res.json();
}
