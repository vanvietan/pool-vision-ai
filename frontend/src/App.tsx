import { useRef, useState } from "react";
import {
  analyzeImage,
  type AnalyzeResult,
  type Corners,
  type Spin,
} from "./api";

// Padding (fraction of image size) around the image so corners that sit
// off-frame or occluded can still be placed in the margin.
const PAD = 0.18;
const CANVAS_W = 700;
const CORNER_LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"];

type Pt = { x: number; y: number }; // normalized to the image; may be <0 or >1

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [aspect, setAspect] = useState(0.5); // h/w
  const [corners, setCorners] = useState<Pt[]>([]);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [targetBall, setTargetBall] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);

  const imgH = CANVAS_W * aspect;
  const padX = CANVAS_W * PAD;
  const padY = imgH * PAD;

  function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setCorners([]);
    setResult(null);
    setTargetBall(null);
    setError(null);
  }

  // pointer (client) -> normalized image coords (account for padding)
  function toNorm(clientX: number, clientY: number): Pt {
    const r = boxRef.current!.getBoundingClientRect();
    return {
      x: (clientX - r.left - padX) / CANVAS_W,
      y: (clientY - r.top - padY) / imgH,
    };
  }

  function onCanvasClick(e: React.MouseEvent) {
    if (corners.length >= 4) return;
    const p = toNorm(e.clientX, e.clientY);
    const next = [...corners, p];
    // after 3 clicks, auto-place the 4th via parallelogram completion
    if (next.length === 3) {
      const [a, b, c] = next;
      next.push({ x: a.x + c.x - b.x, y: a.y + c.y - b.y });
    }
    setCorners(next);
  }

  function dragHandle(i: number, e: React.PointerEvent) {
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);
    const move = (ev: PointerEvent) => {
      const p = toNorm(ev.clientX, ev.clientY);
      setCorners((cs) => cs.map((c, j) => (j === i ? p : c)));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  async function analyze(target?: number | null) {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const c =
        corners.length === 4
          ? (corners.map((p) => [p.x, p.y]) as Corners)
          : undefined;
      const res = await analyzeImage(file, c, target ?? undefined);
      setResult(res);
      setTargetBall(res.shot?.target_ball ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const shot = result?.shot;
  const handleAt = (p: Pt) => ({
    left: padX + p.x * CANVAS_W,
    top: padY + p.y * imgH,
  });

  return (
    <div style={styles.page}>
      <h1 style={styles.h1}>Pool Vision AI</h1>
      <p style={styles.sub}>
        Upload a pool-table photo, then click its 4 corners
        (top-left → top-right → bottom-right → bottom-left). Click 3 and the 4th
        is guessed — drag any corner to fix it, even off the photo edge.
      </p>

      <input type="file" accept="image/*" onChange={onFile} />

      {preview && !result && (
        <div style={{ marginTop: 16 }}>
          <p style={styles.hint}>
            {corners.length < 4
              ? `Click ${CORNER_LABELS[corners.length]} corner (${corners.length}/4)`
              : "Drag corners to adjust, then Analyze."}
          </p>
          <div
            ref={boxRef}
            onClick={onCanvasClick}
            style={{
              position: "relative",
              width: CANVAS_W * (1 + 2 * PAD),
              height: imgH * (1 + 2 * PAD),
              background: "#eef0f3",
              borderRadius: 8,
              cursor: corners.length < 4 ? "crosshair" : "default",
              userSelect: "none",
              touchAction: "none",
            }}
          >
            <img
              src={preview}
              onLoad={(e) =>
                setAspect(
                  e.currentTarget.naturalHeight / e.currentTarget.naturalWidth,
                )
              }
              style={{
                position: "absolute",
                left: padX,
                top: padY,
                width: CANVAS_W,
                height: imgH,
                borderRadius: 4,
              }}
              alt="upload"
            />
            {corners.length === 4 && (
              <svg style={styles.svg}>
                <polygon
                  points={corners
                    .map((p) => `${handleAt(p).left},${handleAt(p).top}`)
                    .join(" ")}
                  fill="rgba(0,200,120,0.15)"
                  stroke="#00c878"
                  strokeWidth={2}
                />
              </svg>
            )}
            {corners.map((p, i) => (
              <div
                key={i}
                onPointerDown={(e) => dragHandle(i, e)}
                onClick={(e) => e.stopPropagation()}
                style={{ ...styles.handle, ...handleAt(p) }}
                title={CORNER_LABELS[i]}
              >
                {i + 1}
              </div>
            ))}
          </div>
          <div style={{ marginTop: 12, display: "flex", gap: 12 }}>
            <button
              onClick={() => analyze()}
              disabled={loading}
              style={styles.btn}
            >
              {loading ? "Analyzing…" : "Analyze"}
            </button>
            <button onClick={() => setCorners([])} style={styles.btnGhost}>
              Reset corners
            </button>
            <span style={styles.hint}>
              No corners set → server auto-detects (works best top-down).
            </span>
          </div>
        </div>
      )}

      {error && <p style={styles.err}>{error}</p>}

      {result && (
        <div style={styles.row}>
          <div>
            <h3>Suggested shot</h3>
            <img
              src={`data:image/png;base64,${result.overlay_png}`}
              style={styles.img}
              alt="overlay"
            />
            <button
              onClick={() => {
                setResult(null);
                setTargetBall(null);
              }}
              style={{ ...styles.btnGhost, marginTop: 8 }}
            >
              ← Re-pick corners
            </button>

            {bestPerBall(result).length > 0 && (
              <div style={{ marginTop: 16 }}>
                <h4 style={{ margin: "0 0 6px" }}>Pick ball to hit</h4>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {bestPerBall(result).map((s) => {
                    const active = s.target_ball === targetBall;
                    return (
                      <button
                        key={s.target_ball}
                        onClick={() => analyze(s.target_ball)}
                        disabled={loading}
                        style={active ? styles.ballBtnActive : styles.ballBtn}
                        title={`best: ${Math.round(
                          s.success_rate * 100,
                        )}% to ${s.target_pocket}`}
                      >
                        #{s.target_ball}
                      </button>
                    );
                  })}
                </div>
                <span style={styles.hint}>
                  Buttons ranked easiest → hardest. Click to re-aim.
                </span>
              </div>
            )}
          </div>

          <div style={styles.panel}>
            <Stat
              label="Table detected"
              value={result.table_detected ? "yes" : "no (fallback)"}
            />
            <Stat label="Balls found" value={String(result.balls.length)} />
            <Stat
              label="Cue ball"
              value={result.cue_ball ? `#${result.cue_ball.id}` : "—"}
            />
            <hr />
            {shot ? (
              <>
                <Stat label="Target ball" value={`#${shot.target_ball}`} />
                <Stat label="Pocket" value={shot.target_pocket} />
                <Stat label="Difficulty" value={shot.difficulty.toFixed(2)} />
                <Stat
                  label="Success rate"
                  value={`${Math.round(shot.success_rate * 100)}%`}
                />
                {shot.spin && <CueBallDiagram spin={shot.spin} />}
              </>
            ) : (
              <p>No valid shot found.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// candidates are pre-sorted easiest→hardest; keep the best shot per ball.
function bestPerBall(result: AnalyzeResult) {
  const seen = new Set<number>();
  return result.candidates.filter((s) => {
    if (seen.has(s.target_ball)) return false;
    seen.add(s.target_ball);
    return true;
  });
}

// Cue-ball strike chart: crosshair circle with a dot at the recommended spot.
function CueBallDiagram({ spin }: { spin: Spin }) {
  const R = 60; // px radius of the drawn cue ball
  const cx = R + 8;
  const cy = R + 8;
  // hit_x/hit_y are -1..1 fractions of the radius; keep the dot inside the rim.
  const k = 0.8;
  const dx = cx + spin.hit_x * R * k;
  const dy = cy + spin.hit_y * R * k;
  return (
    <div style={{ marginTop: 12 }}>
      <h4 style={{ margin: "0 0 6px" }}>Where to hit cue ball</h4>
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <svg width={(R + 8) * 2} height={(R + 8) * 2}>
          <circle cx={cx} cy={cy} r={R} fill="#fafafa" stroke="#333" strokeWidth={2} />
          <line x1={cx - R} y1={cy} x2={cx + R} y2={cy} stroke="#ccc" strokeWidth={1} />
          <line x1={cx} y1={cy - R} x2={cx} y2={cy + R} stroke="#ccc" strokeWidth={1} />
          <circle cx={cx} cy={cy} r={2} fill="#bbb" />
          <circle cx={dx} cy={dy} r={9} fill="#e23b3b" stroke="#fff" strokeWidth={2} />
        </svg>
        <div style={{ maxWidth: 180 }}>
          <div style={{ fontWeight: 700, fontSize: 16 }}>{spin.zone}</div>
          <div style={{ color: "#555", fontSize: 13, marginTop: 4 }}>{spin.tip}</div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div style={styles.stat}>
      <span style={styles.statLabel}>{label}</span>
      <span style={styles.statValue}>{value}</span>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: { fontFamily: "system-ui, sans-serif", maxWidth: 1100, margin: "0 auto", padding: 24, color: "#111" },
  h1: { marginBottom: 4 },
  sub: { color: "#555", marginTop: 0 },
  hint: { color: "#777", fontSize: 14, margin: "4px 0" },
  row: { display: "flex", gap: 24, marginTop: 16, flexWrap: "wrap", alignItems: "flex-start" },
  img: { maxWidth: 700, width: "100%", borderRadius: 8, border: "1px solid #ddd", display: "block" },
  panel: { minWidth: 220, background: "#f6f7f9", borderRadius: 8, padding: 16 },
  stat: { display: "flex", justifyContent: "space-between", padding: "4px 0" },
  statLabel: { color: "#666" },
  statValue: { fontWeight: 600 },
  err: { color: "#b00020" },
  svg: { position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" },
  handle: {
    position: "absolute", width: 24, height: 24, marginLeft: -12, marginTop: -12,
    borderRadius: "50%", background: "#00c878", color: "#fff", fontSize: 12,
    fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center",
    cursor: "grab", border: "2px solid #fff", boxShadow: "0 1px 4px rgba(0,0,0,.4)",
    touchAction: "none",
  },
  btn: {
    background: "#111", color: "#fff", border: "none", borderRadius: 6,
    padding: "8px 18px", fontWeight: 600, cursor: "pointer",
  },
  btnGhost: {
    background: "#fff", color: "#111", border: "1px solid #ccc", borderRadius: 6,
    padding: "8px 14px", cursor: "pointer",
  },
};
