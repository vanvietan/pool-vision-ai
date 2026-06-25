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
  const [targetPt, setTargetPt] = useState<Pt | null>(null); // tapped ball-to-pot
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
    setTargetPt(null);
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
    const p = toNorm(e.clientX, e.clientY);
    // first 4 clicks place corners; after that, a click taps the ball to pot
    if (corners.length >= 4) {
      setTargetPt(p);
      return;
    }
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
      // explicit ball id (post-result re-aim) wins; else send the tapped point
      const pt: [number, number] | undefined =
        target == null && targetPt ? [targetPt.x, targetPt.y] : undefined;
      const res = await analyzeImage(file, c, target ?? undefined, pt);
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
        Upload a pool-table photo, click its 4 corners (top-left → top-right →
        bottom-right → bottom-left), then tap the ball you want to pot and hit
        Analyze. Click 3 corners and the 4th is guessed — drag any corner to fix
        it, even off the photo edge.
      </p>

      <input type="file" accept="image/*" onChange={onFile} />

      {preview && !result && (
        <div style={{ marginTop: 16 }}>
          <p style={styles.hint}>
            {corners.length < 4
              ? `Click ${CORNER_LABELS[corners.length]} corner (${corners.length}/4)`
              : targetPt
                ? "Ball to pot set (red ring). Drag corners to adjust, then Analyze."
                : "Now tap the ball you want to pot, then Analyze."}
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
              cursor: corners.length < 4 || !targetPt ? "crosshair" : "default",
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
            {targetPt && corners.length === 4 && (
              <div
                onClick={(e) => e.stopPropagation()}
                style={{ ...styles.targetMark, ...handleAt(targetPt) }}
                title="ball to pot"
              />
            )}
          </div>
          <div style={{ marginTop: 12, display: "flex", gap: 12 }}>
            <button
              onClick={() => analyze()}
              disabled={loading}
              style={styles.btn}
            >
              {loading ? "Analyzing…" : "Analyze"}
            </button>
            <button
              onClick={() => {
                setCorners([]);
                setTargetPt(null);
              }}
              style={styles.btnGhost}
            >
              Reset
            </button>
            <span style={styles.hint}>
              Tap a ball to aim there, or Analyze with none to auto-pick the
              easiest shot.
            </span>
          </div>
        </div>
      )}

      {error && <p style={styles.err}>{error}</p>}

      {result && (
        <div style={styles.row}>
          <div style={{ flex: "0 0 auto", maxWidth: "100%" }}>
            <h3>Suggested shot</h3>
            <InteractiveTable
              result={result}
              targetBall={targetBall}
              disabled={loading}
              onPick={(id) => analyze(id)}
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
                  Ranked easiest → hardest. Click a button or a ball on the
                  table to re-aim.
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

// OpenCV HSV (H 0-179, S/V 0-255) -> css rgb string, for ball fill color.
function hsvToCss(hsv?: number[] | null): string {
  if (!hsv) return "#888";
  const h = (hsv[0] * 2) % 360;
  const s = hsv[1] / 255;
  const v = hsv[2] / 255;
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  const [r, g, b] =
    h < 60 ? [c, x, 0] :
    h < 120 ? [x, c, 0] :
    h < 180 ? [0, c, x] :
    h < 240 ? [0, x, c] :
    h < 300 ? [x, 0, c] : [c, 0, x];
  const to = (n: number) => Math.round((n + m) * 255);
  return `rgb(${to(r)},${to(g)},${to(b)})`;
}

const DISPLAY_W = 700;

// Warped table image with the shot trajectory drawn on top; colored balls are
// clickable to re-aim the cue ball at that ball.
function InteractiveTable({
  result,
  targetBall,
  disabled,
  onPick,
}: {
  result: AnalyzeResult;
  targetBall: number | null;
  disabled: boolean;
  onPick: (id: number) => void;
}) {
  const scale = DISPLAY_W / result.width;
  const W = DISPLAY_W;
  const H = result.height * scale;
  const P = (p: [number, number]): [number, number] => [p[0] * scale, p[1] * scale];
  const shot = result.shot;

  return (
    <div
      style={{
        position: "relative",
        width: W,
        height: H,
        maxWidth: "100%",
        border: "1px solid #ddd",
        borderRadius: 8,
        overflow: "hidden",
      }}
    >
      <img
        src={`data:image/png;base64,${result.warped_png}`}
        style={{ position: "absolute", inset: 0, width: W, height: H }}
        alt="table"
      />
      <svg style={{ position: "absolute", inset: 0 }} width={W} height={H}>
        {/* pockets */}
        {result.pockets.map((pk) => {
          const [x, y] = P([pk.x, pk.y]);
          return <circle key={pk.name} cx={x} cy={y} r={11} fill="#000" stroke="#555" />;
        })}

        {/* trajectory: cue -> ghost (aim), object -> pocket (roll) */}
        {shot && (() => {
          const cue = P(shot.cue);
          const ghost = P(shot.ghost);
          const obj = P(shot.object_center);
          const pocket = P(shot.pocket);
          const contact = P(shot.contact);
          return (
            <g>
              <line x1={cue[0]} y1={cue[1]} x2={ghost[0]} y2={ghost[1]}
                stroke="#00e676" strokeWidth={3} />
              <line x1={obj[0]} y1={obj[1]} x2={pocket[0]} y2={pocket[1]}
                stroke="#ffab00" strokeWidth={3} strokeDasharray="6 4" />
              <circle cx={ghost[0]} cy={ghost[1]} r={5} fill="none"
                stroke="#00e676" strokeWidth={2} />
              <circle cx={contact[0]} cy={contact[1]} r={4} fill="#ff1744" />
              <circle cx={pocket[0]} cy={pocket[1]} r={16} fill="none"
                stroke="#00e676" strokeWidth={3} />
            </g>
          );
        })()}

        {/* balls (colored ones clickable) */}
        {result.balls.map((b) => {
          const [x, y] = P([b.x, b.y]);
          const r = Math.max(6, b.radius * scale);
          if (b.is_cue) {
            return <circle key={b.id} cx={x} cy={y} r={r} fill="#fff" stroke="#333" strokeWidth={2} />;
          }
          const active = b.id === targetBall;
          return (
            <circle
              key={b.id}
              cx={x}
              cy={y}
              r={r}
              fill={hsvToCss(b.color_hsv)}
              stroke={active ? "#00e676" : "#fff"}
              strokeWidth={active ? 4 : 2}
              style={{ cursor: disabled ? "wait" : "pointer" }}
              onClick={() => !disabled && onPick(b.id)}
            />
          );
        })}
      </svg>
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
  panel: { flex: "1 1 240px", minWidth: 220, background: "#f6f7f9", borderRadius: 8, padding: 16 },
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
  targetMark: {
    position: "absolute", width: 26, height: 26, marginLeft: -13, marginTop: -13,
    borderRadius: "50%", border: "3px solid #e23b3b", boxSizing: "border-box",
    background: "rgba(226,59,59,0.15)", pointerEvents: "auto", cursor: "pointer",
  },
  btn: {
    background: "#111", color: "#fff", border: "none", borderRadius: 6,
    padding: "8px 18px", fontWeight: 600, cursor: "pointer",
  },
  btnGhost: {
    background: "#fff", color: "#111", border: "1px solid #ccc", borderRadius: 6,
    padding: "8px 14px", cursor: "pointer",
  },
  ballBtn: {
    background: "#fff", color: "#111", border: "1px solid #ccc", borderRadius: 6,
    padding: "6px 12px", fontWeight: 700, cursor: "pointer", minWidth: 44,
  },
  ballBtnActive: {
    background: "#00c878", color: "#fff", border: "1px solid #00a866",
    borderRadius: 6, padding: "6px 12px", fontWeight: 700, cursor: "pointer",
    minWidth: 44,
  },
};
