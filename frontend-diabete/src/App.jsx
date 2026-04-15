import { useState } from "react";
 
const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
 
  :root {
    --bg: #070b14;
    --surface: rgba(255,255,255,0.028);
    --surface2: rgba(255,255,255,0.05);
    --border: rgba(255,255,255,0.065);
    --border-glow: rgba(0,212,255,0.28);
    --accent: #00d4ff;
    --accent-glow: rgba(0,212,255,0.18);
    --danger: #ff4560;
    --danger-glow: rgba(255,69,96,0.16);
    --success: #00e5a0;
    --success-glow: rgba(0,229,160,0.16);
    --t1: rgba(255,255,255,0.91);
    --t2: rgba(255,255,255,0.48);
    --t3: rgba(255,255,255,0.24);
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Syne', sans-serif;
    --r: 18px;
    --rs: 10px;
  }
 
  * { box-sizing: border-box; margin: 0; padding: 0; }
 
  .hp-root {
    min-height: 100vh;
    font-family: var(--sans);
    background: var(--bg);
    background-image:
      radial-gradient(ellipse 70% 50% at 15% 0%, rgba(0,90,200,0.13) 0%, transparent 60%),
      radial-gradient(ellipse 50% 40% at 85% 100%, rgba(0,212,255,0.07) 0%, transparent 55%);
    color: var(--t1);
  }
 
  .hp-grid-bg {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
      linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px);
    background-size: 44px 44px;
  }
 
  .hp-z { position: relative; z-index: 1; }
 
  /* ── HEADER ── */
  .hp-header {
    position: sticky; top: 0; z-index: 100;
    background: rgba(7,11,20,0.82);
    backdrop-filter: blur(22px);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 40px;
  }
 
  .hp-logo { display: flex; align-items: center; gap: 14px; }
 
  .hp-logo-icon {
    width: 40px; height: 40px; border-radius: 11px;
    background: linear-gradient(135deg, #00d4ff 0%, #0055cc 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 4px 20px rgba(0,212,255,0.32);
    flex-shrink: 0;
  }
 
  .hp-logo-text { font-size: 21px; font-weight: 800; letter-spacing: -0.4px; }
  .hp-logo-accent { color: var(--accent); }
  .hp-logo-sub {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.06em;
    color: var(--t3); margin-top: 2px;
  }
 
  .hp-header-right { display: flex; align-items: center; gap: 14px; }
  .hp-version { font-family: var(--mono); font-size: 10px; color: var(--t3); letter-spacing: 0.06em; }
 
  .hp-status {
    display: flex; align-items: center; gap: 7px;
    background: rgba(0,229,160,0.07);
    border: 1px solid rgba(0,229,160,0.22);
    border-radius: 100px; padding: 6px 14px;
    font-family: var(--mono); font-size: 11px; font-weight: 700;
    color: var(--success); letter-spacing: 0.06em;
  }
 
  .hp-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 8px var(--success);
    animation: pulse-dot 2.2s ease-in-out infinite;
  }
 
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--success); }
    50% { opacity: 0.38; box-shadow: 0 0 3px var(--success); }
  }
 
  /* ── HERO ── */
  .hp-hero { padding: 40px 40px 0; margin-bottom: 28px; }
  .hp-hero h1 {
    font-size: 38px; font-weight: 800; letter-spacing: -1.2px; line-height: 1.08;
  }
  .hp-hero h1 span { color: var(--accent); }
  .hp-hero p { color: var(--t2); font-size: 14px; margin-top: 7px; }
 
  /* ── LAYOUT ── */
  .hp-layout {
    display: grid;
    grid-template-columns: 1fr 390px;
    gap: 22px;
    padding: 0 40px 60px;
    align-items: start;
    max-width: 1440px;
  }
 
  .hp-sticky { position: sticky; top: 90px; }
 
  /* ── CARDS ── */
  .hp-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 26px;
    margin-bottom: 14px;
    transition: border-color 0.3s;
    animation: card-in 0.55s cubic-bezier(0.16,1,0.3,1) both;
  }
  .hp-card:hover { border-color: var(--border-glow); }
 
  .hp-card:nth-child(1) { animation-delay: 0.04s; }
  .hp-card:nth-child(2) { animation-delay: 0.09s; }
  .hp-card:nth-child(3) { animation-delay: 0.14s; }
  .hp-card:nth-child(4) { animation-delay: 0.19s; }
  .hp-card:nth-child(5) { animation-delay: 0.24s; }
 
  @keyframes card-in {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0); }
  }
 
  .hp-card-header {
    display: flex; align-items: center; gap: 11px;
    padding-bottom: 18px; margin-bottom: 22px;
    border-bottom: 1px solid var(--border);
  }
 
  .hp-card-icon {
    width: 36px; height: 36px; border-radius: 9px;
    background: var(--surface2); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 17px; flex-shrink: 0;
  }
 
  .hp-card-title { font-size: 14px; font-weight: 700; letter-spacing: -0.2px; }
  .hp-card-num {
    margin-left: auto;
    font-family: var(--mono); font-size: 10px; color: var(--t3);
    background: var(--surface2); border: 1px solid var(--border);
    padding: 3px 8px; border-radius: 5px; letter-spacing: 0.05em;
  }
 
  /* ── FIELD LABEL ── */
  .lbl {
    display: flex; justify-content: space-between;
    font-family: var(--mono); font-size: 10px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--t3);
    margin-bottom: 8px;
  }
  .lbl-val { color: var(--accent); font-size: 12px; }
  .lbl-val.danger { color: var(--danger); }
 
  /* ── RANGE ── */
  input[type="range"] {
    -webkit-appearance: none; width: 100%; height: 3px;
    border-radius: 2px; background: rgba(255,255,255,0.09);
    cursor: pointer; display: block; margin: 6px 0;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px; height: 16px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 10px rgba(0,212,255,0.55), 0 0 0 3px rgba(0,212,255,0.1);
    cursor: pointer; transition: transform 0.15s, box-shadow 0.15s;
  }
  input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.3);
    box-shadow: 0 0 18px rgba(0,212,255,0.75), 0 0 0 4px rgba(0,212,255,0.14);
  }
  input[type="range"].danger-range::-webkit-slider-thumb {
    background: var(--danger);
    box-shadow: 0 0 10px rgba(255,69,96,0.55), 0 0 0 3px rgba(255,69,96,0.1);
  }
  input[type="range"].danger-range::-webkit-slider-thumb:hover {
    box-shadow: 0 0 18px rgba(255,69,96,0.75), 0 0 0 4px rgba(255,69,96,0.14);
  }
 
  .range-ticks {
    display: flex; justify-content: space-between; margin-top: 5px;
  }
  .range-tick {
    font-family: var(--mono); font-size: 9px; color: var(--t3); letter-spacing: 0.04em;
  }
 
  /* ── SELECT / INPUT ── */
  select, input[type="number"] {
    width: 100%;
    background: rgba(255,255,255,0.032) !important;
    border: 1px solid var(--border) !important;
    color: var(--t1) !important;
    border-radius: var(--rs) !important;
    padding: 11px 13px !important;
    font-family: var(--sans) !important;
    font-size: 13px !important; font-weight: 500 !important;
    outline: none; -webkit-appearance: none; appearance: none;
    transition: border-color 0.2s, background 0.2s;
  }
  select:focus, input[type="number"]:focus {
    border-color: var(--border-glow) !important;
    background: rgba(0,212,255,0.04) !important;
  }
  select option { background: #0d1421; color: white; }
 
  /* ── GRIDS ── */
  .g2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .g3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
  .col-gap { display: flex; flex-direction: column; gap: 22px; }
 
  /* ── SUBMIT BUTTON ── */
  .hp-btn {
    width: 100%; border: none; border-radius: var(--r);
    padding: 19px 24px; cursor: pointer;
    font-family: var(--sans); font-size: 15px; font-weight: 800;
    letter-spacing: -0.3px;
    background: var(--accent); color: #000;
    box-shadow: 0 8px 30px rgba(0,212,255,0.28), 0 2px 8px rgba(0,212,255,0.14);
    display: flex; align-items: center; justify-content: center; gap: 10px;
    margin-bottom: 14px;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
  }
  .hp-btn:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 14px 42px rgba(0,212,255,0.36), 0 4px 12px rgba(0,212,255,0.2);
  }
  .hp-btn:active:not(:disabled) { transform: translateY(0); }
  .hp-btn:disabled { opacity: 0.58; cursor: not-allowed; }
  .hp-btn-arrow { font-size: 20px; font-weight: 400; }
 
  @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
  .spinning { display: inline-block; animation: spin 0.9s linear infinite; }
 
  /* ── WAITING ── */
  .hp-waiting {
    background: var(--surface); border: 1px dashed var(--border);
    border-radius: var(--r); min-height: 400px;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 14px; padding: 32px; text-align: center;
  }
  .hp-waiting-icon {
    width: 72px; height: 72px; border-radius: 50%;
    background: var(--surface2); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center; font-size: 30px;
  }
  .hp-waiting-tag {
    font-family: var(--mono); font-size: 10px; color: var(--t3);
    letter-spacing: 0.1em; border: 1px dashed var(--border);
    padding: 7px 16px; border-radius: 6px;
  }
 
  /* ── RESULT ── */
  @keyframes result-in {
    from { opacity: 0; transform: scale(0.96); }
    to   { opacity: 1; transform: scale(1); }
  }
 
  .hp-result {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r); padding: 26px;
    animation: result-in 0.4s cubic-bezier(0.16,1,0.3,1) both;
  }
  .hp-result.is-danger { border-color: rgba(255,69,96,0.32); background: rgba(255,69,96,0.04); }
  .hp-result.is-success { border-color: rgba(0,229,160,0.32); background: rgba(0,229,160,0.04); }
 
  /* ── GAUGE ── */
  .gauge-wrap { position: relative; display: flex; align-items: center; justify-content: center; }
  .gauge-ring { transition: stroke-dashoffset 1.3s cubic-bezier(0.16,1,0.3,1), stroke 0.4s; }
  .gauge-center {
    position: absolute; display: flex; flex-direction: column;
    align-items: center; justify-content: center; pointer-events: none;
  }
  .gauge-pct { font-family: var(--mono); font-size: 38px; font-weight: 700; line-height: 1; }
  .gauge-sub {
    font-family: var(--mono); font-size: 9px; color: var(--t3);
    letter-spacing: 0.12em; text-transform: uppercase; margin-top: 4px;
  }
 
  /* ── DIAG BADGE ── */
  .hp-diag {
    border-radius: 11px; padding: 14px 16px;
    display: flex; align-items: center; gap: 11px; margin-top: 14px;
  }
  .hp-diag.is-danger { background: var(--danger-glow); border: 1px solid rgba(255,69,96,0.22); }
  .hp-diag.is-success { background: var(--success-glow); border: 1px solid rgba(0,229,160,0.22); }
  .hp-diag-title { font-size: 16px; font-weight: 800; letter-spacing: -0.3px; }
  .hp-diag-sub { font-family: var(--mono); font-size: 10px; color: var(--t2); margin-top: 2px; }
 
  /* ── STATS ── */
  .hp-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
  .hp-stat {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 10px; padding: 13px;
  }
  .hp-stat-lbl {
    font-family: var(--mono); font-size: 9px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--t3); margin-bottom: 5px;
  }
  .hp-stat-val { font-family: var(--mono); font-size: 20px; font-weight: 700; }
 
  /* ── RECO ── */
  .hp-reco {
    margin-top: 10px; border-radius: 10px; padding: 14px 16px;
    font-size: 12px; line-height: 1.55; color: var(--t2);
  }
  .hp-reco-title {
    font-family: var(--mono); font-size: 9px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--t3);
    margin-bottom: 6px;
  }
 
  /* ── MODEL INFO ── */
  .hp-model {
    margin-top: 12px; padding: 16px 18px;
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  }
  .hp-model-title {
    font-family: var(--mono); font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--t3); margin-bottom: 12px;
  }
  .hp-model-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid var(--border); font-size: 11px;
  }
  .hp-model-row:last-child { border-bottom: none; }
  .hp-model-k { font-family: var(--mono); color: var(--t3); }
  .hp-model-v { font-family: var(--mono); color: var(--t2); font-weight: 600; }
 
  /* ── RESPONSIVE ── */
  @media (max-width: 1100px) {
    .hp-layout { grid-template-columns: 1fr; padding: 0 20px 60px; }
    .hp-hero { padding: 28px 20px 0; }
    .hp-header { padding: 16px 20px; }
    .hp-sticky { position: static; }
    .g3 { grid-template-columns: 1fr 1fr; }
  }
  @media (max-width: 600px) {
    .g2, .g3 { grid-template-columns: 1fr; }
    .hp-hero h1 { font-size: 28px; }
  }
`;
 
function GaugeRing({ pct, isDanger }) {
  const R = 88;
  const circ = 2 * Math.PI * R;
  const offset = circ - (Math.min(100, Math.max(0, pct)) / 100) * circ;
  const color = isDanger ? "#ff4560" : "#00e5a0";
  const glow = isDanger ? "rgba(255,69,96,0.5)" : "rgba(0,229,160,0.5)";
 
  return (
    <div className="gauge-wrap" style={{ height: 215, width: 215 }}>
      <svg width="215" height="215" viewBox="0 0 215 215">
        <defs>
          <filter id="glow-filter">
            <feGaussianBlur stdDeviation="3.5" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        {/* Outer track ring */}
        <circle cx="107.5" cy="107.5" r={R} fill="none"
          stroke="rgba(255,255,255,0.055)" strokeWidth="11" />
        {/* Subtle tick marks */}
        {[0,25,50,75].map(v => {
          const a = (v / 100) * 2 * Math.PI - Math.PI / 2;
          return (
            <line key={v}
              x1={107.5 + (R - 8) * Math.cos(a)} y1={107.5 + (R - 8) * Math.sin(a)}
              x2={107.5 + (R + 8) * Math.cos(a)} y2={107.5 + (R + 8) * Math.sin(a)}
              stroke="rgba(255,255,255,0.12)" strokeWidth="1.5"
            />
          );
        })}
        {/* Glow ring (behind) */}
        <circle cx="107.5" cy="107.5" r={R} fill="none"
          stroke={glow} strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          transform="rotate(-90 107.5 107.5)"
          opacity="0.35"
          className="gauge-ring"
        />
        {/* Main fill ring */}
        <circle cx="107.5" cy="107.5" r={R} fill="none"
          stroke={color} strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          transform="rotate(-90 107.5 107.5)"
          filter="url(#glow-filter)"
          className="gauge-ring"
        />
      </svg>
      <div className="gauge-center">
        <span className="gauge-pct" style={{ color }}>{pct.toFixed(1)}%</span>
        <span className="gauge-sub">probabilité</span>
      </div>
    </div>
  );
}
 
function BinarySelect({ label, name, value, onChange }) {
  return (
    <div>
      <div className="lbl"><span>{label}</span></div>
      <select name={name} value={value} onChange={onChange}>
        <option value={0}>Non</option>
        <option value={1}>Oui</option>
      </select>
    </div>
  );
}
 
function SliderField({ label, name, value, onChange, min, max, step = 1, ticks, danger }) {
  return (
    <div>
      <div className="lbl">
        <span>{label}</span>
        <span className={`lbl-val${danger ? " danger" : ""}`}>{value}</span>
      </div>
      <input
        type="range" name={name} min={min} max={max} step={step}
        value={value} onChange={onChange}
        className={danger ? "danger-range" : ""}
      />
      {ticks && (
        <div className="range-ticks">
          {ticks.map((t, i) => <span key={i} className="range-tick">{t}</span>)}
        </div>
      )}
    </div>
  );
}
 
export default function App() {
  const [formData, setFormData] = useState({
    Age: 9, Sex: 0, BMI: 27.0,
    HighBP: 0, HighChol: 0, CholCheck: 1, Stroke: 0, HeartDiseaseorAttack: 0,
    Smoker: 0, PhysActivity: 1, Fruits: 1, Veggies: 1, HvyAlcoholConsump: 0,
    GenHlth: 3, MentHlth: 0, PhysHlth: 0, DiffWalk: 0,
    AnyHealthcare: 1, NoDocbcCost: 0, Education: 4, Income: 6,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
 
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: parseFloat(value) }));
  };
 
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      setResult(await res.json());
    } catch (err) {
      console.error("Erreur API:", err);
    }
    setLoading(false);
  };
 
  const isDanger = result?.diagnostic?.includes("Élevé");
  const pct = result ? result.probabilite_brute * 100 : 0;
 
  const modelRows = [
    ["Algorithme", "Gradient Boosting"],
    ["Dataset", "CDC BRFSS 2015"],
    ["Features", "21 variables"],
    ["Endpoint", "/predict"],
  ];
 
  return (
    <>
      <style>{STYLES}</style>
      <div className="hp-root">
        <div className="hp-grid-bg" />
        <div className="hp-z">
 
          {/* ── HEADER ── */}
          <header className="hp-header">
            <div className="hp-logo">
              <div className="hp-logo-icon">🧬</div>
              <div>
                <div className="hp-logo-text">
                  Prédiction<span className="hp-logo-accent">Diabète</span>
                </div>
                <div className="hp-logo-sub">MLOps · Diabetes Screening · version 1.0 · Développé par le GROUPE 2</div>
              </div>
            </div>
            <div className="hp-header-right">
              <div className="hp-status">
                <div className="hp-dot" />
                API CONNECTÉE
              </div>
            </div>
          </header>
 
          {/* ── HERO ── */}
          <div className="hp-hero">
            <h1>Dossier Patient<br /><span>Diagnostic IA</span></h1>
            <p>Renseignez les données cliniques pour obtenir une prédiction du risque diabétique par notre modèle de MLP.</p>
          </div>
 
          {/* ── MAIN GRID ── */}
          <div className="hp-layout">
 
            {/* LEFT — FORM */}
            <form id="patient-form" onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column" }}>
 
              {/* 01 — Biométrie */}
              <div className="hp-card">
                <div className="hp-card-header">
                  <div className="hp-card-icon">👤</div>
                  <span className="hp-card-title">Biométrie & Démographie</span>
                  <span className="hp-card-num">01</span>
                </div>
                <div className="col-gap">
                  <SliderField label="Indice de Masse Corporelle (IMC)" name="BMI"
                    value={formData.BMI} onChange={handleChange}
                    min={15} max={50} step={0.1}
                    ticks={["Sous-poids", "Normal", "Surpoids", "Obésité"]} />
                  <div className="g2">
                    <div>
                      <div className="lbl"><span>Sexe Biologique</span></div>
                      <select name="Sex" value={formData.Sex} onChange={handleChange}>
                        <option value={0}>Femme</option>
                        <option value={1}>Homme</option>
                      </select>
                    </div>
                    <SliderField label="Âge — Catégorie" name="Age"
                      value={formData.Age} onChange={handleChange}
                      min={1} max={13} ticks={["18–24 ans", "80+ ans"]} />
                  </div>
                </div>
              </div>
 
              {/* 02 — Cardiovasculaire */}
              <div className="hp-card">
                <div className="hp-card-header">
                  <div className="hp-card-icon">🫀</div>
                  <span className="hp-card-title">Antécédents Cardiovasculaires</span>
                  <span className="hp-card-num">02</span>
                </div>
                <div className="g3">
                  <BinarySelect label="Hypertension" name="HighBP" value={formData.HighBP} onChange={handleChange} />
                  <BinarySelect label="Cholestérol Élevé" name="HighChol" value={formData.HighChol} onChange={handleChange} />
                  <BinarySelect label="Bilan Cholestérol" name="CholCheck" value={formData.CholCheck} onChange={handleChange} />
                  <BinarySelect label="Antécédent AVC" name="Stroke" value={formData.Stroke} onChange={handleChange} />
                  <BinarySelect label="Maladie Cardiaque" name="HeartDiseaseorAttack" value={formData.HeartDiseaseorAttack} onChange={handleChange} />
                </div>
              </div>
 
              {/* 03 — Mode de vie */}
              <div className="hp-card">
                <div className="hp-card-header">
                  <div className="hp-card-icon">🏃</div>
                  <span className="hp-card-title">Mode de Vie & Alimentation</span>
                  <span className="hp-card-num">03</span>
                </div>
                <div className="g3">
                  <BinarySelect label="Tabagisme (>100 cig.)" name="Smoker" value={formData.Smoker} onChange={handleChange} />
                  <BinarySelect label="Activité Physique" name="PhysActivity" value={formData.PhysActivity} onChange={handleChange} />
                  <BinarySelect label="Forte Conso. Alcool" name="HvyAlcoholConsump" value={formData.HvyAlcoholConsump} onChange={handleChange} />
                  <BinarySelect label="Consommation Fruits" name="Fruits" value={formData.Fruits} onChange={handleChange} />
                  <BinarySelect label="Consommation Légumes" name="Veggies" value={formData.Veggies} onChange={handleChange} />
                </div>
              </div>
 
              {/* 04 — Santé perçue */}
              <div className="hp-card">
                <div className="hp-card-header">
                  <div className="hp-card-icon">🧠</div>
                  <span className="hp-card-title">Santé Perçue — 30 Derniers Jours</span>
                  <span className="hp-card-num">04</span>
                </div>
                <div className="col-gap">
                  <SliderField label="Jours de mauvaise santé Mentale" name="MentHlth"
                    value={formData.MentHlth} onChange={handleChange} min={0} max={30} danger />
                  <SliderField label="Jours de mauvaise santé Physique" name="PhysHlth"
                    value={formData.PhysHlth} onChange={handleChange} min={0} max={30} danger />
                  <div className="g2">
                    <SliderField label="Santé Générale (1=Exc. / 5=Mauv.)" name="GenHlth"
                      value={formData.GenHlth} onChange={handleChange}
                      min={1} max={5} danger
                      ticks={["Excellente", "Mauvaise"]} />
                    <BinarySelect label="Difficulté à Marcher" name="DiffWalk" value={formData.DiffWalk} onChange={handleChange} />
                  </div>
                </div>
              </div>
 
              {/* 05 — Socio-éco */}
              <div className="hp-card" style={{ marginBottom: 0 }}>
                <div className="hp-card-header">
                  <div className="hp-card-icon">💼</div>
                  <span className="hp-card-title">Profil Socio-Économique</span>
                  <span className="hp-card-num">05</span>
                </div>
                <div className="col-gap">
                  <div className="g2">
                    <BinarySelect label="Couverture Médicale" name="AnyHealthcare" value={formData.AnyHealthcare} onChange={handleChange} />
                    <BinarySelect label="Problème Coût Médecin" name="NoDocbcCost" value={formData.NoDocbcCost} onChange={handleChange} />
                  </div>
                  <div className="g2">
                    <SliderField label="Niveau d'Éducation" name="Education"
                      value={formData.Education} onChange={handleChange}
                      min={1} max={6} ticks={["Aucun", "Diplômé+"]} />
                    <SliderField label="Niveau de Revenus" name="Income"
                      value={formData.Income} onChange={handleChange}
                      min={1} max={8} ticks={["< $10k", "$75k+"]} />
                  </div>
                </div>
              </div>
            </form>
 
            {/* RIGHT — RESULT */}
            <div className="hp-sticky">
              <button form="patient-form" type="submit" disabled={loading} className="hp-btn">
                {loading ? (
                  <><span className="spinning">⟳</span> Analyse en cours…</>
                ) : (
                  <><span>Lancer le Diagnostic</span><span className="hp-btn-arrow">→</span></>
                )}
              </button>
 
              {result ? (
                <div className={`hp-result${isDanger ? " is-danger" : " is-success"}`} key={result.probabilite_brute}>
                  <GaugeRing pct={pct} isDanger={isDanger} />
 
                  <div className={`hp-diag${isDanger ? " is-danger" : " is-success"}`}>
                    <span style={{ fontSize: 24 }}>{isDanger ? "⚠️" : "✅"}</span>
                    <div>
                      <div className="hp-diag-title" style={{ color: isDanger ? "var(--danger)" : "var(--success)" }}>
                        {result.diagnostic}
                      </div>
                      <div className="hp-diag-sub">
                        Seuil appliqué · {result.seuil_applique}
                      </div>
                    </div>
                  </div>
 
                  <div className="hp-stats">
                    <div className="hp-stat">
                      <div className="hp-stat-lbl">Probabilité</div>
                      <div className="hp-stat-val" style={{ color: isDanger ? "var(--danger)" : "var(--success)" }}>
                        {(result.probabilite_brute * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="hp-stat">
                      <div className="hp-stat-lbl">Seuil</div>
                      <div className="hp-stat-val" style={{ fontSize: 16 }}>{result.seuil_applique}</div>
                    </div>
                  </div>
 
                  <div className="hp-reco" style={{
                    background: isDanger ? "rgba(255,69,96,0.06)" : "rgba(0,229,160,0.06)",
                    border: `1px solid ${isDanger ? "rgba(255,69,96,0.16)" : "rgba(0,229,160,0.16)"}`,
                  }}>
                    <div className="hp-reco-title">Recommandation Clinique</div>
                    {isDanger
                      ? "Examens complémentaires requis. Orientez le patient vers un endocrinologue pour confirmation diagnostique et bilan glycémique complet."
                      : "Aucun marqueur critique détecté. Maintenir les habitudes de vie saines et assurer un suivi annuel préventif."}
                  </div>
                </div>
              ) : (
                <div className="hp-waiting">
                  <div className="hp-waiting-icon">🤖</div>
                  <div style={{ fontWeight: 700, fontSize: 15, color: "var(--t2)" }}>En attente d'analyse</div>
                  <div style={{ fontSize: 12, color: "var(--t3)", lineHeight: 1.65, maxWidth: 210 }}>
                    Complétez le dossier patient et lancez le diagnostic pour visualiser les résultats IA.
                  </div>
                  <div className="hp-waiting-tag">MODÈLE PRÊT · 21 FEATURES</div>
                </div>
              )}
 
              <div className="hp-model">
                <div className="hp-model-title">Informations Modèle</div>
                {modelRows.map(([k, v]) => (
                  <div key={k} className="hp-model-row">
                    <span className="hp-model-k">{k}</span>
                    <span className="hp-model-v">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}