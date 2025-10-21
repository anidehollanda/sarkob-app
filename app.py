import streamlit as st
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

st.set_page_config(page_title="SARKOB – Evaluación de la Sarcopenia en la Obesidad", page_icon="🧬", layout="centered")

# =========================
# UMBRALES Y CONSTANTES
# =========================
@dataclass
class Thr:
    # Fuerza (dinamometría)
    handgrip_low = {"female": 16.0, "male": 27.0}  # kg

    # Función
    gait_slow_mps = 0.8
    chair5_slow_s = 17.0  # corte solicitado

    # Antropometría
    waist_elev = {"female": 88.0, "male": 102.0}   # cm
    whr_elev   = {"female": 0.85, "male": 0.90}
    whtr_elev  = 0.5
    neck_ow    = {"female": 34.0, "male": 37.0}    # sobrepeso
    neck_ob    = {"female": 36.5, "male": 39.5}    # obesidad

    # Pantorrilla – punto base (antes de ajustar por IMC)
    calf_base  = {"female": 33.0, "male": 34.0}    # cm

    # Composición
    fat_obese_pct = {"female": 35.0, "male": 25.0} # %
    vat_high_cm2  = {"female": 80.0,  "male": 160.0}
    vatsat_high   = 0.4

    # Masa muscular – SMM/peso (BIA): bandas y cortes
    smmwt_sarc_cut = {"female": 22.1, "male": 31.5}      # "muy baja masa muscular"
    smmwt_lowmass_upper = {"female": 27.6, "male": 37.0} # límite superior de "baja masa muscular"

    # DXA ALM/peso (%)
    dxa_almwt_low_pct = {"female": 19.4, "male": 25.7}   # "baja masa muscular (DXA)"

    # SARC-F
    sarcf_pos_cut = 4

T = Thr()

# Normalizador de sexo
def norm_sex(s: str) -> str:
    return {"f":"female","m":"male","female":"female","male":"male"}[s]

# =========================
# Percentiles de prensión (tabla completa)
# =========================
_PERC_TABLE: Dict[str, List[Dict[str, Any]]] = {
    "male": [
        {"age_min":20,"age_max":24,"p5":33.9,"p10":36.8,"p20":40.5,"p30":43.2,"p40":45.7,"p50":48.0,"p60":50.4,"p70":52.9,"p80":56.0,"p90":60.1,"p95":63.6},
        {"age_min":25,"age_max":29,"p5":35.5,"p10":38.8,"p20":42.1,"p30":44.8,"p40":47.1,"p50":49.3,"p60":51.5,"p70":53.9,"p80":56.7,"p90":60.7,"p95":64.0},
        {"age_min":30,"age_max":34,"p5":35.0,"p10":38.3,"p20":42.2,"p30":45.0,"p40":47.4,"p50":49.7,"p60":52.0,"p70":54.4,"p80":57.4,"p90":61.5,"p95":64.9},
        {"age_min":35,"age_max":39,"p5":33.8,"p10":37.3,"p20":41.5,"p30":44.5,"p40":47.1,"p50":49.5,"p60":51.9,"p70":54.4,"p80":57.5,"p90":61.8,"p95":65.3},
        {"age_min":40,"age_max":44,"p5":32.3,"p10":36.0,"p20":40.4,"p30":43.6,"p40":46.3,"p50":48.8,"p60":51.2,"p70":53.9,"p80":57.1,"p90":61.5,"p95":65.1},
        {"age_min":45,"age_max":49,"p5":30.6,"p10":34.4,"p20":39.0,"p30":42.3,"p40":45.1,"p50":47.6,"p60":50.2,"p70":52.9,"p80":56.2,"p90":60.7,"p95":64.4},
        {"age_min":50,"age_max":54,"p5":28.9,"p10":32.8,"p20":37.4,"p30":40.7,"p40":43.5,"p50":46.2,"p60":48.8,"p70":51.6,"p80":54.8,"p90":59.4,"p95":63.1},
        {"age_min":55,"age_max":59,"p5":27.2,"p10":31.0,"p20":35.6,"p30":38.9,"p40":41.7,"p50":44.4,"p60":47.0,"p70":49.8,"p80":53.1,"p90":57.7,"p95":61.4},
        {"age_min":60,"age_max":64,"p5":25.5,"p10":29.1,"p20":33.6,"p30":36.9,"p40":39.7,"p50":42.4,"p60":45.0,"p70":47.8,"p80":51.1,"p90":55.6,"p95":59.3},
        {"age_min":65,"age_max":69,"p5":23.7,"p10":27.2,"p20":31.5,"p30":34.7,"p40":37.5,"p50":40.1,"p60":42.8,"p70":45.6,"p80":48.8,"p90":53.2,"p95":56.8},
        {"age_min":70,"age_max":74,"p5":21.9,"p10":25.2,"p20":29.3,"p30":32.4,"p40":35.1,"p50":37.7,"p60":40.3,"p70":43.1,"p80":46.3,"p90":50.6,"p95":54.1},
        {"age_min":75,"age_max":79,"p5":20.0,"p10":23.1,"p20":27.0,"p30":29.9,"p40":32.5,"p50":35.1,"p60":37.6,"p70":40.3,"p80":43.5,"p90":47.7,"p95":51.1},
        {"age_min":80,"age_max":84,"p5":18.0,"p10":20.8,"p20":24.5,"p30":27.3,"p40":29.8,"p50":32.3,"p60":34.8,"p70":37.5,"p80":40.5,"p90":44.7,"p95":48.0},
        {"age_min":85,"age_max":89,"p5":15.9,"p10":18.5,"p20":21.9,"p30":24.6,"p40":27.0,"p50":29.4,"p60":31.8,"p70":34.4,"p80":37.4,"p90":41.5,"p95":44.6},
        {"age_min":90,"age_max":94,"p5":13.7,"p10":16.1,"p20":19.2,"p30":21.7,"p40":24.0,"p50":26.3,"p60":28.7,"p70":31.2,"p80":34.2,"p90":38.1,"p95":41.2},
        {"age_min":95,"age_max":99,"p5":11.3,"p10":13.5,"p20":16.4,"p30":18.8,"p40":20.9,"p50":23.1,"p60":25.4,"p70":27.9,"p80":30.8,"p90":34.6,"p95":37.5},
        {"age_min":100,"age_max":150,"p5":8.8,"p10":10.8,"p20":13.5,"p30":15.7,"p40":17.8,"p50":19.8,"p60":22.0,"p70":24.5,"p80":27.2,"p90":30.9,"p95":33.8},
    ],
    "female": [
        {"age_min":20,"age_max":24,"p5":19.7,"p10":21.7,"p20":24.0,"p30":25.7,"p40":27.2,"p50":28.6,"p60":30.0,"p70":31.6,"p80":33.6,"p90":36.6,"p95":39.1},
        {"age_min":25,"age_max":29,"p5":20.0,"p10":22.0,"p20":24.5,"p30":26.3,"p40":27.9,"p50":29.4,"p60":30.9,"p70":32.6,"p80":34.6,"p90":37.4,"p95":39.7},
        {"age_min":30,"age_max":34,"p5":19.6,"p10":21.8,"p20":24.4,"p30":26.4,"p40":28.1,"p50":29.7,"p60":31.3,"p70":33.1,"p80":35.2,"p90":38.0,"p95":40.4},
        {"age_min":35,"age_max":39,"p5":19.0,"p10":21.3,"p20":24.1,"p30":26.2,"p40":28.0,"p50":29.7,"p60":31.4,"p70":33.2,"p80":35.4,"p90":38.4,"p95":40.8},
        {"age_min":40,"age_max":44,"p5":18.3,"p10":20.7,"p20":23.7,"p30":25.8,"p40":27.6,"p50":29.4,"p60":31.1,"p70":33.0,"p80":35.2,"p90":38.3,"p95":40.8},
        {"age_min":45,"age_max":49,"p5":17.6,"p10":20.1,"p20":23.1,"p30":25.2,"p40":27.1,"p50":28.9,"p60":30.6,"p70":32.5,"p80":34.8,"p90":37.9,"p95":40.4},
        {"age_min":50,"age_max":54,"p5":16.9,"p10":19.4,"p20":22.4,"p30":24.5,"p40":26.4,"p50":28.2,"p60":29.9,"p70":31.8,"p80":34.0,"p90":37.1,"p95":39.7},
        {"age_min":55,"age_max":59,"p5":16.1,"p10":18.5,"p20":21.5,"p30":23.7,"p40":25.5,"p50":27.3,"p60":29.0,"p70":30.9,"p80":33.0,"p90":36.1,"p95":38.6},
        {"age_min":60,"age_max":64,"p5":15.2,"p10":17.6,"p20":20.6,"p30":22.7,"p40":24.5,"p50":26.2,"p60":27.9,"p70":29.7,"p80":31.8,"p90":34.9,"p95":37.4},
        {"age_min":65,"age_max":69,"p5":14.3,"p10":16.6,"p20":19.5,"p30":21.6,"p40":23.3,"p50":25.0,"p60":26.6,"p70":28.4,"p80":30.5,"p90":33.4,"p95":35.8},
        {"age_min":70,"age_max":74,"p5":13.2,"p10":15.5,"p20":18.3,"p30":20.3,"p40":22.0,"p50":23.6,"p60":25.2,"p70":26.9,"p80":28.9,"p90":31.8,"p95":34.1},
        {"age_min":75,"age_max":79,"p5":12.0,"p10":14.3,"p20":17.0,"p30":18.9,"p40":20.5,"p50":22.1,"p60":23.6,"p70":25.2,"p80":27.2,"p90":29.9,"p95":32.2},
        {"age_min":80,"age_max":84,"p5":10.7,"p10":12.9,"p20":15.5,"p30":17.4,"p40":18.9,"p50":20.4,"p60":21.9,"p70":23.5,"p80":25.3,"p90":28.0,"p95":30.2},
        {"age_min":85,"age_max":89,"p5":9.3,"p10":11.4,"p20":13.9,"p30":15.7,"p40":17.2,"p50":18.6,"p60":20.0,"p70":21.5,"p80":23.3,"p90":25.9,"p95":28.0},
        {"age_min":90,"age_max":94,"p5":7.8,"p10":9.8,"p20":12.2,"p30":13.9,"p40":15.3,"p50":16.7,"p60":18.0,"p70":19.5,"p80":21.2,"p90":23.6,"p95":25.7},
        {"age_min":95,"age_max":99,"p5":6.1,"p10":8.0,"p20":10.3,"p30":11.9,"p40":13.3,"p50":14.6,"p60":15.9,"p70":17.3,"p80":18.9,"p90":21.2,"p95":23.2},
        {"age_min":100,"age_max":150,"p5":4.2,"p10":6.1,"p20":8.3,"p30":9.8,"p40":11.2,"p50":12.4,"p60":13.6,"p70":14.9,"p80":16.5,"p90":18.7,"p95":20.6},
    ]
}
_PERCENT_KEYS = ["p5","p10","p20","p30","p40","p50","p60","p70","p80","p90","p95"]

def handgrip_percentile(handgrip_kg: Optional[float], sex: str, age: Optional[float]) -> Optional[float]:
    if not handgrip_kg: return None
    rows = _PERC_TABLE[sex]
    row = rows[0]
    if age is not None:
        for r in rows:
            if r["age_min"] <= age <= r["age_max"]:
                row = r; break
    pts: List[Tuple[float, float]] = [(float(k[1:]), row[k]) for k in _PERCENT_KEYS]
    pts.sort(key=lambda x: x[1])
    if handgrip_kg <= pts[0][1]:
        p1,v1=pts[0]; p2,v2=pts[1]
        if v2==v1: return p1
        return max(0.0, p1 + (handgrip_kg - v1)/(v2 - v1)*(p2 - p1))
    if handgrip_kg >= pts[-1][1]:
        p1,v1=pts[-2]; p2,v2=pts[-1]
        if v2==v1: return p2
        return min(100.0, p1 + (handgrip_kg - v1)/(v2 - v1)*(p2 - p1))
    for i in range(len(pts)-1):
        p1,v1=pts[i]; p2,v2=pts[i+1]
        if v1 <= handgrip_kg <= v2:
            if v2==v1: return p1
            return p1 + (handgrip_kg - v1)/(v2 - v1)*(p2 - p1)
    return None

# =========================
# Helpers clínicos
# =========================
def gait_speed_4m(time_s: Optional[float]) -> Optional[Dict[str, Any]]:
    if not time_s or time_s <= 0: return None
    v = 4.0 / time_s
    return {"speed_mps": v, "is_slow": v <= T.gait_slow_mps}

def label_strength(hand_kg: Optional[float], sex: str) -> Dict[str, Any]:
    if hand_kg is None: return {"label": "", "is_low": False}
    low = hand_kg < T.handgrip_low[sex]
    return {"label": ("baja" if low else "normal"), "is_low": low}

def calf_cutoff_adjusted(sex: str, bmi: Optional[float]) -> float:
    base = T.calf_base[sex]
    if bmi is None: return base
    if 25.0 <= bmi < 30.0: return base - 3.0
    if 30.0 <= bmi < 40.0: return base - 7.0
    if bmi >= 40.0:       return base - 12.0
    return base

def smm_weight_pct_label(smm_pct: Optional[float], sex: str) -> str:
    if smm_pct is None: return ""
    sarc_cut = T.smmwt_sarc_cut[sex]          # muy baja
    low_up   = T.smmwt_lowmass_upper[sex]     # límite sup. de baja
    if smm_pct < sarc_cut: return "muy baja masa muscular"
    if smm_pct < low_up:   return "baja masa muscular"
    return "normal"

def sarcf_score(a,b,c,d,e):
    score = sum(int(max(0,min(2,v))) for v in (a,b,c,d,e))
    return {"score": score, "risk": score >= T.sarcf_pos_cut}

# =========================
# UI
# =========================
st.title("🧬 SARKOB – Evaluación de la Sarcopenia en la Obesidad")
st.caption("Herramienta de evaluación clínica desarrollada para el proyecto SARKOB")

# ---- 1) Datos basales ----
st.header("1) Datos basales")
c1,c2,c3,c4 = st.columns(4)
with c1:
    sex_in = st.selectbox("Sexo", ["female","male"], index=1, format_func=lambda s: "Mujer" if s=="female" else "Hombre")
with c2:
    age_in = st.number_input("Edad (años)", min_value=0, max_value=120, value=45, step=1)
with c3:
    height_cm = st.number_input("Talla (cm)", min_value=0.0, value=170.0, step=0.5)
with c4:
    weight_kg = st.number_input("Peso (kg)", min_value=0.0, value=70.0, step=0.1)

# IMC
bmi = (weight_kg / ((height_cm/100)**2)) if (weight_kg and height_cm) else None

# ---- 2) SARC-F ----
st.header("2) Cuestionario funcional – SARC-F (0 ninguna, 1 algo, 2 mucha)")
sc1,sc2,sc3,sc4,sc5 = st.columns(5)
with sc1:
    s1 = st.number_input("1) Fuerza", 0, 2, 0, 1)
with sc2:
    s2 = st.number_input("2) Caminar", 0, 2, 0, 1)
with sc3:
    s3 = st.number_input("3) Levantarse", 0, 2, 0, 1)
with sc4:
    s4 = st.number_input("4) Escaleras", 0, 2, 0, 1)
with sc5:
    falls_n = st.number_input("5) Caídas/año", 0, 50, 0, 1)
s5 = 0 if falls_n<=0 else (1 if falls_n<=3 else 2)
sarcf = sarcf_score(s1,s2,s3,s4,s5)
sarc_pos_text = "Positivo" if sarcf["risk"] else "Negativo"
st.markdown(f"**SARC-F:** {sarcf['score']}/10 → cribado {sarc_pos_text}")

# ---- 3) Función física ----
st.header("3) Función física")
f1,f2,f3 = st.columns(3)
with f1:
    hand_kg = st.number_input("Fuerza de prensión (kg)", min_value=0.0, value=0.0, step=0.1)
with f2:
    time_4m = st.number_input("Tiempo en 4 m (s)", min_value=0.0, value=0.0, step=0.1)
with f3:
    chair5_s = st.number_input("Silla-5 (s)", min_value=0.0, value=0.0, step=0.1)

gait = gait_speed_4m(time_4m if time_4m>0 else None)
strength = label_strength(hand_kg if hand_kg>0 else None, sex_in)

# ---- 4) Composición corporal ----
st.header("4) Composición corporal")
cc1,cc2,cc3 = st.columns(3)
with cc1:
    fat_pct = st.number_input("% Grasa total (%)", min_value=0.0, value=0.0, step=0.1)
with cc2:
    vat_cm2 = st.number_input("Área grasa visceral (cm²)", min_value=0.0, value=0.0, step=1.0)
with cc3:
    vatsat = st.number_input("Cociente VAT/SAT", min_value=0.0, value=0.0, step=0.01, format="%.2f")

cc4,cc5,cc6 = st.columns(3)
with cc4:
    waist_cm = st.number_input("Cintura (cm)", min_value=0.0, value=0.0, step=0.1)
with cc5:
    hip_cm   = st.number_input("Cadera (cm)", min_value=0.0, value=0.0, step=0.1)
with cc6:
    neck_cm  = st.number_input("Cuello (cm)", min_value=0.0, value=0.0, step=0.1)

cc7,cc8,cc9 = st.columns(3)
with cc7:
    smm_kg = st.number_input("SMM (kg, BIA)", min_value=0.0, value=0.0, step=0.1)
with cc8:
    dxa_alm_kg = st.number_input("DXA ALM (kg)", min_value=0.0, value=0.0, step=0.1)
with cc9:
    calf_cm = st.number_input("Perímetro de pantorrilla (cm)", min_value=0.0, value=0.0, step=0.1)

# Cálculos composición
smm_pct = (smm_kg/weight_kg*100.0) if (smm_kg and weight_kg) else None
smm_label = smm_weight_pct_label(smm_pct, sex_in) if smm_pct is not None else ""

dxa_alm_wt_pct = (dxa_alm_kg/weight_kg*100.0) if (dxa_alm_kg and weight_kg) else None
dxa_cut = T.dxa_almwt_low_pct[sex_in]
dxa_label = ""
if dxa_alm_wt_pct is not None:
    dxa_label = "baja masa muscular (DXA)" if dxa_alm_wt_pct < dxa_cut else "normal (DXA)"

waist_label = ("elevada" if waist_cm and waist_cm > T.waist_elev[sex_in] else ("normal" if waist_cm else ""))
whr = (waist_cm/hip_cm) if (waist_cm and hip_cm) else None
whr_label = ("elevado" if (whr is not None and whr > T.whr_elev[sex_in]) else ("normal" if whr is not None else ""))
whtr = (waist_cm/height_cm) if (waist_cm and height_cm) else None
whtr_label = ("elevado" if (whtr is not None and whtr > T.whtr_elev) else ("normal" if whtr is not None else ""))

neck_label = ""
if neck_cm:
    if neck_cm > T.neck_ob[sex_in]: neck_label = "obesidad"
    elif neck_cm > T.neck_ow[sex_in]: neck_label = "sobrepeso"
    else: neck_label = "normal"

calf_cut = calf_cutoff_adjusted(sex_in, bmi)
calf_label = ""
if calf_cm:
    calf_label = "baja" if calf_cm < calf_cut else "normal"

fat_label = ""
if fat_pct:
    fat_label = "obesidad grasa" if fat_pct >= T.fat_obese_pct[sex_in] else "normal"

vat_label = ""
if vat_cm2:
    vat_label = "elevada" if vat_cm2 > T.vat_high_cm2[sex_in] else "normal"

vatsat_label = ""
if vatsat:
    vatsat_label = "exceso VAT" if vatsat > T.vatsat_high else "normal"

# Percentil de prensión
perc = handgrip_percentile(hand_kg if hand_kg>0 else None, sex_in, age_in)

# ---- 5) Resultado global (automático) ----
st.header("5) Resultado global")

# Función baja por cualquier técnica
strength_low = strength["is_low"]
gait_slow = bool(gait and gait["is_slow"])
chair_slow = (chair5_s and chair5_s > T.chair5_slow_s)
function_low = bool(strength_low or gait_slow or chair_slow)

# Composición: criterios de masa
very_low_mass_smm = (smm_pct is not None and smm_pct < T.smmwt_sarc_cut[sex_in])     # "muy baja masa muscular"
low_mass_dxa      = (dxa_alm_wt_pct is not None and dxa_alm_wt_pct < dxa_cut)        # "baja masa muscular (DXA)"

# Diagnóstico:
# - Sarcopenia = función baja y (muy baja SMM/peso o baja DXA ALM/peso)
# - Dinapenia  = función baja y composición normal
# - No sarcopenia = resto
if function_low and (very_low_mass_smm or low_mass_dxa):
    diagnostico = "sarcopenia"
elif function_low and not (very_low_mass_smm or low_mass_dxa):
    diagnostico = "dinapenia"
else:
    diagnostico = "no sarcopenia"

# Preparar tabla
def pf(val):
    return "" if val is None else (f"{val:.2f}" if isinstance(val, float) else str(val))

cut_hand = "< 16 kg → baja" if sex_in=="female" else "< 27 kg → baja"
cut_gait = "≤ 0.8 → lenta"
cut_chair = "> 17 → lenta"
smm_ref_text = ("< 22,1% muy baja; 22,1–27,6% baja; > 27,6% normal"
                if sex_in=="female"
                else "< 31,5% muy baja; 31,5–37% baja; > 37% normal")
cut_dxa = f"< {dxa_cut:.1f}% → baja (DXA)"
cut_calf = f"< {calf_cut:.1f} → baja" if calf_cm else ""

rows = []
rows.append(["Fuerza prensil (kg)", pf(hand_kg if hand_kg>0 else None), cut_hand, (strength["label"] if hand_kg>0 else "")])
rows.append(["Percentil de fuerza", ("" if perc is None else f"{perc:.1f}%"),
             "P50 ~ referencia por edad/sexo", ("muy bajo" if perc is not None and perc < 5 else ("bajo" if perc is not None and perc < 25 else ("normal" if perc is not None else "")))])
rows.append(["Velocidad de la marcha (m/s)", ("" if not gait else f"{gait['speed_mps']:.2f}"), cut_gait if gait else "", ("lenta" if gait_slow else ("normal" if gait else ""))])
rows.append(["Silla-5 (s)", ("" if not chair5_s else f"{chair5_s:.1f}"), cut_chair if chair5_s else "", ("lenta" if chair_slow else ("normal" if chair5_s else ""))])

rows.append(["SMM/peso (%)", ("" if smm_pct is None else f"{smm_pct:.1f}%"), (smm_ref_text if smm_pct is not None else ""), smm_label])
rows.append(["DXA ALM/peso (%)", ("" if dxa_alm_wt_pct is None else f"{dxa_alm_wt_pct:.1f}%"), (cut_dxa if dxa_alm_wt_pct is not None else ""), dxa_label])

rows.append(["Pantorrilla (cm)", ("" if not calf_cm else f"{calf_cm:.1f}"), cut_calf, calf_label])
rows.append(["Cintura (cm)", ("" if not waist_cm else f"{waist_cm:.1f}"), (f"> {T.waist_elev[sex_in]:.0f} → elevada" if waist_cm else ""), waist_label])
rows.append(["Cintura/Cadera (WHR)", ("" if whr is None else f"{whr:.2f}"), (f"> {T.whr_elev[sex_in]:.2f} → elevado" if whr is not None else ""), whr_label])
rows.append(["Cintura/Altura (WHtR)", ("" if whtr is None else f"{whtr:.2f}"), (f"> {T.whtr_elev:.2f} → elevado" if whtr is not None else ""), whtr_label])
rows.append(["Cuello (cm)", ("" if not neck_cm else f"{neck_cm:.1f}"), "H: >37 SP, >39,5 OB · M: >34 SP, >36,5 OB", neck_label])
rows.append(["% Grasa total", ("" if not fat_pct else f"{fat_pct:.1f}%"), (f"≥ {T.fat_obese_pct[sex_in]:.0f}% → obesidad" if fat_pct else ""), fat_label])
rows.append(["Grasa visceral (cm²)", ("" if not vat_cm2 else f"{vat_cm2:.0f}"), (f"> {T.vat_high_cm2[sex_in]:.0f} → elevada" if vat_cm2 else ""), vat_label])
rows.append(["VAT/SAT", ("" if not vatsat else f"{vatsat:.2f}"), (f"> {T.vatsat_high:.1f} → exceso VAT" if vatsat else ""), vatsat_label])

# Fila de diagnóstico global (solicitada)
rows.append(["Diagnóstico global", diagnostico, "Definición SARKOB", 
             "Función baja + (SMM/peso muy baja o DXA ALM/peso baja)" if diagnostico=="sarcopenia"
             else ("Función baja + composición normal" if diagnostico=="dinapenia" else "—")])

df = pd.DataFrame(rows, columns=["Parámetro","Valor","Punto de corte / normalidad","Interpretación"])
st.table(df)

# Informe para copiar/descargar
informe = []
informe.append("SARKOB – Evaluación de la Sarcopenia en la Obesidad")
informe.append("")
informe.append("Datos basales:")
informe.append(f"- Sexo: {'Mujer' if sex_in=='female' else 'Hombre'}")
informe.append(f"- Edad: {age_in} años | Talla: {pf(height_cm)} cm | Peso: {pf(weight_kg)} kg | IMC: {pf(bmi)}")
informe.append("")
informe.append("Resultados clave:")
if hand_kg: informe.append(f"- Fuerza prensil: {hand_kg:.1f} kg ({strength['label']})")
if perc is not None: informe.append(f"- Percentil de fuerza: {perc:.1f}%")
if gait: informe.append(f"- Velocidad 4 m: {gait['speed_mps']:.2f} m/s ({'lenta' if gait_slow else 'normal'})")
if chair5_s: informe.append(f"- Silla-5: {chair5_s:.1f} s ({'lenta' if chair_slow else 'normal'})")
if smm_pct is not None: informe.append(f"- SMM/peso: {smm_pct:.1f}% ({smm_label})")
if dxa_alm_wt_pct is not None: informe.append(f"- DXA ALM/peso: {dxa_alm_wt_pct:.1f}% ({dxa_label})")
if calf_cm: informe.append(f"- Pantorrilla: {calf_cm:.1f} cm (corte usado: {calf_cut:.1f} cm → {calf_label})")
if fat_pct: informe.append(f"- % Grasa: {fat_pct:.1f}% ({fat_label})")
if vat_cm2: informe.append(f"- VAT: {vat_cm2:.0f} cm² ({vat_label}) | VAT/SAT: {pf(vatsat)} ({vatsat_label})")
if waist_cm: informe.append(f"- Cintura: {waist_cm:.1f} cm ({waist_label}) | WHR: {pf(whr)} ({whr_label}) | WHtR: {pf(whtr)} ({whtr_label})")
informe.append("")
informe.append(f"Diagnóstico global: {diagnostico}")
informe_txt = "\n".join(informe)

st.divider()
st.subheader("Copiar / descargar informe")
st.text_area("Informe clínico (selecciona y copia):", informe_txt, height=240)
st.download_button("Descargar informe (.txt)", data=informe_txt, file_name="sarkob_informe.txt", mime="text/plain")

