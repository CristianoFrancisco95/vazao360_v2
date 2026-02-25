# pages/02_Calculadora.py
import streamlit as st
import numpy as np
import pathlib
import pandas as pd
import altair as alt
import io
from pathlib import Path

from app.ui import inject_global_css, sidebar_brand, page_header
from app.auth import require_login
from app.core.models.inputs import ProcessConditions
from app.core.equations.models import (
    AsmeMFC3M, AGA31, ANPModel, YuHu1, YuHu2, Montiel, Cengel
)
from app.core.thermo.mixture_props import compute_mixture_props
from app.core.selection.selector import evaluate_applicability

st.set_page_config(page_title="Vazão360 — Calculadora", page_icon="🧮", layout="wide")
require_login()
inject_global_css()
sidebar_brand()


st.title("03 — Calculadora")

# ============================================================
# Verificações & carregamento do cenário
# ============================================================
_comp_mode = st.session_state.get("composition_mode", "Sim")

if "process_conditions" not in st.session_state:
    st.warning("Cenário incompleto. Volte à **02 — Entrada de Dados** e salve o cenário.")
    st.stop()
if _comp_mode == "Sim" and "composition" not in st.session_state:
    st.warning("Cenário incompleto. Volte à **02 — Entrada de Dados** e salve o cenário.")
    st.stop()
if _comp_mode == "Não" and "molar_mass_estimated" not in st.session_state:
    st.warning("Cenário incompleto. Volte à **02 — Entrada de Dados** e salve o cenário.")
    st.stop()

pc = ProcessConditions(**st.session_state["process_conditions"])
composition = st.session_state.get("composition") if _comp_mode == "Sim" else None
is_transient = (pc.regime == "Transiente")

# ============================================================
# Propriedades da mistura (usa sessão se existir; senão calcula)
# ============================================================
def ensure_props_from_session():
    class _MP: pass

    if _comp_mode == "Não":
        # Usa Massa Molar Estimada e k fixo = 1,4
        mw = float(st.session_state.get("molar_mass_estimated", 16.043))
        mp = st.session_state.get("mixture_props")
        if mp and mp.get("_mode") == "estimated" and mp.get("MW_mix") == mw:
            mpp = _MP()
            mpp.MW_mix = float(mp["MW_mix"])
            mpp.Cp_mass = float(mp["Cp_mass"])
            mpp.Cv_mass = float(mp["Cv_mass"])
            mpp.k = 1.4
            return mpp
        # Calcula Cp/Cv para gás ideal com k=1,4
        R = 8314.0  # J/(kmol·K)
        cv = R / mw / (1.4 - 1)
        cp = 1.4 * cv
        mpp = _MP()
        mpp.MW_mix = mw
        mpp.Cp_mass = cp
        mpp.Cv_mass = cv
        mpp.k = 1.4
        st.session_state["mixture_props"] = {
            "_mode": "estimated",
            "MW_mix": mw,
            "Cp_mass": float(cp),
            "Cv_mass": float(cv),
            "k": 1.4,
        }
        return mpp

    # Modo Sim: usa composição molar
    mp = st.session_state.get("mixture_props")
    if mp and all(k in mp for k in ("MW_mix", "Cp_mass", "Cv_mass", "k")) and mp.get("_mode") != "estimated":
        mpp = _MP()
        mpp.MW_mix = float(mp["MW_mix"])
        mpp.Cp_mass = float(mp["Cp_mass"])
        mpp.Cv_mass = float(mp["Cv_mass"])
        mpp.k = float(mp["k"])
        return mpp
    # calcular a partir da composição e T
    from app.core.models.composition import Composition
    comp_obj = Composition(**composition)
    ROOT = Path(__file__).resolve().parents[1]
    catalog_path = str(ROOT / "data" / "species_catalog.yaml")
    props = compute_mixture_props(comp_obj, catalog_path, T_K=pc.t0_c + 273.15)
    # guardar em sessão em formato simples
    st.session_state["mixture_props"] = {
        "MW_mix": float(props.MW_mix),
        "Cp_mass": float(props.Cp_mass),
        "Cv_mass": float(props.Cv_mass),
        "k": float(props.k),
    }
    return props

props = ensure_props_from_session()

# ── Formatação numérica: 2 casas decimais; científico se |v| < 1; separador decimal "," ─────
def _fmt(v) -> str:
    try:
        f = float(v)
        s = f"{f:.2e}" if 0 < abs(f) < 1 else f"{f:.2f}"
        return s.replace(".", ",")
    except (TypeError, ValueError):
        return str(v)

def _fmt_rho(v) -> str:
    """4 casas decimais fixas, sem notação científica — exclusivo para massa específica."""
    try:
        return f"{float(v):.4f}".replace(".", ",")
    except (TypeError, ValueError):
        return str(v)

# ============================================================
# Modelos disponíveis
# ============================================================
MODEL_MAP = {
    "ASME MFC-3M": AsmeMFC3M,
    "AGA 3.1": AGA31,
    "ANP": ANPModel,
    "YuHu 1": YuHu1,
    "YuHu 2": YuHu2,
    "Montiel": Montiel,
    "Cengel": Cengel,
}
MODEL_ORDER = list(MODEL_MAP.keys())

# ============================================================
# Funções utilitárias para calcular todos os modelos
# ============================================================



col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Massa Molar Média (kg/kmol)", _fmt(props.MW_mix))
with col2:
    st.metric("k = Cp/Cv", _fmt(props.k))
with col3:
    st.metric("Coeficiente de Arrasto (Cd)", _fmt(0.61))

    

app = evaluate_applicability(props, pc)
st.info(f"Regime: **{'estrangulado (choked)' if app.choked else 'subcrítico'}** — {app.reason}")

def compute_one(model_name: str, pc_override: ProcessConditions = None):
    MClass = MODEL_MAP[model_name]
    model = MClass()
    scenario = pc if pc_override is None else pc_override
    res = model.compute(scenario, props)

    # Fallback robusto para extras
    ex = (getattr(res, "extras", None) or {})

    
    row = {
        "Modelo": model_name,
        "Aplicável": bool(getattr(res, "applicable", False)),
        "Motivo": res.reason,
        "Vazão Mássica (kg/s)": getattr(res, "mass_flow_kg_s", None),
    }
    return res, row


def compute_all_models_continuous():
    """Cálculos para regime contínuo (comportamento original)."""
    rows = []
    results_obj = {}
    for name in MODEL_ORDER:
        res, row = compute_one(name)
        rows.append(row)
        results_obj[name] = res
    df = pd.DataFrame(rows, columns=["Modelo", "Aplicável", "Motivo", "Vazão Mássica (kg/s)"])
    return df, results_obj


# alias para retrocompatibilidade com o código abaixo
compute_all_models = compute_all_models_continuous


def compute_all_models_transient():
    """
    Para cada intervalo da tabela transiente, calcula todos os modelos.
    Retorna:
      - df_detail : DataFrame com resultados por intervalo e modelo
      - df_summary: DataFrame com massa total por modelo (soma dos intervalos)
      - model_masses: dict {nome_modelo: {"total_mass": float, "applicable": bool}}
    """
    intervals = pc.transient_table or []
    # acumula massa por modelo
    model_masses = {name: {"total_mass": 0.0, "applicable": False} for name in MODEL_ORDER}
    interval_rows = []

    for idx, iv in enumerate(intervals):
        # iv é TransientInterval ou dict
        dt_min = float(iv.dt_min if hasattr(iv, "dt_min") else iv["dt_min"])
        p0 = float(iv.p0_bar if hasattr(iv, "p0_bar") else iv["p0_bar"])
        dt_s = dt_min * 60.0

        pc_iv = ProcessConditions(
            p0_bar=p0,
            pback_bar=pc.pback_bar,
            t0_c=pc.t0_c,
            orifice_d_mm=pc.orifice_d_mm,
            pipe_D_mm=pc.pipe_D_mm,
            duration_s=dt_s,
            ambiente_fechado=pc.ambiente_fechado,
            rho_cntp=pc.rho_cntp,
            mu_pa_s=pc.mu_pa_s,
            tap_type=pc.tap_type,
            L1_m=pc.L1_m,
            regime="Contínuo",  # cada intervalo é tratado como estático
        )

        row = {
            "Intervalo": idx + 1,
            "dt (min)": dt_min,
            "p0 (bar abs)": p0,
        }
        for name in MODEL_ORDER:
            res, _ = compute_one(name, pc_override=pc_iv)
            qm = getattr(res, "mass_flow_kg_s", None)
            ok = bool(getattr(res, "applicable", False))
            if qm is not None and ok:
                mass_iv = qm * dt_s
                row[f"{name} — q_m (kg/s)"] = qm
                row[f"{name} — Massa (kg)"] = mass_iv
                model_masses[name]["total_mass"] += mass_iv
                model_masses[name]["applicable"] = True
            else:
                row[f"{name} — q_m (kg/s)"] = None
                row[f"{name} — Massa (kg)"] = None
        interval_rows.append(row)

    df_detail = pd.DataFrame(interval_rows)

    summary_rows = [
        {
            "Modelo": name,
            "Aplicável": model_masses[name]["applicable"],
            "Massa Total (kg)": model_masses[name]["total_mass"] if model_masses[name]["applicable"] else None,
        }
        for name in MODEL_ORDER
    ]
    df_summary = pd.DataFrame(summary_rows, columns=["Modelo", "Aplicável", "Massa Total (kg)"])
    return df_detail, df_summary, model_masses

# ── Diâmetro estimado por imagem (IA + OpenCV) ──────────────────────────
_img_dia_02 = st.session_state.get("image_diameter_result_latest")
if _img_dia_02 and _img_dia_02.get("success"):
    import base64 as _b64_02
    with st.expander("📷 Diâmetro estimado por imagem (IA + OpenCV)", expanded=True):
        col_ov_02, col_mt_02 = st.columns([1.4, 1])
        with col_ov_02:
            if _img_dia_02.get("overlay_b64"):
                st.image(
                    _b64_02.b64decode(_img_dia_02["overlay_b64"]),
                    caption="Overlay — máscara IA + círculo equivalente",
                    use_container_width=True,
                )
        with col_mt_02:
            _d02     = _img_dia_02.get("diameter_mm")
            _unc02   = _img_dia_02.get("uncertainty_pct")
            _conf02  = (_img_dia_02.get("confidence") or 0.0) * 100
            _arc02   = "✅ Sim" if _img_dia_02.get("aruco_detected") else "❌ Não"
            st.metric("d estimado (mm)",  f"{_d02:.3f}" if _d02 else "—")
            st.metric("d confirmado (mm)", f"{_img_dia_02.get('diameter_mm_confirmed', _d02):.3f}" if _d02 else "—")
            st.metric("Incerteza",        f"±{_unc02:.1f}%" if _unc02 else "—")
            st.metric("Confiança IA",    f"{_conf02:.0f}%")
            st.metric("Método",          _img_dia_02.get("method", "—"))
            st.metric("ArUco",            _arc02)
            _mpp02 = _img_dia_02.get("mm_per_pixel")
            if _mpp02:
                st.caption(f"Escala: {_mpp02:.4f} mm/px")

# ============================================================
# SEÇÃO ORIGINAL — Resultados por modelo, CSV e auditoria
# ============================================================
st.subheader("Resultados por Modelo")

if not is_transient:
    # ──── REGIME CONTÍNUO ──── (comportamento original)
    df_models, results_obj = compute_all_models()

    _df_display = df_models.copy()
    _df_display["Vazão Mássica (kg/s)"] = _df_display["Vazão Mássica (kg/s)"].apply(
        lambda x: _fmt(x) if pd.notna(x) else "—")
    st.dataframe(_df_display, width="stretch")

    csv_buf = io.StringIO()
    df_models.to_csv(csv_buf, index=False)
    st.download_button(
        "Baixar resultados (CSV)",
        data=csv_buf.getvalue(),
        file_name="resultados_modelos.csv",
        mime="text/csv",
    )

    qm_valid = df_models.loc[
        df_models["Aplicável"] & df_models["Vazão Mássica (kg/s)"].notna(),
        "Vazão Mássica (kg/s)",
    ]
    if len(qm_valid) > 0:
        qm_median = float(np.median(qm_valid.values))
        st.session_state["qm_median"] = qm_median
        st.caption(f"Mediana de q_m (modelos aplicáveis): **{_fmt(qm_median)} kg/s**")
    else:
        st.session_state["qm_median"] = 0.0
        st.warning("Nenhum modelo aplicável retornou q_m válido.")

    # Análise dimensional
    st.markdown("### Análise dimensional (por modelo)")
    for name in MODEL_ORDER:
        res = results_obj.get(name)
        if res is None:
            continue
        extras = (getattr(res, "extras", None) or {})
        audit = extras.get("audit")
        if audit:
            with st.expander(f"Auditoria — {name}"):
                st.code(audit)

else:
    # ──── REGIME TRANSIENTE ────
    st.info(
        f"⚡ **Regime Transiente** — {len(pc.transient_table or [])} intervalo(s). "
        "Para cada intervalo a pressão é tratada como constante; a massa liberada é somada em todos os intervalos."
    )

    df_detail, df_summary, model_masses = compute_all_models_transient()

    st.markdown("#### Resultados por Intervalo")
    # Formata para exibição
    df_detail_disp = df_detail.copy()
    for col in df_detail_disp.columns:
        if col not in ("Intervalo",):
            df_detail_disp[col] = df_detail_disp[col].apply(
                lambda x: _fmt(x) if pd.notna(x) else "—"
            )
    st.dataframe(df_detail_disp, width="stretch")

    st.markdown("#### Massa Total por Modelo (soma de todos os intervalos)")
    df_sum_disp = df_summary.copy()
    df_sum_disp["Massa Total (kg)"] = df_sum_disp["Massa Total (kg)"].apply(
        lambda x: _fmt(x) if pd.notna(x) else "—"
    )
    st.dataframe(df_sum_disp, width="stretch")

    # Exporta CSV com os dois níveis
    csv_buf = io.StringIO()
    df_detail.to_csv(csv_buf, index=False)
    st.download_button(
        "Baixar resultados por intervalo (CSV)",
        data=csv_buf.getvalue(),
        file_name="resultados_transiente_intervalos.csv",
        mime="text/csv",
    )

    # Mediana da massa total (modelos aplicáveis)
    mt_valid = df_summary.loc[
        df_summary["Aplicável"] & df_summary["Massa Total (kg)"].notna(),
        "Massa Total (kg)",
    ]
    total_dur_s = float(pc.duration_s) if pc.duration_s else 1.0
    if len(mt_valid) > 0:
        mt_median = float(np.median(mt_valid.values))
        # q_m efetivo (apenas para uso interno nas seções abaixo)
        qm_median = mt_median / total_dur_s if total_dur_s > 0 else 0.0
        st.session_state["qm_median"] = qm_median
        st.caption(
            f"Mediana da Massa Total (modelos aplicáveis): **{_fmt(mt_median)} kg** "
            f"— q_m efetivo médio: **{_fmt(qm_median)} kg/s**"
        )
    else:
        mt_median = 0.0
        qm_median = 0.0
        st.session_state["qm_median"] = 0.0
        st.warning("Nenhum modelo aplicável retornou massa válida.")

    # Análise dimensional — apenas do primeiro intervalo (representação)
    st.markdown("### Análise dimensional (primeiro intervalo, por modelo)")
    intervals_raw = pc.transient_table or []
    if intervals_raw:
        iv0 = intervals_raw[0]
        dt0_s = float(iv0.dt_min if hasattr(iv0, "dt_min") else iv0["dt_min"]) * 60.0
        p0_0  = float(iv0.p0_bar if hasattr(iv0, "p0_bar") else iv0["p0_bar"])
        pc_iv0 = ProcessConditions(
            p0_bar=p0_0, pback_bar=pc.pback_bar, t0_c=pc.t0_c,
            orifice_d_mm=pc.orifice_d_mm, pipe_D_mm=pc.pipe_D_mm,
            duration_s=dt0_s, ambiente_fechado=pc.ambiente_fechado,
            rho_cntp=pc.rho_cntp, mu_pa_s=pc.mu_pa_s,
            tap_type=pc.tap_type, L1_m=pc.L1_m, regime="Contínuo",
        )
        for name in MODEL_ORDER:
            res, _ = compute_one(name, pc_override=pc_iv0)
            extras = (getattr(res, "extras", None) or {})
            audit = extras.get("audit")
            if audit:
                with st.expander(f"Auditoria — {name} (intervalo 1)"):
                    st.code(audit)



# ============================================================
# CONFIGURAÇÕES (Classificação & Relatório)
# ============================================================
st.markdown("---")
st.header("Configurações (Classificação & Relatório)")

default_q = float(st.session_state.get("qm_median", 5.0))
col1, col2 = st.columns(2)
with col1:
    q_m = st.number_input("Vazão mássica (kg/s)", value=default_q, disabled=True)
with col2:
    # UI em minutos; cálculo em segundos
    duration_min_default = (pc.duration_s / 60.0) if pc.duration_s is not None else 1.0
    if not is_transient:
        duration_min = st.number_input(
            "Duração do vazamento (min)",
            value=float(duration_min_default),
            disabled=True,
        )
    else:
        duration_min = float(duration_min_default)
        st.number_input(
            "Duração do vazamento (min) — calculada dos intervalos",
            value=duration_min,
            disabled=True,
            key="calc_dur_readonly",
            help="Soma dos intervalos transientes informados na Aba 01. Não editável.",
        )
duration_s = duration_min * 60.0

# ============================================================
# MASSA TOTAL LIBERADA
# ============================================================
st.markdown("---")

# Tooltip texto (em português)
tooltip_text = (
    "Selecionada a Mediana dos resultados dos modelos. Valor representativo, consolidado e robusto do ponto de vista estatístico e físico, sem misturar hipóteses incompatíveis"
)

st.header("Massa Total Liberada")
st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:8px;">
      <!-- ícone tooltip -->
      <div style="display:inline-block; vertical-align: middle;">
        <span title="{tooltip_text}" style="
          display:inline-block;
          width:20px;height:20px;
          border-radius:50%;
          background:#eee;
          color:#333;
          text-align:center;
          line-height:20px;
          font-weight:bold;
          box-shadow:0 0 0 2px rgba(0,0,0,0.02);
          border:1px solid rgba(0,0,0,0.08);
          cursor:help;
        ">?</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write(r"**Definição:** $m_{\text{total}} = q_m \times \text{duração (s)}$")
m_total = q_m * duration_s
st.metric("Massa total liberada (kg)", _fmt(m_total))
st.caption(f"Duração: {_fmt(duration_min)} min ({_fmt(duration_s)} s)")

# ============================================================
# VOLUME CNTP
# ============================================================
st.markdown("---")
st.header("Volume de Gás Liberado (CNTP)")
st.write("**Cálculo:** Volume (CNTP) = Massa total liberada / Massa Específica (CNTP) informada na Aba 01")

rho_cntp = pc.rho_cntp
if rho_cntp <= 0:
    st.error("Massa Específica (CNTP) deve ser > 0. Verifique a **Aba 01 — Entrada de Dados**.")
    st.stop()
V_cnp = m_total / rho_cntp
st.metric("Volume liberado (CNTP) [Nm³]", _fmt(V_cnp))
st.caption(f"Massa Específica (CNTP) usada (kg/m³): {_fmt_rho(rho_cntp)}")

# ============================================================
# CLASSIFICAÇÃO TIER (pior hora)
# ============================================================
st.markdown("---")
st.header("Classificação TIER")

st.subheader("Massa vazada na pior hora")
st.markdown("**Regras:**")
st.markdown("- **Mais de 60 minutos:**  \n  "
            r"$m_{\text{pior}} = m_{\text{total}} \cdot \dfrac{60}{\text{duração (min)}}$")
st.markdown("- **Até 60 minutos:**  \n  "
            r"$m_{\text{pior}} = m_{\text{total}}$")

if duration_min > 60.0:
    m_pior = m_total * (60.0 / duration_min)
    calc_text = (f"Como a duração foi {_fmt(duration_min)} min (> 60), "
                 f"m_pior = m_total × 60 / duração(min) = {_fmt(m_total)} × 60 / {_fmt(duration_min)}")
else:
    m_pior = m_total
    calc_text = (f"Como a duração foi {_fmt(duration_min)} min (≤ 60), "
                 f"m_pior = m_total = {_fmt(m_total)}")

st.metric("Massa na pior hora (kg)", _fmt(m_pior))
with st.expander("Detalhe do cálculo da pior hora"):
    st.code(calc_text, language="text")

from app.core.tier.classifier import classify_general, classify_enclosed
tier_geral = classify_general(m_pior)
st.subheader("Resultado — Classificação TIER (geral)")
st.success(f"**{tier_geral}** (com base na massa da pior hora)")

if pc.ambiente_fechado:
    tier_fechado = classify_enclosed(m_pior)
    st.subheader("Resultado — Classificação TIER (ambiente fechado)")
    st.info(f"**{tier_fechado}** (com base na massa da pior hora)")
else:
    st.caption("Ambiente fechado **não** marcado nas condições do cenário (página 01).")

# ============================================================
# GRÁFICOS — Altair, tooltips, “TODOS”, npts=1000 FIXO,
#            respeitando critério de aplicabilidade
# ============================================================

st.markdown("---")
st.header("Gráficos")

def compute_qm(model_name: str, pc_override=None):
    """Retorna (q_m, aplicável)."""
    MClass = MODEL_MAP[model_name]
    model = MClass()
    scenario = pc if pc_override is None else pc_override
    res = model.compute(scenario, props)
    if getattr(res, "mass_flow_kg_s", None) is not None:
        return res.mass_flow_kg_s, bool(getattr(res, "applicable", False))
    return None, False

with st.expander("Configurações dos gráficos"):
    model_options = ["TODOS"] + list(MODEL_MAP.keys())
    model_name = st.selectbox("Modelo para os gráficos", model_options, index=0)
    st.caption("Número de pontos: **1000** (fixo)")
    npts = 1000  # fixo e implícito

def pior_hora(qm: float, duration_s: float | None) -> float | None:
    if qm is None or duration_s is None or duration_s <= 0:
        return None
    return qm * 3600.0 if duration_s >= 3600.0 else qm * duration_s

def tier_overlay_chart(x0: float, x1: float, y_max: float, is_pressure_chart: bool = False):
    """
    Faixas TIER com cores padronizadas e rótulos flutuantes à direita.
    - TIER 3 (verde): 0–50 kg
    - TIER 2 (amarelo): 50–500 kg
    - TIER 1 (vermelho): >500 kg
    """
    import pandas as pd
    import altair as alt

    # --- Definição das faixas (ordem correta e fixa) ---
    tiers = pd.DataFrame([
        {"label": "TIER 3 (< 50 kg)",   "y0": 0.0,   "y1": min(50.0, y_max),  "color": "#4CAF50"},  # Verde
        {"label": "TIER 2 (50–500 kg)", "y0": 50.0,  "y1": min(500.0, y_max), "color": "#FFC107"},  # Amarelo
        {"label": "TIER 1 (> 500 kg)",  "y0": 500.0, "y1": max(500.0, y_max), "color": "#F44336"},  # Vermelho
    ])
    tiers["x0"] = x0
    tiers["x1"] = x1

    # --- Cálculo de posições ---
    tiers["y_mid"] = (tiers["y0"] + tiers["y1"]) / 2.0
    tiers["x_pad"] = x1 - 0.01 * (x1 - x0)  # recuo 1% da borda direita

    # --- Faixas coloridas ---
    rect = (
        alt.Chart(tiers)
        .mark_rect(opacity=0.2)
        .encode(
            x=alt.X("x0:Q", title=None, axis=alt.Axis(labels=False, ticks=False, domain=False)),
            x2="x1:Q",
            y=alt.Y("y0:Q", title=None, axis=alt.Axis(labels=False, ticks=False, domain=False)),
            y2="y1:Q",
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=["TIER 1 (> 500 kg)", "TIER 2 (50–500 kg)", "TIER 3 (< 50 kg)"],
                    range=["#F44336", "#FFC107", "#4CAF50"]
                ),
                legend=alt.Legend(title="Faixas TIER"),
            ),
        )
    )


    return rect


# ============================================================
# Gráfico 01
# ============================================================

st.subheader("Gráfico 1 — Massa Liberada na pior hora em função de d/D")
st.caption("Varia **diâmetro do furo (d)** de 0,1% até 100% de **D**, mantendo **P** e **T** constantes. Mostra apenas pontos onde o modelo é aplicável.")

D_mm_base = pc.pipe_D_mm
d_min_mm = max(0.001 * D_mm_base, 0.001)  # 0,1% de D
d_max_mm = D_mm_base
d_series_mm = np.linspace(d_min_mm, d_max_mm, npts)

def build_df_mpior_vs_d(model_name_local: str) -> pd.DataFrame:
    rows = []
    names = [model_name_local] if model_name_local != "TODOS" else list(MODEL_MAP.keys())
    for mname in names:
        for dmm in d_series_mm:
            if not is_transient:
                # ── Contínuo: comportamento original ──
                pc_var = ProcessConditions(
                    p0_bar=pc.p0_bar, pback_bar=pc.pback_bar, t0_c=pc.t0_c,
                    orifice_d_mm=float(dmm), pipe_D_mm=pc.pipe_D_mm,
                    duration_s=pc.duration_s, ambiente_fechado=pc.ambiente_fechado,
                    rho_cntp=pc.rho_cntp, mu_pa_s=pc.mu_pa_s,
                    tap_type=pc.tap_type, L1_m=pc.L1_m,
                )
                qm_val, ok = compute_qm(mname, pc_override=pc_var)
                if ok and qm_val is not None:
                    mp = pior_hora(qm_val, pc.duration_s)
                    if mp is not None:
                        rows.append({
                            "Modelo": mname,
                            "d_over_D": dmm / pc.pipe_D_mm,
                            "d_mm": dmm,
                            "m_pior": mp,
                        })
            else:
                # ── Transiente: soma massa em todos os intervalos ──
                total_mass = 0.0
                any_ok = False
                intervals_raw = pc.transient_table or []
                for iv in intervals_raw:
                    dt_min_iv = float(iv.dt_min if hasattr(iv, "dt_min") else iv["dt_min"])
                    p0_iv = float(iv.p0_bar if hasattr(iv, "p0_bar") else iv["p0_bar"])
                    dt_s_iv = dt_min_iv * 60.0
                    pc_iv = ProcessConditions(
                        p0_bar=p0_iv, pback_bar=pc.pback_bar, t0_c=pc.t0_c,
                        orifice_d_mm=float(dmm), pipe_D_mm=pc.pipe_D_mm,
                        duration_s=dt_s_iv, ambiente_fechado=pc.ambiente_fechado,
                        rho_cntp=pc.rho_cntp, mu_pa_s=pc.mu_pa_s,
                        tap_type=pc.tap_type, L1_m=pc.L1_m, regime="Contínuo",
                    )
                    qm_val, ok = compute_qm(mname, pc_override=pc_iv)
                    if ok and qm_val is not None:
                        total_mass += qm_val * dt_s_iv
                        any_ok = True
                if any_ok:
                    total_dur_min = float(pc.duration_s or 0) / 60.0
                    mp = total_mass * (60.0 / total_dur_min) if total_dur_min > 60 else total_mass
                    rows.append({
                        "Modelo": mname,
                        "d_over_D": dmm / pc.pipe_D_mm,
                        "d_mm": dmm,
                        "m_pior": mp,
                    })
    return pd.DataFrame(rows, columns=["Modelo", "d_over_D", "d_mm", "m_pior"])

df1 = build_df_mpior_vs_d(model_name)
if df1.empty:
    st.warning("Nenhum ponto aplicável encontrado para as condições atuais (Gráfico 1).")
else:

    # curva principal (forçando eixos visíveis)
    # curva principal
    line1 = alt.Chart(df1).mark_line().encode(
        x=alt.X(
            "d_over_D:Q",
            title="d / D (adimensional)",
            axis=alt.Axis(labels=True, ticks=True, domain=True, grid=True,
                          labelColor="black", titleColor="black")
        ),
        y=alt.Y(
            "m_pior:Q",
            title="Massa vazada na pior hora (kg)",
            axis=alt.Axis(labels=True, ticks=True, domain=True, grid=True,
                          labelColor="black", titleColor="black")
        ),
        color=(alt.Color("Modelo:N").legend(title="Modelo") if model_name=="TODOS" else alt.value("#1f77b4")),
    )
    
    hover1 = alt.Chart(df1).mark_circle(size=60, opacity=0).encode(
        x="d_over_D:Q",
        y="m_pior:Q",
        tooltip=[
            alt.Tooltip("Modelo:N", title="Modelo"),
            alt.Tooltip("d_over_D:Q", title="d/D", format=".4f"),
            alt.Tooltip("d_mm:Q", title="d (mm)", format=".2f"),
            alt.Tooltip("m_pior:Q", title="Massa na pior hora (kg)", format=".6f"),
        ]
    ).interactive()
    
    # limites p/ faixas
    # --- montagem final GRÁFICO 1 (forçando eixos e TITULOS) ---
    y_max_1 = float(df1["m_pior"].max()) if not df1.empty else 600.0
    bg1 = tier_overlay_chart(0.0, 1.0, y_max_1)  # faixas TIER (fundo)
    
    # encoding do eixo X com título explícito
    x_enc_1 = alt.X(
        "d_over_D:Q",
        title="d / D (adimensional)",
        axis=alt.Axis(
            labels=True, ticks=True, domain=True, grid=True,
            labelColor="black", titleColor="black",
            labelFontSize=11, titleFontSize=13, titleFontWeight="bold",
            labelExpr="replace(format(datum.value, '.2f'), '.', ',')"
        )
    )
    
    # encoding do eixo Y com título explícito
    y_enc_1 = alt.Y(
        "m_pior:Q",
        title="Massa Liberada na Pior Hora (kg)",
        axis=alt.Axis(
            labels=True, ticks=True, domain=True, grid=True,
            labelColor="black", titleColor="black",
            labelFontSize=11, titleFontSize=13, titleFontWeight="bold",
            labelExpr="datum.value < 1 ? replace(format(datum.value, '.2e'), '.', ',') : replace(format(datum.value, '.2f'), '.', ',')"
        )
    )
    
    line1 = (
        alt.Chart(df1)
        .mark_line(strokeWidth=2.5, opacity=1.0)
        .encode(
            x=x_enc_1,
            y=y_enc_1,
            color=alt.Color("Modelo:N") if model_name == "TODOS" else alt.value("#1f77b4"),
            detail=alt.Detail("Modelo:N")
        )
    )
    
    hover1 = (
        alt.Chart(df1)
        .mark_circle(size=80, opacity=0)
        .encode(
            x="d_over_D:Q",
            y="m_pior:Q",
            tooltip=[
                alt.Tooltip("Modelo:N", title="Modelo"),
                alt.Tooltip("d_over_D:Q", title="d/D", format=".4f"),
                alt.Tooltip("d_mm:Q", title="d (mm)", format=".2f"),
                alt.Tooltip("m_pior:Q", title="Massa na pior hora (kg)", format=".6f"),
            ],
        )
        .interactive()
    )
    
    # scaffold invisível QUE GARANTE EIXOS (deve vir primeiro) — com os mesmos titles
    axis_scaffold1 = (
        alt.Chart(df1)
        .mark_point(opacity=0)
        .encode(
            x=x_enc_1,
            y=y_enc_1,
        )
    )
    
    title1 = "Massa Liberada na Pior Hora (kg) × (d/D) — P e T constantes" + (" — TODOS" if model_name=="TODOS" else f" — {model_name}")
    
    chart1 = (axis_scaffold1 + bg1 + line1 + hover1).properties(title=title1).resolve_scale(color="independent")
    
    # força configuração de axis (garante que não existe override global que esconda titles)
    chart1 = chart1.configure_axis(
        labelColor="black",
        titleColor="black",
        labelFontSize=11,
        titleFontSize=13,
        titleFontWeight="bold"
    )
    
    st.altair_chart(chart1, width="stretch")












# ============================================================
# Gráfico 02
# ============================================================

if not is_transient:
    # ── Contínuo: gráfico original (massa vs pressão) ──
    st.subheader("Gráfico 2 — Massa na pior hora em função da Pressão de Operação")
    st.caption("Varia **Pressão de Operação** de 1,01 a 300 bar, mantendo **diâmetro do furo** e **T** constantes. Mostra apenas pontos onde o modelo é aplicável.")

    p_series_bar = np.linspace(1.01, 300.0, npts)

    def build_df_mpior_vs_p(model_name_local: str) -> pd.DataFrame:
        rows = []
        names = [model_name_local] if model_name_local != "TODOS" else list(MODEL_MAP.keys())
        for mname in names:
            for pbar in p_series_bar:
                pc_var = ProcessConditions(
                    p0_bar=float(pbar), pback_bar=pc.pback_bar, t0_c=pc.t0_c,
                    orifice_d_mm=pc.orifice_d_mm, pipe_D_mm=pc.pipe_D_mm,
                    duration_s=pc.duration_s, ambiente_fechado=pc.ambiente_fechado,
                    rho_cntp=pc.rho_cntp, mu_pa_s=pc.mu_pa_s,
                    tap_type=pc.tap_type, L1_m=pc.L1_m,
                )
                qm_val, ok = compute_qm(mname, pc_override=pc_var)
                if ok and qm_val is not None:
                    mp = pior_hora(qm_val, pc.duration_s)
                    if mp is not None:
                        rows.append({
                            "Modelo": mname,
                            "p0_bar": pbar,
                            "m_pior": mp,
                        })
        return pd.DataFrame(rows, columns=["Modelo", "p0_bar", "m_pior"])

    df2 = build_df_mpior_vs_p(model_name)
    if df2.empty:
        st.warning("Nenhum ponto aplicável encontrado para as condições atuais (Gráfico 2).")
    else:
        line2 = alt.Chart(df2).mark_line().encode(
            x=alt.X("p0_bar:Q", title="p0 (bar abs)",
                    axis=alt.Axis(labels=True, ticks=True, domain=True, grid=True,
                                  labelColor="black", titleColor="black")),
            y=alt.Y("m_pior:Q", title="Massa vazada na pior hora (kg)",
                    axis=alt.Axis(labels=True, ticks=True, domain=True, grid=True,
                                  labelColor="black", titleColor="black")),
            color=(alt.Color("Modelo:N").legend(title="Modelo") if model_name == "TODOS" else alt.value("#1f77b4")),
        )
        hover2 = alt.Chart(df2).mark_circle(size=60, opacity=0).encode(
            x="p0_bar:Q", y="m_pior:Q",
            tooltip=[
                alt.Tooltip("Modelo:N", title="Modelo"),
                alt.Tooltip("p0_bar:Q", title="p0 (bar)", format=".2f"),
                alt.Tooltip("m_pior:Q", title="Massa na pior hora (kg)", format=".6f"),
            ]
        ).interactive()
        y_max_2 = float(df2["m_pior"].max()) if not df2.empty else 600.0
        bg2 = tier_overlay_chart(1.01, 300.0, y_max_2)
        x_enc_2 = alt.X("p0_bar:Q", title="Pressão de Operação (bar)",
                         axis=alt.Axis(labels=True, ticks=True, domain=True, grid=True,
                                       labelColor="black", titleColor="black",
                                       labelFontSize=11, titleFontSize=13, titleFontWeight="bold",
                                       labelExpr="replace(format(datum.value, '.2f'), '.', ',')"))
        y_enc_2 = alt.Y("m_pior:Q", title="Massa Liberada na Pior Hora (kg)",
                         axis=alt.Axis(labels=True, ticks=True, domain=True, grid=True,
                                       labelColor="black", titleColor="black",
                                       labelFontSize=11, titleFontSize=13, titleFontWeight="bold",
                                       labelExpr="datum.value < 1 ? replace(format(datum.value, '.2e'), '.', ',') : replace(format(datum.value, '.2f'), '.', ',')"))
        line2 = (alt.Chart(df2).mark_line(strokeWidth=2.5, opacity=1.0)
                 .encode(x=x_enc_2, y=y_enc_2,
                         color=alt.Color("Modelo:N") if model_name == "TODOS" else alt.value("#1f77b4"),
                         detail=alt.Detail("Modelo:N")))
        hover2 = (alt.Chart(df2).mark_circle(size=80, opacity=0)
                  .encode(x="p0_bar:Q", y="m_pior:Q",
                          tooltip=[alt.Tooltip("Modelo:N", title="Modelo"),
                                   alt.Tooltip("p0_bar:Q", title="p0 (bar)", format=".2f"),
                                   alt.Tooltip("m_pior:Q", title="Massa na pior hora (kg)", format=".6f")])
                  .interactive())
        axis_scaffold2 = alt.Chart(df2).mark_point(opacity=0).encode(x=x_enc_2, y=y_enc_2)
        title2 = ("Massa Liberda na Pior Hora (kg) × Pressão de Operação (bar) — d e T constantes"
                  + (" — TODOS" if model_name == "TODOS" else f" — {model_name}"))
        chart2 = (axis_scaffold2 + bg2 + line2 + hover2).properties(title=title2).resolve_scale(color="independent")
        chart2 = chart2.configure_axis(
            labelColor="black", titleColor="black",
            labelFontSize=11, titleFontSize=13, titleFontWeight="bold"
        )
        st.altair_chart(chart2, width="stretch")

else:
    # ── Transiente: gráfico de massa por intervalo ──
    st.subheader("Gráfico 2 — Massa Liberada por Intervalo (Regime Transiente)")
    st.caption(
        "Cada painel representa um modelo de cálculo. "
        "O eixo X mostra os intervalos transientes em ordem cronológica e "
        "o eixo Y a massa liberada em cada intervalo."
    )
    try:
        _df_det = df_detail
    except NameError:
        _df_det, _, _ = compute_all_models_transient()

    _interval_chart_rows = []
    for _, r in _df_det.iterrows():
        iv_label = f"Iv {int(r['Intervalo'])} ({r['dt (min)']} min, {r['p0 (bar abs)']} bar)"
        for mname in MODEL_ORDER:
            col_mass = f"{mname} — Massa (kg)"
            if col_mass in r and not pd.isna(r[col_mass]):
                _interval_chart_rows.append({
                    "Intervalo": iv_label,
                    "Número do Intervalo": int(r["Intervalo"]),
                    "Modelo": mname,
                    "Massa (kg)": float(r[col_mass]),
                })
    if _interval_chart_rows:
        df_ichart = pd.DataFrame(_interval_chart_rows)
        # um painel por modelo, eixo X compartilhado na ordem cronológica
        _interval_order = (
            df_ichart.sort_values("Número do Intervalo")
            ["Intervalo"].drop_duplicates().tolist()
        )
        _base = (
            alt.Chart(df_ichart)
            .mark_bar()
            .encode(
                x=alt.X("Intervalo:N",
                        sort=_interval_order,
                        title="Intervalo",
                        axis=alt.Axis(labelAngle=-35, labelColor="black", titleColor="black")),
                y=alt.Y("Massa (kg):Q",
                        title="Massa Liberada (kg)",
                        axis=alt.Axis(labelColor="black", titleColor="black")),
                color=alt.Color("Intervalo:N",
                                sort=_interval_order,
                                legend=alt.Legend(title="Intervalo")),
                tooltip=[
                    alt.Tooltip("Intervalo:N", title="Intervalo"),
                    alt.Tooltip("Modelo:N",    title="Modelo"),
                    alt.Tooltip("Massa (kg):Q", title="Massa (kg)", format=".4f"),
                ],
            )
        )
        _n_models = len(df_ichart["Modelo"].unique())
        _facet_chart = (
            _base
            .facet(
                facet=alt.Facet("Modelo:N", title="Modelo"),
                columns=_n_models,
            )
            .properties(title="Evolução da Massa Liberada por Modelo (Regime Transiente)")
            .configure_axis(labelColor="black", titleColor="black",
                            labelFontSize=10, titleFontSize=12, titleFontWeight="bold")
            .configure_title(fontSize=14, fontWeight="bold", color="black")
        )
        st.altair_chart(_facet_chart, use_container_width=True)
    else:
        st.warning("Nenhum modelo aplicável com massa válida nos intervalos transientes.")












# ============================================================
# Persistir cache para a Aba 03 — Relatório
# ============================================================
_cache = {
    "q_m": float(q_m),
    "duration_min": float(duration_min),
    "duration_s": float(duration_s),
    "m_total": float(m_total),
    "V_cnp": float(V_cnp),
    "m_pior": float(m_pior),
    "tier_geral": tier_geral,
    "tier_fechado": (classify_enclosed(m_pior) if pc.ambiente_fechado else None),
    "regime": pc.regime,
}
# Para transiente: guarda tabela de detalhe por intervalo (serializada)
if is_transient:
    try:
        _cache["transient_detail"] = df_detail.to_dict(orient="records")
        _cache["transient_summary"] = df_summary.to_dict(orient="records")
    except NameError:
        pass
st.session_state["__calc_report_cache__"] = _cache

