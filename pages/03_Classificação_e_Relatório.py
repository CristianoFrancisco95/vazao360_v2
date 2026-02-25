import datetime, json
from pathlib import Path
import streamlit as st
import pandas as pd
import base64

from app.core.models.inputs import ProcessConditions
from app.ui import inject_global_css, sidebar_brand, page_header
from app.auth import require_login

st.set_page_config(page_title="Vazão360 — Classificação e Relatório", page_icon="📊", layout="wide")
require_login()
inject_global_css()
sidebar_brand()

st.title("04 — Relatório (Resultados Compilados)")

_comp_mode_03 = st.session_state.get("composition_mode", "Sim")

if "process_conditions" not in st.session_state:
    st.warning("Cenário incompleto. Volte à página **01 — Entrada de Dados** e salve o cenário.")
    st.stop()
if _comp_mode_03 == "Sim" and "composition" not in st.session_state:
    st.warning("Cenário incompleto. Volte à página **01 — Entrada de Dados** e salve o cenário.")
    st.stop()

pc = ProcessConditions(**st.session_state["process_conditions"])
composition = st.session_state.get("composition")
mix = st.session_state.get("mixture_props", {})  # deve conter MW, k, etc., se calculado na aba 02
calc = st.session_state.get("__calc_report_cache__", None)

# Regime de operação
_regime = calc.get("regime", pc.regime) if calc else pc.regime
is_transient_report = (_regime == "Transiente")

if calc is None:
    st.info("Para preencher o relatório, primeiro calcule na **Aba 02 — Calculadora**.")
    st.stop()

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

# -----------------------------
# Dados do Fluido
# -----------------------------
st.header("Dados do fluido")

# Composição (tabela simples)
st.subheader("Composição do gás")
comp_map = (composition or {}).get("fractions", {})
if comp_map:
    df_comp = pd.DataFrame({
        "Componente": list(comp_map.keys()),
        "Fração molar": list(comp_map.values())
    })
    st.dataframe(df_comp, width="stretch")
else:
    _mw_est_03 = st.session_state.get("molar_mass_estimated", None)
    if _mw_est_03 is not None:
        st.info(f"Composição não disponível. Massa Molar Estimada: **{_mw_est_03:.3f} kg/kmol** | k=Cp/Cv fixado em 1,4.")
    else:
        st.info("Composição não disponível.")

# Massa Específica CNTP e Massa Molar
rho_cntp = pc.rho_cntp
mw = mix.get("MW_mix", None)

# cálculo da Massa Específica (Operação)
p0_pa = float(pc.p0_bar) * 1e5  # converte bar -> Pa
T_K = float(pc.t0_c) + 273.15
rho_oper = (float(p0_pa) * float(mw)) / (8.31446261815324 * float(T_K)) / 1000.0

colf1, colf2 = st.columns(2)
with colf1:
    st.metric("Massa Específica (CNTP) [kg/m³]", _fmt_rho(rho_cntp))
    st.metric("Massa Específica (Operação) [kg/m³]", _fmt_rho(rho_oper))
with colf2:
    if mw is not None:
        st.metric("Massa Molar Média [kg/kmol]", _fmt(float(mw)))
    else:
        st.info("Massa molar indisponível — calcule as propriedades na Aba 02.")

# -----------------------------
# Dados de Operação
# -----------------------------
st.header("Dados de operação")

# Regime (sempre visível)
st.metric("Regime", _regime)

colop1, colop2 = st.columns(2)
with colop1:
    if not is_transient_report:
        st.metric("Pressão de Operação (bar abs)", _fmt(pc.p0_bar))
    else:
        # Transiente: p0 é média ponderada — apenas informativo
        st.metric("Pressão média ponderada p0 (bar abs)", _fmt(pc.p0_bar),
                  help="Média ponderada pela duração de cada intervalo.")
with colop2:
    st.metric("Temperatura (°C)", _fmt(pc.t0_c))

# Tabela de intervalos transientes
if is_transient_report:
    st.subheader("⚡ Intervalos Transientes (Pressão × Tempo)")
    _tt = pc.transient_table or []
    if _tt:
        _tt_rows = [
            {
                "Intervalo": i + 1,
                "Duração (min)": float(iv.dt_min if hasattr(iv, "dt_min") else iv["dt_min"]),
                "p0 (bar abs)": float(iv.p0_bar if hasattr(iv, "p0_bar") else iv["p0_bar"]),
            }
            for i, iv in enumerate(_tt)
        ]
        df_tt = pd.DataFrame(_tt_rows)
        df_tt_disp = df_tt.copy()
        for col in ["Duração (min)", "p0 (bar abs)"]:
            df_tt_disp[col] = df_tt_disp[col].apply(_fmt)
        st.dataframe(df_tt_disp, width="stretch")
        st.caption(
            f"Duração total: **{_fmt(sum(r['Duração (min)'] for _, r in df_tt.iterrows()))} min** | "
            f"{len(_tt)} intervalo(s)"
        )

    # Resultados por intervalo (se disponível no cache)
    _td = calc.get("transient_detail") if calc else None
    if _td:
        st.markdown("**Resultados detalhados por intervalo (Calculadora)**")
        st.dataframe(pd.DataFrame(_td), width="stretch")

    _ts = calc.get("transient_summary") if calc else None
    if _ts:
        st.markdown("**Massa Total por Modelo**")
        df_ts = pd.DataFrame(_ts)
        df_ts_disp = df_ts.copy()
        if "Massa Total (kg)" in df_ts_disp.columns:
            df_ts_disp["Massa Total (kg)"] = df_ts_disp["Massa Total (kg)"].apply(
                lambda x: _fmt(x) if pd.notna(x) else "—"
            )
        st.dataframe(df_ts_disp, width="stretch")

# -----------------------------
# Dados do Evento
# -----------------------------
st.header("Dados do evento")
colde1, colde2 = st.columns(2)
with colde1:
    st.metric("Diâmetro do Furo (mm)", _fmt(pc.orifice_d_mm))
with colde2:
    st.metric("Diâmetro do Tubo (mm)", _fmt(pc.pipe_D_mm))

# Exibe metadados da estimativa por imagem (se disponível)
_img_meta = st.session_state.get("orifice_image_meta", None)
if _img_meta is not None:
    with st.expander("📷 Medição do diâmetro por imagem (metadados)", expanded=True):
        _col_a, _col_b, _col_c = st.columns(3)
        with _col_a:
            st.metric("Diâmetro estimado (mm)", f"{_img_meta.get('diameter_mm', '—'):.2f}" if _img_meta.get('diameter_mm') else "—")
            st.metric("Diâmetro confirmado (mm)", f"{_img_meta.get('diameter_mm_confirmed', '—'):.2f}" if _img_meta.get('diameter_mm_confirmed') else "—")
        with _col_b:
            st.metric("Incerteza (%)", f"±{_img_meta.get('uncertainty_pct', '—'):.0f}%" if _img_meta.get('uncertainty_pct') is not None else "—")
            st.metric("Confiança", f"{_img_meta.get('confidence', 0)*100:.0f}%")
        with _col_c:
            st.metric("Método de detecção", _img_meta.get('method', '—'))
            st.metric("ArUco detectado?", "Sim" if _img_meta.get('aruco_detected') else "Não")
        st.caption(
            f"Escala: {_img_meta.get('mm_per_pixel', 'N/A'):.4f} mm/px  |  "
            f"Marcador real: {_img_meta.get('marker_real_mm', '—')} mm  |  "
            f"IDs detectados: {_img_meta.get('aruco_ids', [])}  |  "
            f"Tempo: {_img_meta.get('processing_time_s', 0):.2f} s"
        )

# Exibe overlay IA (se disponível)
_img_dia_03 = st.session_state.get("image_diameter_result_latest")
if _img_dia_03 and _img_dia_03.get("success") and _img_dia_03.get("overlay_b64"):
    import base64 as _b64_03
    with st.expander("🤖 Overlay IA + OpenCV (diâmetro estimado por imagem)", expanded=True):
        col_ov03, col_mt03 = st.columns([1.4, 1])
        with col_ov03:
            st.image(
                _b64_03.b64decode(_img_dia_03["overlay_b64"]),
                caption="Overlay — máscara IA + círculo equivalente",
                use_container_width=True,
            )
        with col_mt03:
            _d03   = _img_dia_03.get("diameter_mm")
            _unc03 = _img_dia_03.get("uncertainty_pct")
            st.metric("d estimado (mm)",  f"{_d03:.3f}" if _d03 else "—")
            st.metric("Incerteza",        f"±{_unc03:.1f}%" if _unc03 else "—")
            st.metric("Método",          _img_dia_03.get("method", "—"))
            st.metric("ArUco",            "✅ Sim" if _img_dia_03.get("aruco_detected") else "❌ Não")

colde3, colde4 = st.columns(2)
with colde3:
    st.metric("Massa Total Liberada (kg)", _fmt(calc['m_total']))
with colde4:
    st.metric("Volume Liberado (CNTP) [m³]", _fmt(calc['V_cnp']))

_pf_display = st.session_state.get("pessoas_feridas", "não").capitalize()
st.metric("Pessoas Feridas?", _pf_display)

# -----------------------------
# Classificação do Evento (ATUALIZADO)
# -----------------------------
st.header("Classificação do evento")

# Recupera valores com tolerância a nomes diferentes em 'calc'
m_pior = calc.get("m_pior", calc.get("q_pior", None))   # aqui 'm_pior' é massa (kg)
q_total = calc.get("q_total", calc.get("m_total", None))  # massa total liberada (kg)
# Se existir uma taxa explícita em kg/s (ex.: 'qm_rate' ou 'q_rate'), considere-a também
q_rate_explicit = calc.get("q_rate", calc.get("qm_rate", None))

# Calcula taxa q_pior (kg/s) para verificação ANP:
# Se houver taxa explícita use ela; caso contrário, derive como m_pior / 3600 (kg/s).
q_pior_rate = None
if q_rate_explicit is not None:
    try:
        q_pior_rate = float(q_rate_explicit)
    except Exception:
        q_pior_rate = None
elif m_pior is not None:
    try:
        q_pior_rate = float(m_pior) / 3600.0  # aproximação: taxa média na pior hora
    except Exception:
        q_pior_rate = None

# Normaliza q_total
if q_total is not None:
    try:
        q_total_val = float(q_total)
    except Exception:
        q_total_val = None
else:
    q_total_val = None

# Critério ANP: comunicar se (a) taxa q_pior_rate > 0.1 kg/s  OR  (b) massa total > 1 kg
#               OU (c) houve pessoas feridas
pessoas_feridas = st.session_state.get("pessoas_feridas", "não").strip().lower()
houve_feridos = pessoas_feridas == "sim"

comunicar_anp = False
try:
    if houve_feridos:
        comunicar_anp = True
    elif (q_pior_rate is not None and q_pior_rate > 0.1) or (q_total_val is not None and q_total_val > 1.0):
        comunicar_anp = True
except Exception:
    comunicar_anp = False

# Tooltip texto (em português)
tooltip_text = (
    "Não devem ser comunicados eventos de vazamento de gás inflamável com taxa de "
    "liberação inferior a 0,1 kg/s ou com uma massa total liberada inferior a 1 kg. "
    "Eventos com pessoas feridas sempre deverão ser comunicados - Manual de Incidentes ANP."
)

# Layout: 3 colunas (primeira coluna para m_pior, segunda para tier geral, terceira para Comunicar ANP?)
col1, col2, col3 = st.columns([1.2, 1.0, 1.0])
with col1:
    st.metric("Massa Vazada na Pior Hora (kg)", _fmt(m_pior if m_pior is not None else 0))
with col2:
    st.metric("Classificação TIER (geral)", f"{calc.get('tier_geral', '—')}")

# Exibe também TIER para ambiente fechado se aplicável (mantido abaixo dos metrics)
if pc.ambiente_fechado and calc.get("tier_fechado", None):
    st.metric("Classificação TIER (ambiente fechado)", f"{calc.get('tier_fechado')}")

# Coluna 3: Comunicar ANP? com ícone tooltip
with col3:
    # Valor textual
    resp_text = "🚨 Sim" if comunicar_anp else "☘️ Não"
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="font-size:14px; font-weight:600;">Comunicar ANP?</div>
          <div style="font-size:18px; font-weight:700; color:{ '#d32f2f' if comunicar_anp else '#2e7d32'}">{resp_text}</div>
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

# Nota explicativa pequena
#st.caption("Critério ANP: comunicar se taxa > 0,1 kg/s ou massa total > 1 kg (Manual ANP).")



# -----------------------------
# Tabela compilada (única)
# -----------------------------
st.markdown("---")
st.subheader("Resultados Compilados")

rows = [
    # Dados do fluido
    {"Seção": "Dados do fluido", "Parâmetro": "Massa Específica (CNTP) [kg/m³]", "Valor": _fmt_rho(rho_cntp)},
    {"Seção": "Dados do fluido", "Parâmetro": "Massa Molar Média [kg/kmol]", "Valor": (_fmt(float(mw)) if mw is not None else "—")},
    # Dados de operação
    {"Seção": "Dados de operação", "Parâmetro": "Regime", "Valor": _regime},
    {"Seção": "Dados de operação", "Parâmetro": "Pressão de Operação (bar abs)" + (" — média ponderada" if is_transient_report else ""), "Valor": _fmt(pc.p0_bar)},
    {"Seção": "Dados de operação", "Parâmetro": "Temperatura (°C)", "Valor": _fmt(pc.t0_c)},
    # Dados do evento
    {"Seção": "Dados do evento", "Parâmetro": "Diâmetro do Furo (mm)", "Valor": _fmt(pc.orifice_d_mm)},
    {"Seção": "Dados do evento", "Parâmetro": "Diâmetro do Tubo (mm)", "Valor": _fmt(pc.pipe_D_mm)},
    {"Seção": "Dados do evento", "Parâmetro": "Massa Total Liberada (kg)", "Valor": _fmt(calc['m_total'])},
    {"Seção": "Dados do evento", "Parâmetro": "Volume Liberado (CNTP) [m³]", "Valor": _fmt(calc['V_cnp'])},
    {"Seção": "Dados do evento", "Parâmetro": "Pessoas Feridas?", "Valor": st.session_state.get("pessoas_feridas", "não").capitalize()},
    # Classificação
    {"Seção": "Classificação do evento", "Parâmetro": "Massa Vazada na Pior Hora (kg)", "Valor": _fmt(calc['m_pior'])},
    {"Seção": "Classificação do evento", "Parâmetro": "TIER (Geral)", "Valor": f"{calc['tier_geral']}"},
]
if pc.ambiente_fechado and calc.get("tier_fechado", None):
    rows.append({"Seção": "Classificação do evento", "Parâmetro": "TIER (Ambiente Fechado)", "Valor": f"{calc['tier_fechado']}"})

df_report = pd.DataFrame(rows, columns=["Seção","Parâmetro","Valor"])
st.dataframe(df_report, width="stretch")

# Informação: composição já listada acima
st.caption("A composição detalhada foi listada no início desta aba.")





# =============================
# Exportar — HTML (imprimível) com Dados do Evento
# (somente download; não grava em disco)
# =============================
import datetime
import pandas as pd
from html import escape
from app.core.models.inputs import ProcessConditions

# >>> NOVOS IMPORTS para o logotipo <<<
import base64
from pathlib import Path
# <<< FIM NOVOS IMPORTS >>>

st.markdown("---")
st.subheader("Imprimir Relatório")

calc = st.session_state.get("__calc_report_cache__")
if calc is None:
    st.info("Calcule primeiro na **Aba 03 — Calculadora** para habilitar a exportação.")
else:
    # Dados do evento (vindos da Home)
    evt_desc_raw = st.session_state.get("evt_desc", "").strip()
    evt_desc_html = escape(evt_desc_raw).replace("\n", "<br>") if evt_desc_raw else "—"
    evt_date_str = st.session_state.get("evt_date_str", "—")
    evt_analyst = st.session_state.get("evt_analyst", "—")
    evt_pessoas_feridas = st.session_state.get("pessoas_feridas", "não").capitalize()

    def build_printable_html(pc, composition, mix, calc) -> str:
        # Metadados de medição por imagem
        _img_meta_html = st.session_state.get("orifice_image_meta", None)
        # >>> BLOCO DO LOGO (lê arquivo e converte para base64) <<<
        # Caminho esperado: app/logo_vazao360.png
        # (Este arquivo deve existir no repositório.)
        #LOGO VAZAO360
        logo_path = Path(__file__).resolve().parents[1] / "app" / "logo_vazao360.png"
        logo_b64 = ""
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode("utf-8")

        #LOGO PETROBRAS
        logo_petrob_path = Path(__file__).resolve().parents[1] / "app" / "logo_petrob.png"
        logo_petrob_b64 = ""
        if logo_petrob_path.exists():
            with open(logo_petrob_path, "rb") as f:
                logo_petrob_b64 = base64.b64encode(f.read()).decode("utf-8")

        # RODAPÉ (interno.png)
        footer_img_path = Path(__file__).resolve().parents[1] / "app" / "interno.png"
        footer_b64 = ""
        if footer_img_path.exists():
            with open(footer_img_path, "rb") as f:
                footer_b64 = base64.b64encode(f.read()).decode("utf-8")



        # <<< FIM BLOCO DO LOGO >>>

        # Tabela de composição (HTML simples)
        comp_map = (composition or {}).get("fractions", {})
        if comp_map:
            def _fmt_molar(v) -> str:
                try:
                    return f"{float(v):.5f}".replace(".", ",")
                except (TypeError, ValueError):
                    return str(v)
            df_comp = pd.DataFrame({
                "Componente": list(comp_map.keys()),
                "Fração molar": [float(v) for v in comp_map.values()],
            })
            df_comp["Fração molar"] = df_comp["Fração molar"].apply(_fmt_molar)
            comp_html = df_comp.to_html(index=False)
        else:
            comp_html = "<p><em>Sem informação disponível</em></p>"

        # Tabela de intervalos transientes (HTML) — vazia se regime Contínuo
        _regime_html = calc.get("regime", pc.regime) if calc else pc.regime
        _is_trans_html = (_regime_html == "Transiente")
        transient_table_html = ""
        if _is_trans_html:
            _tt_html = pc.transient_table or []
            if _tt_html:
                _rows_html = "".join(
                    f"<tr><td>{i+1}</td><td>{_fmt(float(iv.dt_min if hasattr(iv,'dt_min') else iv['dt_min']))}</td>"
                    f"<td>{_fmt(float(iv.p0_bar if hasattr(iv,'p0_bar') else iv['p0_bar']))}</td></tr>"
                    for i, iv in enumerate(_tt_html)
                )
                transient_table_html = f"""
          <div class="section">
            <h3>&#9889; Intervalos Transientes (Pressão &times; Tempo)</h3>
            <table>
              <thead><tr><th>Intervalo</th><th>Duração (min)</th><th>p0 (bar abs)</th></tr></thead>
              <tbody>{_rows_html}</tbody>
            </table>
          </div>"""

        # CSS de impressão
        css = """
        <style>
          @page { size: A4; margin: 18mm 16mm; }
          @media print {
            body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
            .no-print { display: none !important; }
          }
          body { font-family: Arial, Helvetica, sans-serif; color: #111; }
          h1,h2,h3 { margin: 0 0 8px 0; }
          h1 { font-size: 20pt; }
          h2 { font-size: 14pt; margin-top: 16px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
          h3 { font-size: 12pt; margin-top: 10px; }
          .meta { font-size: 10pt; color: #555; margin-bottom: 12px; }
          ul { margin-top: 4px; }
          .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 24px; }
          .kv { display: flex; justify-content: space-between; border-bottom: 1px dotted #ddd; padding: 4px 0; }
          .kv b { color: #333; }
          table { border-collapse: collapse; width: 100%; font-size: 10pt; }
          th, td { border: 1px solid #ddd; padding: 6px 8px; }
          th { background: #f7f7f7; text-align: left; }
          .section { margin-top: 14px; }
          .footer { font-size: 9pt; color: #666; margin-top: 14px; }
          .box {
            background: #f7faff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 10px 12px; margin: 6px 0 10px 0;
          }
          /* >>> Estilo do cabeçalho com logotipo <<< */
          .brand { text-align: center; margin-bottom: 8px; }
          .brand img { width: 180px; max-width: 45%; }
          /* <<< Fim estilo do cabeçalho >>> */
          @page {
              @bottom-right {
                  content: "";
              }
          }
        
          .footer-img {
              position: fixed;
              bottom: 0px;
              right: 0px;
              height: 45px;
          }

        </style>
        """

        mw = mix.get("MW_mix", None)
        ambiente = "Sim" if pc.ambiente_fechado else "Não"

        # Seção de medição por imagem (aparece no HTML somente se existir)
        _img_dia_html = st.session_state.get("image_diameter_result_latest", None)
        _use_ia_html  = bool(_img_dia_html and _img_dia_html.get("success"))
        _src_html     = _img_dia_html if _use_ia_html else _img_meta_html
        if _src_html:
            _aruco_ok = "Sim" if _src_html.get("aruco_detected") else "Não"
            _unc_str  = f"±{_src_html.get('uncertainty_pct', 0):.0f}%" if _src_html.get('uncertainty_pct') is not None else "—"
            _conf_raw = _src_html.get('confidence') or (_src_html.get('ai_confidence') or 0)
            _conf_str = f"{float(_conf_raw)*100:.0f}%" if _conf_raw else "—"
            _scale_str = f"{_src_html.get('mm_per_pixel', 0):.4f} mm/px" if _src_html.get('mm_per_pixel') else "—"
            # Overlay base64 (somente pipeline IA)
            _ov_b64_html = ""
            if _use_ia_html and _img_dia_html.get("overlay_b64"):
                _ov_b64_html = _img_dia_html["overlay_b64"]
            _overlay_img_tag = (
                f'<img src="data:image/png;base64,{_ov_b64_html}" '
                f'style="max-width:100%; max-height:320px; border:1px solid #ddd; border-radius:4px; margin-top:10px; display:block;">'
                if _ov_b64_html else ""
            )
            _method_lbl = escape(str(_src_html.get('method', '—')))
            _title_lbl  = "Medição do Diâmetro por Imagem (IA + OpenCV)" if _use_ia_html else "Medição do Diâmetro por Imagem (OpenCV + ArUco)"
            _img_section_html = f"""
          <div class="section" style="background:#f0faf5; border:1px solid #b2dfdb; border-radius:6px; padding:10px 14px; margin-top:10px;">
            <h3>&#128247; {_title_lbl}</h3>
            <div class="grid">
              <div class="kv"><span>Diâmetro estimado (mm)</span><b>{_fmt(_src_html.get('diameter_mm'))}</b></div>
              <div class="kv"><span>Diâmetro confirmado (mm)</span><b>{_fmt(_src_html.get('diameter_mm_confirmed'))}</b></div>
              <div class="kv"><span>Incerteza</span><b>{_unc_str}</b></div>
              <div class="kv"><span>Confiança</span><b>{_conf_str}</b></div>
              <div class="kv"><span>Método de detecção</span><b>{_method_lbl}</b></div>
              <div class="kv"><span>ArUco detectado?</span><b>{_aruco_ok}</b></div>
              <div class="kv"><span>Escala (mm/px)</span><b>{_scale_str}</b></div>
              <div class="kv"><span>Marcador real (mm)</span><b>{_fmt(_src_html.get('marker_real_mm'))}</b></div>
            </div>
            {_overlay_img_tag}
            <p style="font-size:9pt; color:#555; margin-top:6px;">
              Os valores acima foram gerados automaticamente por análise de imagem.
              O valor confirmado foi usado no cálculo. Registre a imagem original para auditoria.
            </p>
          </div>"""
        else:
            _img_section_html = ""

        # ── Gráfico transiente para o relatório HTML ────────────────────
        _transient_chart_html = ""
        if _is_trans_html and calc.get("transient_detail"):
            try:
                import altair as _alt
                import json as _json
                _df_det_rep = pd.DataFrame(calc["transient_detail"])
                _mass_cols_rep = [c for c in _df_det_rep.columns if c.endswith(" — Massa (kg)")]
                _ichart_rows_rep = []
                for _, _rr in _df_det_rep.iterrows():
                    _iv_lbl = f"Iv {int(_rr['Intervalo'])} ({_rr['dt (min)']} min, {_rr['p0 (bar abs)']} bar)"
                    for _mc in _mass_cols_rep:
                        _mname = _mc.replace(" — Massa (kg)", "")
                        if not pd.isna(_rr[_mc]):
                            _ichart_rows_rep.append({
                                "Intervalo": _iv_lbl,
                                "Número do Intervalo": int(_rr["Intervalo"]),
                                "Modelo": _mname,
                                "Massa (kg)": float(_rr[_mc]),
                            })
                if _ichart_rows_rep:
                    _df_crep = pd.DataFrame(_ichart_rows_rep)
                    _iv_order_rep = (
                        _df_crep.sort_values("Número do Intervalo")
                        ["Intervalo"].drop_duplicates().tolist()
                    )
                    _n_mod_rep = len(_df_crep["Modelo"].unique())
                    _chart_spec = (
                        _alt.Chart(_df_crep)
                        .mark_bar()
                        .encode(
                            x=_alt.X("Intervalo:N", sort=_iv_order_rep, title="Intervalo",
                                     axis=_alt.Axis(labelAngle=-35)),
                            y=_alt.Y("Massa (kg):Q", title="Massa Liberada (kg)"),
                            color=_alt.Color("Intervalo:N", sort=_iv_order_rep,
                                            legend=_alt.Legend(title="Intervalo")),
                            tooltip=[
                                _alt.Tooltip("Intervalo:N"),
                                _alt.Tooltip("Modelo:N"),
                                _alt.Tooltip("Massa (kg):Q", format=".4f"),
                            ],
                        )
                        .facet(
                            facet=_alt.Facet("Modelo:N", title="Modelo"),
                            columns=_n_mod_rep,
                        )
                        .properties(title="Evolução da Massa Liberada por Modelo (Regime Transiente)")
                    )
                    _spec_json = _json.dumps(_chart_spec.to_dict())
                    _transient_chart_html = f"""
          <div class="section" style="margin-top:12px;">
            <h3>Gráfico — Massa Liberada por Intervalo (Regime Transiente)</h3>
            <div id="transient-chart" style="width:100%; overflow-x:auto;"></div>
            <script>vegaEmbed('#transient-chart', {_spec_json}, {{renderer:'svg', actions:false}}).catch(console.error);</script>
          </div>"""
            except Exception as _e_crep:
                _transient_chart_html = f"<p><em>Gráfico não disponível: {_e_crep}</em></p>"

        html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>Relatório Vazão360 — {datetime.date.today().isoformat()}</title>
            {css}
            <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
        </head>
        <body>
        
          <!-- CABEÇALHO NOVO -->
          <table style="width:100%; border-collapse:collapse; margin-bottom:14px;">
            <tr>
                <td style="width:25%; text-align:center;">
                    {"<img src='data:image/png;base64," + logo_petrob_b64 + "' style='height:55px;'>" if logo_petrob_b64 else ""}
                </td>
        
                <td style="width:50%; text-align:center; font-size:18pt; font-weight:700;">
                    RELATÓRIO DE CLASSIFICAÇÃO DE EVENTO LOPC
                </td>
        
                <td style="width:25%; text-align:center;">
                    {"<img src='data:image/png;base64," + logo_b64 + "' style='height:70px;'>" if logo_b64 else ""}
                </td>
            </tr>
          </table>
        
          <div class="meta">Gerado em: {datetime.datetime.now().isoformat()}</div>
        
          <h2>Dados do Evento</h2>
          <div class="box">
            <div class="kv"><span><b>Data do Evento</b></span><span>{evt_date_str}</span></div>
            <div class="kv"><span><b>Analista (Chave)</b></span><span>{escape(evt_analyst) if evt_analyst else "—"}</span></div>
            <div style="margin-top:8px;">
              <div style="font-weight:600; margin-bottom:4px;">Descrição do Evento</div>
              <div>{evt_desc_html}</div>
            </div>
          </div>

          <h2>Dados do fluido</h2>
          <div class="kv"><span>Massa Específica (CNTP) [kg/m³]</span><b>{_fmt_rho(pc.rho_cntp)}</b></div>
          <div class="kv"><span>Massa Específica (Operação) [kg/m³]</span><b>{_fmt_rho(rho_oper)}</b></div>
          <div class="kv"><span>Massa molar média [kg/kmol]</span><b>{(_fmt(float(mw)) if mw is not None else "—")}</b></div>
          <div class="section">
            <h3>Composição do gás</h3>
            {comp_html}
          </div>

          <h2>Dados de operação</h2>
          <div class="grid">
            <div class="kv"><span>Regime</span><b>{escape(_regime_html)}</b></div>
            <div class="kv"><span>Pressão de Operação (bar abs){' &mdash; m&eacute;dia ponderada' if _is_trans_html else ''}</span><b>{_fmt(pc.p0_bar)}</b></div>
            <div class="kv"><span>Temperatura (°C)</span><b>{_fmt(pc.t0_c)}</b></div>
            <div class="kv"><span>Ambiente fechado?</span><b>{ambiente}</b></div>
          </div>
          {transient_table_html}
          {_transient_chart_html}

          <h2>Dados do evento</h2>
          <div class="grid">
            <div class="kv"><span>Diâmetro do furo (mm)</span><b>{_fmt(pc.orifice_d_mm)}</b></div>
            <div class="kv"><span>Diâmetro do tubo (mm)</span><b>{_fmt(pc.pipe_D_mm)}</b></div>
            <div class="kv"><span>Massa total liberada (kg)</span><b>{_fmt(calc['m_total'])}</b></div>
            <div class="kv"><span>Volume (CNTP) [m³]</span><b>{_fmt(calc['V_cnp'])}</b></div>
            <div class="kv"><span>Pessoas Feridas?</span><b>{evt_pessoas_feridas}</b></div>
          </div>
          {_img_section_html}

          <h2>Classificação do evento</h2>
          <div class="grid">
            <div class="kv"><span>Massa na pior hora (kg)</span><b>{_fmt(calc.get('m_pior', calc.get('q_pior', 0)))}</b></div>
            <div class="kv"><span>TIER (geral)</span><b>{calc.get('tier_geral','—')}</b></div>
            <div class="kv"><span>TIER (ambiente fechado)</span><b>{calc.get('tier_fechado','—')}</b></div>
        
            <!-- Comunicar ANP? -->
            <div class="kv">
              <span>Comunicar ANP?</span>
              <b style="color:{'#d32f2f' if comunicar_anp else '#2e7d32'}">{'Sim' if comunicar_anp else 'Não'}</b>
              <span style="margin-left:8px; display:inline-block; width:18px; height:18px; border-radius:50%; background:#; text-align:right; line-height:18px; font-weight:bold; border:1px solid rgba(0,0,0,0.08);" title="{tooltip_text}"></span>
            </div>
          </div>



          <div class="footer">
            Observação: este relatório compila o conteúdo da aba 03, incluindo Dados do evento, para impressão e rastreabilidade.
          </div>

        <div>
            <img class="footer-img" src="data:image/png;base64,{footer_b64}">
        </div>
        </body>
        </html>
        """
        return html

    printable_html = build_printable_html(
        pc=ProcessConditions(**st.session_state["process_conditions"]),
        composition=st.session_state["composition"],
        mix=st.session_state.get("mixture_props", {}),
        calc=calc
    )

    st.download_button(
        label="Baixar Relatório (imprimível)",
        data=printable_html.encode("utf-8"),
        file_name="relatorio_vazao360_imprimivel.html",
        mime="text/html"
    )
    st.caption("Dica: abra o HTML baixado e use **Imprimir → Salvar como PDF** para arquivar.")



