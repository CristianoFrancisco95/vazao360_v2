import streamlit as st
import pandas as pd
from app.core.models.composition import Composition
from app.core.models.inputs import ProcessConditions
from app.ui import inject_global_css, sidebar_brand, page_header
from app.auth import require_login



st.set_page_config(page_title="Vazão360 — Entrada de Dados", page_icon="📥", layout="wide")
require_login()
inject_global_css()
sidebar_brand()

st.title("02 — Entrada de Dados")

# ═══════════════════════════════════════════════════════════
# NOTA: este arquivo gerencia dois regimes de operação:
#   • Contínuo  – pressão de operação constante durante todo o evento
#   • Transiente – pressão de operação varia por intervalo de tempo
#                  (tabela editável; duração = soma dos intervalos)
# ═══════════════════════════════════════════════════════════

# --- Helpers para inicializar/persistir valores ---
def _get_ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# Valores padrão (usados apenas se não houver nada salvo ainda)
_defaults = {
    "p0_bar": 50.0,
    "pback_bar": 1.01325,
    "t0_c": 25.0,
    "d_mm": 10.0,
    "D_mm": 100.0,
    "dur_s": 60.0,
    "amb_fech": False,
    "pessoas_feridas": "não",
    "rho": 0.0,
    "mu": 1.0e-5,
    "tap_type": "flange",
    "L1_m": 0.0254,
    "regime": "Contínuo",
    "d_avail": "Sim",
}

# Se já existe cenário salvo, usa-o como default
if "process_conditions" in st.session_state:
    pc_saved = st.session_state["process_conditions"]
    _defaults.update({
        "p0_bar": float(pc_saved.get("p0_bar", _defaults["p0_bar"])),
        "pback_bar": float(pc_saved.get("pback_bar", _defaults["pback_bar"])),
        "t0_c": float(pc_saved.get("t0_c", _defaults["t0_c"])),
        "d_mm": float(pc_saved.get("orifice_d_mm", _defaults["d_mm"])),
        "D_mm": float(pc_saved.get("pipe_D_mm", _defaults["D_mm"])),
        "dur_s": float(pc_saved.get("duration_s", _defaults["dur_s"]) or _defaults["dur_s"]),
        "amb_fech": bool(pc_saved.get("ambiente_fechado", _defaults["amb_fech"])),
        "rho": float(pc_saved.get("rho_cntp", 0.0) or 0.0),
        "mu": float(pc_saved.get("mu_pa_s", _defaults["mu"])),
        "tap_type": pc_saved.get("tap_type", _defaults["tap_type"]),
        "L1_m": float(pc_saved.get("L1_m", _defaults["L1_m"])),
        "regime": pc_saved.get("regime", "Contínuo"),
        "d_avail": st.session_state.get("inp_d_avail", _defaults["d_avail"]),
    })

# Restaura pessoas_feridas da sessão (salvo separadamente)
if "pessoas_feridas" in st.session_state:
    _defaults["pessoas_feridas"] = st.session_state["pessoas_feridas"]

# Chaves de widgets (para conseguirmos limpá-los no reset)
W_KEYS = {
    "p0_bar": "inp_p0_bar",
    "pback_bar": "inp_pback_bar",
    "t0_c": "inp_t0_c",
    "d_mm": "inp_d_mm",
    "D_mm": "inp_D_mm",
    "dur_s": "inp_dur_s",
    "amb_fech": "inp_amb_fech",
    "pessoas_feridas": "inp_pessoas_feridas",
    "rho": "inp_rho",
    "mu": "inp_mu",
    "tap_type": "inp_tap_type",
    "L1_m": "inp_L1_m",
    "regime": "inp_regime",
    "d_avail": "inp_d_avail",
}

# --- Dados de Composição do Gás ---
_comp_mode_options = ["Sim", "Não"]
_comp_mode_default = st.session_state.get("composition_mode", "Sim")
comp_mode = st.selectbox(
    "Dados de Composição do Gás Disponíveis?",
    options=_comp_mode_options,
    index=_comp_mode_options.index(_comp_mode_default) if _comp_mode_default in _comp_mode_options else 0,
    key="inp_comp_mode",
    help="Selecione 'Sim' para informar a composição molar do gás. Selecione 'Não' para informar apenas a Massa Molar Estimada (k=Cp/Cv será fixado em 1,4).",
)

# --- Composição (fração molar) ---
if comp_mode == "Sim":
    st.subheader("Composição do Gás (fração molar)")
    components = [
        "C1 - Metano","C2 - Etano","C3 - Propano","iC4 - i-Butano","nC4 - n-Butano","iC5 - i-Pentano","nC5 - n-Pentano","nC6 - n-Hexano","nC7 - n-Heptano","nC8 - n-Octano","nC9 - n-Nonano","nC10+ - n-Decano+",
        "CO2 - Dióxido de Carbono","O2 - Oxigênio","N2 - Nitrogênio","H2O - Água","H2S - Ácido Sulfídrico"
    ]

    # Pré-carrega composição da sessão (se existir)
    if "composition" in st.session_state and st.session_state["composition"] is not None:
        comp_saved = st.session_state["composition"]
        saved_map = comp_saved.get("fractions", {})
        data = {"Componente": components, "Fração Molar": [float(saved_map.get(c, 0.0)) for c in components]}
    else:
        data = {"Componente": components, "Fração Molar": [0.0]*len(components)}

    df = pd.DataFrame(data)
    edited = st.data_editor(df, num_rows="fixed", width="stretch", key="comp_editor")
    fractions = dict(zip(edited["Componente"], edited["Fração Molar"]))
    total = sum(fractions.values())
    st.caption(f"Soma das frações: **{total:.5f}**")
    if not (0.999 <= total <= 1.001):
        st.warning("A soma das frações deve ser 1.0 (±0.001).")
    molar_mass_estimated = None
else:
    fractions = {}
    _molar_mass_est_default = float(st.session_state.get("molar_mass_estimated", 16.043))
    molar_mass_estimated = st.number_input(
        "Massa Molar Estimada (kg/kmol)",
        min_value=1.0,
        value=_molar_mass_est_default,
        key="inp_molar_mass_estimated",
        help="Massa Molar Média estimada do gás. Será usada como Massa Molar Média (kg/kmol) na Calculadora. O parâmetro k=Cp/Cv será fixado em 1,4.",
    )

# --- Condições de Processo ---
st.subheader("Condições de Processo")

# ── Regime ───────────────────────────────────────────────────────────
_reg_options = ["Contínuo", "Transiente"]
_reg_default = _get_ss(W_KEYS["regime"], _defaults["regime"])
regime = st.selectbox(
    "Regime",
    options=_reg_options,
    index=_reg_options.index(_reg_default) if _reg_default in _reg_options else 0,
    key=W_KEYS["regime"],
    help=(
        "Contínuo: pressão de operação constante durante todo o evento.\n"
        "Transiente: pressão de operação varia ao longo do evento — "
        "informe uma tabela de intervalos de tempo e pressões."
    ),
)
is_transient = (regime == "Transiente")

# ── Entradas principais ───────────────────────────────────────────────
if not is_transient:
    # Regime contínuo — entrada normal de p0
    col1, col2, col3 = st.columns(3)
    with col1:
        p0_bar = st.number_input("Pressão de Operação (bar abs)", key=W_KEYS["p0_bar"],
                                 min_value=0.001, value=_get_ss(W_KEYS["p0_bar"], _defaults["p0_bar"]))
    with col2:
        t0_c = st.number_input("Temperatura (°C)", key=W_KEYS["t0_c"],
                               value=_get_ss(W_KEYS["t0_c"], _defaults["t0_c"]))
    with col3:
        D_mm = st.number_input("Diâmetro Interno do Tubo (mm)", key=W_KEYS["D_mm"],
                               min_value=0.001, value=_get_ss(W_KEYS["D_mm"], _defaults["D_mm"]))
    transient_intervals = None  # não usado no contínuo
else:
    # Regime transiente — temperatura e diâmetro do tubo
    col2, col3 = st.columns(2)
    with col2:
        t0_c = st.number_input("Temperatura (°C)", key=W_KEYS["t0_c"],
                               value=_get_ss(W_KEYS["t0_c"], _defaults["t0_c"]))
    with col3:
        D_mm = st.number_input("Diâmetro interno do tubo (mm)", key=W_KEYS["D_mm"],
                               min_value=0.001, value=_get_ss(W_KEYS["D_mm"], _defaults["D_mm"]))

    # Tabela de intervalos transientes
    st.markdown(
        "**Pressão de Operação — Tabela de Intervalos Transientes**  \n"
        "Preencha cada intervalo de tempo (min) com a pressão correspondente. "
        "A duração total será calculada automaticamente como a soma dos intervalos."
    )

    # Carrega tabela salva na sessão (se existir)
    _saved_tt = None
    if "process_conditions" in st.session_state:
        _raw_tt = st.session_state["process_conditions"].get("transient_table")
        if _raw_tt:
            try:
                _saved_tt = [
                    {"Intervalo (min)": float(r.get("dt_min", r.get("Intervalo (min)", 0.0))),
                     "Pressão de Operação (bar abs)": float(r.get("p0_bar", r.get("Pressão de Operação (bar abs)", 1.0)))}
                    for r in _raw_tt
                ]
            except Exception:
                _saved_tt = None

    if _saved_tt is None:
        _saved_tt = [
            {"Intervalo (min)": 10.0, "Pressão de Operação (bar abs)": 50.0},
            {"Intervalo (min)": 20.0, "Pressão de Operação (bar abs)": 40.0},
            {"Intervalo (min)": 10.0, "Pressão de Operação (bar abs)": 30.0},
        ]

    # Converte para string com separador decimal "," para exibição
    def _to_str_comma(v, decimals=2):
        try:
            return f"{float(v):.{decimals}f}".replace(".", ",")
        except (ValueError, TypeError):
            return str(v)

    df_transient = pd.DataFrame({
        "Intervalo (min)":     [_to_str_comma(r["Intervalo (min)"],     2) for r in _saved_tt],
        "Pressão de Operação (bar abs)":[_to_str_comma(r["Pressão de Operação (bar abs)"],2) for r in _saved_tt],
    })

    edited_transient = st.data_editor(
        df_transient,
        num_rows="dynamic",
        key="transient_table_editor",
        column_config={
            "Intervalo (min)": st.column_config.TextColumn(
                "Intervalo (min)",
                help="Duração do intervalo em minutos — use vírgula como separador decimal (ex.: 10,5)",
            ),
            "Pressão de Operação (bar abs)": st.column_config.TextColumn(
                "Pressão de Operação (bar abs)",
                help="Pressão de Operação absoluta neste intervalo — use vírgula como separador decimal (ex.: 49,5)",
            ),
        },
    )

    # Parseia de volta para float (aceita tanto "," quanto "." como separador decimal)
    def _parse_float(v):
        try:
            return float(str(v).replace(",", ".").strip())
        except (ValueError, TypeError):
            return None

    _valid_rows_raw = [
        {
            "Intervalo (min)":      _parse_float(row["Intervalo (min)"]),
            "Pressão de Operação (bar abs)": _parse_float(row["Pressão de Operação (bar abs)"]),
        }
        for _, row in edited_transient.iterrows()
    ]
    _valid_rows = [
        r for r in _valid_rows_raw
        if r["Intervalo (min)"] is not None and r["Intervalo (min)"] > 0
        and r["Pressão de Operação (bar abs)"] is not None and r["Pressão de Operação (bar abs)"] > 0
    ]
    transient_intervals = [
        {"dt_min": r["Intervalo (min)"], "p0_bar": r["Pressão de Operação (bar abs)"]}
        for r in _valid_rows
    ]
    dur_transient_min = sum(r["dt_min"] for r in transient_intervals)

    # Botão "Atualizar" — confirma a soma dos intervalos e atualiza a Duração do Evento
    col_upd, col_info = st.columns([1, 4])
    with col_upd:
        if st.button("🔄 Atualizar", help="Clique para atualizar a Duração do evento com a soma atual dos intervalos"):
            st.session_state["transient_dur_confirmed"] = dur_transient_min
            # Força atualização do widget de duração (contorna cache interno do Streamlit)
            st.session_state["inp_dur_readonly"] = dur_transient_min
            st.rerun()

    # Valor confirmado (usa o botão; antes do primeiro clique, segue o valor calculado)
    dur_transient_min_confirmed = st.session_state.get("transient_dur_confirmed", dur_transient_min)
    # Garante que o slot interno do widget também está sincronizado
    st.session_state["inp_dur_readonly"] = dur_transient_min_confirmed

    with col_info:
        st.info(
            f"Duração total confirmada: **{dur_transient_min_confirmed:.2f} min** "
            f"— tabela atual: **{dur_transient_min:.2f} min** ({len(transient_intervals)} intervalo(s))"
        )
    if not transient_intervals:
        st.warning("Adicione pelo menos um intervalo válido na tabela acima.")

    # Pressão representativa = média ponderada pelo intervalo (usada para props da mistura)
    if dur_transient_min > 0:
        p0_bar = sum(r["dt_min"] * r["p0_bar"] for r in transient_intervals) / dur_transient_min
    else:
        p0_bar = _defaults["p0_bar"]

# --- Dados do Evento ---
st.subheader("Dados do Evento")
colE1, colE2, colE3 = st.columns(3)
with colE1:
    pback_bar = st.number_input("Pressão na Saída do Furo/Ambiente (bar abs)", key=W_KEYS["pback_bar"],
                                min_value=0.001, value=_get_ss(W_KEYS["pback_bar"], _defaults["pback_bar"]))
with colE2:
    if not is_transient:
        dur_min_default = (_defaults["dur_s"] / 60.0) if _defaults["dur_s"] else 0.0
        dur_min = st.number_input(
            "Duração do Evento (min)",
            key=W_KEYS["dur_s"],
            min_value=0.0,
            value=_get_ss(W_KEYS["dur_s"], dur_min_default),
            step=1.0,
        )
    else:
        dur_min = dur_transient_min_confirmed
        st.number_input(
            "Duração do evento (min) — confirmada pelo botão Atualizar",
            value=float(dur_min),
            disabled=True,
            key="inp_dur_readonly",
            help="Soma dos intervalos transientes confirmada pelo botão 'Atualizar' na seção Condições de Processo. Não editável.",
        )
    amb_fech = st.toggle(
        "Ambiente fechado?",
        key=W_KEYS["amb_fech"],
        value=_get_ss(W_KEYS["amb_fech"], _defaults["amb_fech"])
    )
with colE3:
    _pf_options = ["não", "sim"]
    _pf_default = _get_ss(W_KEYS["pessoas_feridas"], _defaults["pessoas_feridas"])
    pessoas_feridas = st.selectbox(
        "Pessoas Feridas?",
        options=_pf_options,
        index=_pf_options.index(_pf_default) if _pf_default in _pf_options else 0,
        key=W_KEYS["pessoas_feridas"],
        help="Informe se houve pessoas feridas no evento."
    )

# ═══════════════════════════════════════════════════════════
# DIÂMETRO DO ORIFÍCIO / FURO
# ═══════════════════════════════════════════════════════════
st.markdown("**Diâmetro do Furo / Orifício**")

_d_source_opts = ["Importar da página 'Diâmetro por Imagem (IA)'", "Inserir manualmente"]
_d_source = st.radio(
    "Valor do Diâmetro do Furo/Orifício",
    options=_d_source_opts,
    key="d_source_radio",
    horizontal=True,
)

if _d_source == _d_source_opts[0]:
    _img_meta = st.session_state.get("orifice_image_meta")
    if _img_meta:
        _d_imported = float(_img_meta.get("diameter_mm_confirmed") or _img_meta.get("diameter_mm", 0.0))
        st.success(
            f"✅ Diâmetro importado da página 'Diâmetro por Imagem (IA)': **{_d_imported:.3f} mm**  \n"
            f"Incerteza: ±{_img_meta.get('uncertainty_pct') or 0:.0f}%"
        )
        st.session_state[W_KEYS["d_mm"]] = _d_imported
        d_mm = _d_imported
    else:
        st.info(
            "Nenhum resultado disponível. Acesse a página **Diâmetro por Imagem (IA)** "
            "no menu lateral, calcule o diâmetro e clique em "
            "**✅ Usar este diâmetro na Calculadora**."
        )
        d_mm = float(st.session_state.get(W_KEYS["d_mm"], _defaults["d_mm"]))
else:
    d_mm = st.number_input(
        "Diâmetro do orifício d (mm)",
        key=W_KEYS["d_mm"],
        min_value=0.001,
        value=_get_ss(W_KEYS["d_mm"], _defaults["d_mm"]),
    )

st.subheader("Propriedades do Escoamento & Tomadas de Pressão")
colA, colB, colC = st.columns(3)
with colA:
    # OBRIGATÓRIO agora
    rho_cntp = st.number_input("Massa Específica (CNTP) [kg/m³]", key=W_KEYS["rho"],
                               min_value=0.001, value=_get_ss(W_KEYS["rho"], 1.250))
with colB:
    mu_pa_s = st.number_input("Viscosidade Dinâmica (Pa·s)", key=W_KEYS["mu"],
                              min_value=1e-6, value=_get_ss(W_KEYS["mu"], _defaults["mu"]), format="%.2e")
with colC:
    tap_type = st.selectbox("Tipo de Tomada de Pressão", key=W_KEYS["tap_type"],
                            options=["flange","corner","D_D/2"],
                            index=["flange","corner","D_D/2"].index(_get_ss(W_KEYS["tap_type"], _defaults["tap_type"])),
                            help="Flange: L1≈1'' (0,0254 m), Corner: L1≈0, D&D/2: L1≈D")
L1_m = st.number_input("L1 (m)", key=W_KEYS["L1_m"],
                       min_value=0.0, value=_get_ss(W_KEYS["L1_m"], _defaults["L1_m"]),
                       help="Distância a montante da tomada")

col_save, col_reset = st.columns([2,1])

# --- CSS customizado do botão ---
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #10bb82;       /* azul */
        color: white;                    /* texto branco */
        font-size: 30px;                 /* letras maiores */
        font-weight: bold;
        height: 2.5em;                   /* aumenta altura */
        width: 50%;                     /* ocupa toda a coluna */
        border-radius: 8px;              /* cantos arredondados */
        border: none;
        transition: 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #27ae60;       /* verde mais escuro no hover */
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with col_save:
    if st.button("SALVAR CENÁRIO E CALCULAR"):
        try:
            comp_mode_val = st.session_state.get("inp_comp_mode", "Sim")
            if comp_mode_val == "Sim":
                comp = Composition(fractions=fractions)
            else:
                comp = None

            # Duração em segundos
            dur_s_val = float(dur_min) * 60.0 if float(dur_min) > 0 else None

            # Pressão representativa (para contínuo, usa widget; transiente já calculou a média)
            _p0_save = float(p0_bar) if is_transient else float(st.session_state[W_KEYS["p0_bar"]])

            pc_kwargs = dict(
                p0_bar=_p0_save,
                pback_bar=float(st.session_state[W_KEYS["pback_bar"]]),
                t0_c=float(st.session_state[W_KEYS["t0_c"]]),
                orifice_d_mm=float(d_mm),
                pipe_D_mm=float(st.session_state[W_KEYS["D_mm"]]),
                duration_s=dur_s_val,
                ambiente_fechado=bool(st.session_state[W_KEYS["amb_fech"]]),
                rho_cntp=float(st.session_state[W_KEYS["rho"]]),
                mu_pa_s=float(st.session_state[W_KEYS["mu"]]),
                tap_type=st.session_state[W_KEYS["tap_type"]],
                L1_m=float(st.session_state[W_KEYS["L1_m"]]),
                regime=st.session_state[W_KEYS["regime"]],
            )

            if is_transient:
                if not transient_intervals:
                    st.error("Adicione pelo menos um intervalo válido na tabela transiente antes de salvar.")
                    st.stop()
                pc_kwargs["transient_table"] = transient_intervals
            else:
                pc_kwargs["transient_table"] = None

            pc = ProcessConditions(**pc_kwargs)

            st.session_state["composition"] = comp.model_dump() if comp is not None else None
            st.session_state["composition_mode"] = comp_mode_val
            if comp_mode_val == "Não":
                _mw_est = float(st.session_state.get("inp_molar_mass_estimated", 16.043))
                st.session_state["molar_mass_estimated"] = _mw_est
            else:
                st.session_state.pop("molar_mass_estimated", None)
            st.session_state["process_conditions"] = pc.model_dump()
            st.session_state["pessoas_feridas"] = st.session_state.get(W_KEYS["pessoas_feridas"], "não")
            # Invalida caches de cálculo para forçar recálculo na Calculadora
            for _stale in ("mixture_props", "qm_median"):
                st.session_state.pop(_stale, None)
            st.success("Cenário salvo. Prossiga para **03 — Calculadora**.")
        except Exception as e:
            st.error(f"Erro de validação: {e}")