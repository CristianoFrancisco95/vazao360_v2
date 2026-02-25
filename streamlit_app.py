# streamlit_app.py — Home

import streamlit as st
import time
import base64
from pathlib import Path
from datetime import date

# ── 1. Page config (DEVE vir antes de qualquer comando Streamlit) ────────────
st.set_page_config(page_title="Vazão360 — Home", page_icon="🏠", layout="wide")

# ── 2. Login gate — bloqueia acesso sem autenticação ────────────────────────
from app.auth import require_login
require_login()

# ── 3. CSS global + sidebar (só chega aqui se autenticado) ──────────────────
from app.ui import inject_global_css, sidebar_brand, page_header
inject_global_css()
sidebar_brand()

# ── 4. Splash screen (1× por sessão, logo após o primeiro login) ─────────────

def _embed_logo_base64(path: Path) -> str:
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    return ""

def show_splash_once(duration_sec: float = 5.0):
    if "show_splash" not in st.session_state:
        st.session_state.show_splash = True
    if not st.session_state.show_splash:
        return

    # Caminho do logo (ajuste conforme sua estrutura)
    logo_path = Path(__file__).resolve().parent / "app" / "logo_vazao360.png"
    logo_b64 = _embed_logo_base64(logo_path)

    # HTML do splash com gradiente verde → azul-petróleo e fontes maiores
    splash_html = f"""
    <style>
      #splash-overlay {{
        position: fixed; inset: 0;
        z-index: 999999;
        background: linear-gradient(180deg, #2ECC71 0%, #004AAD 100%);
        display: flex; flex-direction: column; align-items: center; justify-content: center;
      }}
      .splash-loading {{
        color: #ffffff;
        font-size: 32px;           /* aumentada */
        font-weight: 700;
        font-family: 'Segoe UI', Arial, sans-serif;
        letter-spacing: 0.8px;
        margin-bottom: 20px;
        animation: pulse 1.2s ease-in-out infinite;
      }}
      .splash-logo {{
        width: min(60vw, 560px);
        max-width: 560px;
        margin-top: 18px;
        filter: drop-shadow(0 4px 18px rgba(0,0,0,0.25));
      }}
      .splash-title {{
        color: #fff;
        font-family: 'Segoe UI', Arial, sans-serif;
        text-align: center;
        margin-top: 18px;
        font-size: 22px;          /* aumentada */
        opacity: 0.95;
        font-weight: 500;
      }}
      @keyframes pulse {{
        0% {{ opacity: 0.4; }}
        50% {{ opacity: 1; }}
        100% {{ opacity: 0.4; }}
      }}
      .dots::after {{
        content: '…';
        animation: dots 1.2s steps(3, end) infinite;
      }}
      @keyframes dots {{
        0% {{ content: ''; }}
        33% {{ content: '.'; }}
        66% {{ content: '..'; }}
        100% {{ content: '...'; }}
      }}
    </style>

    <div id="splash-overlay">
      <div class="splash-loading">Carregando Software<span class="dots"></span></div>
      {"<img class='splash-logo' src='data:image/png;base64," + logo_b64 + "' alt='Vazão360'/>" if logo_b64 else ""}
      <div class="splash-title">Software de cálculo de eventos de LOPC</div>
    </div>
    """

    holder = st.empty()
    holder.markdown(splash_html, unsafe_allow_html=True)
    time.sleep(duration_sec)
    holder.empty()
    st.session_state.show_splash = False
    st.rerun()

# >>> Executa após login confirmado <<<
show_splash_once(duration_sec=5.0)

# ── 5. Conteúdo da Home ──────────────────────────────────────────────────────

st.title("Vazão360 ♨️ — Software de cálculo para eventos de LOPC")


# ====== DADOS DO EVENTO ======
st.markdown("### 📝 Dados do Evento")

def _ss_get(k, default):
    return st.session_state.get(k, default)

evt_desc = st.text_area(
    "Descrição do Evento",
    value=_ss_get("evt_desc", ""),
    placeholder="descrever brevemente o evento, número de RTA se houver, e número da plataforma.",
    help="Escreva em texto corrido. Ex.: Vazamento de gás em linha de 8'' no módulo X. RTA-12345. Plataforma P-XX.",
)
evt_date_default = _ss_get("evt_date", date.today())
evt_date = st.date_input(
    "Data do Evento",
    value=evt_date_default,
    format="DD/MM/YYYY",
    help="Selecione no calendário — será salvo como dd/mm/aaaa.",
)
# Chave autenticada — preenchida automaticamente e bloqueada
_user_key = st.session_state.get("user_key", "")
st.session_state["evt_analyst"] = _user_key          # sempre sincronizado
evt_analyst = st.text_input(
    "Analista (Chave)",
    value=_user_key,
    disabled=True,
    help="Preenchido automaticamente com a chave informada no login.",
)

col_save, col_hint = st.columns([1,3])
with col_save:
    if st.button("💾 Salvar Dados"):
        st.session_state["evt_desc"] = evt_desc
        st.session_state["evt_date"] = evt_date
        st.session_state["evt_date_str"] = evt_date.strftime("%d/%m/%Y") if evt_date else ""
        st.session_state["evt_analyst"] = evt_analyst
        st.success("Dados do evento salvos.")
with col_hint:
    st.caption("👉 Esses dados ficam disponíveis nas outras abas e exportações.")

st.markdown("---")

# ====== STATUS DO CENÁRIO ======
st.markdown("### 📦 Status do Cenário")
comp_loaded = ("composition" in st.session_state and isinstance(st.session_state["composition"], dict))
cond_loaded = ("process_conditions" in st.session_state)
props_loaded = ("mixture_props" in st.session_state)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Composição carregada", "Sim" if comp_loaded else "Não")
with c2:
    st.metric("Condições de processo", "Sim" if cond_loaded else "Não")
with c3:
    st.metric("Propriedades calculadas", "Sim" if props_loaded else "Não")

if not (comp_loaded and cond_loaded):
    st.info("Carregue/salve um cenário em **Entrada de Dados** para ver os detalhes completos.")
st.markdown("---")

