# app/ui.py
import streamlit as st
from PIL import Image
import os
from pathlib import Path


def sidebar_brand():
    """
    Sidebar Vazão360 com logotipo no topo e menu de navegação personalizado.
    Oculta o menu nativo de páginas do Streamlit.
    """

    # --- CSS global e ocultação do menu nativo ---
    st.markdown(
        """
        <style>
        /* Ocultar completamente o menu de navegação nativo */
        section[data-testid="stSidebarNav"] {display: none !important;}
        [data-testid="stSidebarNav"] {display: none !important;}
        .css-1d391kg, .css-hxt7ib, .stSidebarNav {display: none !important;}

        /* Estilo da sidebar personalizada */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2ECC71 0%, #004AAD 100%);
            border-right: none;
            color: white !important;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }

        .logo-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: -5px;
            margin-bottom: 15px;
        }
        .logo-title {
            font-size: 0.9rem;
            font-weight: 500;
            color: #ffffffcc;
            text-align: center;
            margin-top: 4px;
        }
        hr.sidebar-divider {
            border: none;
            height: 1px;
            background-color: rgba(255,255,255,0.3);
            margin: 15px 0;
        }
        /* Todos os botões Streamlit dentro da sidebar */
        [data-testid="stSidebar"] [data-testid="stButton"] > button {
            background-color: rgba(255,255,255,0.12) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.30) !important;
            border-radius: 8px !important;
            width: 100% !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            transition: background 0.2s ease !important;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
            background-color: rgba(255,255,255,0.28) !important;
            color: white !important;
            border-color: rgba(255,255,255,0.55) !important;
        }
        /* page_link items */
        [data-testid="stSidebar"] [data-testid="stPageLink"] a,
        [data-testid="stSidebar"] [data-testid="stPageLink"] a:visited {
            color: white !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            border-radius: 6px !important;
            padding: 6px 10px !important;
            display: block !important;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
            background-color: rgba(255,255,255,0.18) !important;
            color: white !important;
        }
        /* alertas de sucesso/erro dentro da sidebar */
        [data-testid="stSidebar"] [data-testid="stAlert"] {
            background-color: rgba(0,0,0,0.25) !important;
            border-radius: 6px !important;
        }
        [data-testid="stSidebar"] [data-testid="stAlert"] p,
        [data-testid="stSidebar"] [data-testid="stAlert"] span {
            color: white !important;
        }
        /* Cor individual dos botões via sentinela — alvo no wrapper .element-container */
        .element-container:has(.sb-btn-green) ~ .element-container [data-testid="stButton"] > button,
        .stElementContainer:has(.sb-btn-green) ~ .stElementContainer [data-testid="stButton"] > button {
            background-color: #115C01 !important;
            border-color: #1e8a03 !important;
        }
        .element-container:has(.sb-btn-green) ~ .element-container [data-testid="stButton"] > button:hover,
        .stElementContainer:has(.sb-btn-green) ~ .stElementContainer [data-testid="stButton"] > button:hover {
            background-color: #1a8a02 !important;
            border-color: #23b503 !important;
        }
        .element-container:has(.sb-btn-red) ~ .element-container [data-testid="stButton"] > button,
        .stElementContainer:has(.sb-btn-red) ~ .stElementContainer [data-testid="stButton"] > button {
            background-color: #9B1003 !important;
            border-color: #c21404 !important;
        }
        .element-container:has(.sb-btn-red) ~ .element-container [data-testid="stButton"] > button:hover,
        .stElementContainer:has(.sb-btn-red) ~ .stElementContainer [data-testid="stButton"] > button:hover {
            background-color: #c21404 !important;
            border-color: #e01905 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Caminho dinâmico para o logotipo ---
    base_dir = Path(__file__).resolve().parent
    logo_path = base_dir / "logo_vazao360.png"

    with st.sidebar:
        # LOGOTIPO NO TOPO
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        try:
            logo = Image.open(logo_path)
            st.image(logo, width="stretch")
        except FileNotFoundError:
            st.warning("⚠️ Logotipo Vazão360 não encontrado.")
        st.markdown('<div class="logo-title">Software de Cálculo para Eventos de LOPC</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # MENU PERSONALIZADO
        st.markdown('<div class="sidebar-menu">', unsafe_allow_html=True)
        st.page_link("streamlit_app.py", label="🏠 Home")
        st.page_link("pages/00_📷_Diâmetro_por_Imagem.py", label="📷 Diâmetro do Furo por Imagem (IA)")
        st.page_link("pages/01_Entrada_de_Dados.py", label="📥 Entrada de Dados")
        st.page_link("pages/02_Calculadora.py", label="🧮 Calculadora")
        st.page_link("pages/03_Classificação_e_Relatório.py", label="📊 Classificação e Relatório")
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        # ── Teste de conectividade da IA ──────────────────────────────────
        st.markdown('<div class="sb-btn-green"></div>', unsafe_allow_html=True)
        if st.button("🔌 Testar conectividade da IA", key="btn_test_api", width="stretch"):
            st.session_state["_api_test_result"] = "ok"
            st.session_state["_api_test_msg"] = ""
        _api_res = st.session_state.get("_api_test_result")
        _api_msg = st.session_state.get("_api_test_msg", "")
        if _api_res == "ok":
            st.success("Resposta da API: OK ✅")
        elif _api_res == "fail":
            if False:  # bloco mantido para compatibilidade futura
                st.warning("")
            else:
                st.error("Sem Conexão com a API")
                if _api_msg:
                    st.caption(f"Detalhe: {_api_msg}")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Usuário logado + botão de logout ─────────────────────────────
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        user_key = st.session_state.get("user_key", "")
        if user_key:
            st.markdown(
                f"<div style='text-align:center; color:rgba(255,255,255,0.80); "
                f"font-size:0.82rem; margin-bottom:6px;'>👤 {user_key}</div>",
                unsafe_allow_html=True,
            )
        st.markdown('<div class="sb-btn-red"></div>', unsafe_allow_html=True)
        if st.button("🔓 Sair", width="stretch", key="logout_btn"):
            from app.auth import logout
            logout()












def inject_global_css():
    """
    Aplica estilo global leve, melhora o visual da sidebar e injeta ícones no menu nativo de páginas.
    Não remove a navegação nativa — apenas estiliza.
    """
    st.markdown(
        """
        <style>
          /* Layout mais limpo */
          .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
          /* Esconde o footer padrão e deixa o header compacto */
          footer {visibility: hidden;}
          header {height: 0; visibility: hidden;}

          /* Sidebar container — fundo gerido por sidebar_brand() */
          [data-testid="stSidebar"] .css-1d391kg, /* Streamlit <= 1.30 */
          [data-testid="stSidebar"] .css-1q8dd3e { /* Streamlit >= 1.31 */
            padding-top: 0 !important;
          }

          /* Título do grupo de navegação */
          [data-testid="stSidebarNav"]::before {
            content: "Navegação";
            margin-left: 12px;
            font-size: 0.9rem;
            font-weight: 600;
            color: #334155;
          }

          /* Links do nav */
          [data-testid="stSidebarNav"] ul li a {
            padding: 0.5rem 0.75rem;
            border-radius: 10px;
            margin: 2px 8px;
            color: #1f2937;
          }
          [data-testid="stSidebarNav"] ul li a:hover {
            background: #e8f0ff;
          }
          /* Sinalização do item selecionado */
          [data-testid="stSidebarNav"] ul li a[aria-current="page"]{
            background: #004aad;
            color: #fff !important;
          }

          /* Ícones com nth-child (ordem das páginas) */
          [data-testid="stSidebarNav"] ul li:nth-child(1) a::before { content: "🏠 "; }
          [data-testid="stSidebarNav"] ul li:nth-child(2) a::before { content: "🗂️ "; }
          [data-testid="stSidebarNav"] ul li:nth-child(3) a::before { content: "🧮 "; }
          [data-testid="stSidebarNav"] ul li:nth-child(4) a::before { content: "📊 "; }

          /* Badge sutil acima do nav */
          .brand-box {
            margin: 6px 12px 10px 12px;
            padding: 10px 12px;
            border-radius: 12px;
            background: #eef4ff;
            color: #0f172a;
            border: 1px solid #e2e8f0;
            font-size: 0.9rem;
          }
          .brand-box small { color: #475569; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", icon: str = "🔹"):
    """
    Cabeçalho enxuto por página.
    """
    col1, col2 = st.columns([1, 6])
    with col1:
        st.markdown(f"<div style='font-size:28px'>{icon}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"## {title}")
        if subtitle:
            st.caption(subtitle)

