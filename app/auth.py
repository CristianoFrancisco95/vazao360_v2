# app/auth.py — Login / autenticação Vazão360

import base64
import os
import urllib.parse
import streamlit as st
from pathlib import Path

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────
#  Helpers internos
# ──────────────────────────────────────────────────────────────────

def _embed_logo_base64(path: Path) -> str:
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    return ""


# ──────────────────────────────────────────────────────────────────
#  Proxy Petrobras
# ──────────────────────────────────────────────────────────────────

_PROXY_HOST = "inet-sys.petrobras.com.br"
_PROXY_PORT = 804
_NO_PROXY    = "127.0.0.1,localhost,petrobras.com.br,petrobras.biz"
_VERIFY_URL  = "http://www.petrobras.com.br"  # URL usada para testar as credenciais


def configurar_proxy_petrobras(username: str, password: str) -> "httpx.Client | None":
    """
    Configura variáveis de ambiente de proxy e retorna um httpx.Client
    pronto para uso com as credenciais Petrobras.
    Retorna None se httpx não estiver instalado.
    """
    proxy_url = (
        f"http://{urllib.parse.quote(username)}:"
        f"{urllib.parse.quote(password)}@{_PROXY_HOST}:{_PROXY_PORT}"
    )

    os.environ["HTTP_PROXY"]  = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["NO_PROXY"]    = _NO_PROXY

    if not _HTTPX_AVAILABLE:
        return None

    proxies = {
        "http://":  proxy_url,
        "https://": proxy_url,
    }
    http_client = httpx.Client(
        proxies=proxies,
        verify=False,       # certificados corporativos internos
        timeout=10.0,
    )
    return http_client


def verificar_credenciais(username: str, password: str) -> tuple[bool, str]:
    """
    Tenta acessar _VERIFY_URL através do proxy Petrobras com as credenciais
    fornecidas.

    Retorna (True, "") em caso de sucesso ou
           (False, mensagem_de_erro) em caso de falha.
    """
    if not _HTTPX_AVAILABLE:
        # httpx não instalado — aceita as credenciais sem verificação de rede
        configurar_proxy_petrobras(username, password)
        return True, ""

    try:
        client = configurar_proxy_petrobras(username, password)
        response = client.head(_VERIFY_URL, follow_redirects=True)
        client.close()

        if response.status_code == 407:
            return False, "Credenciais inválidas (proxy rejeitou usuário/senha)."
        # qualquer outro código HTTP indica que o proxy aceitou as credenciais
        return True, ""

    except httpx.ProxyError:
        return False, "Não foi possível conectar ao proxy Petrobras."
    except httpx.TimeoutException:
        return False, "Tempo de conexão esgotado. Verifique se está na rede Petrobras."
    except Exception as exc:  # noqa: BLE001
        return False, f"Erro ao verificar credenciais: {exc}"


# ──────────────────────────────────────────────────────────────────
#  API pública
# ──────────────────────────────────────────────────────────────────

def is_authenticated() -> bool:
    """Verdadeiro se o usuário já fez login nesta sessão."""
    return st.session_state.get("authenticated", False)


def logout():
    """Encerra a sessão, apaga todos os dados e volta para a tela de login."""
    st.session_state.clear()
    st.rerun()


def require_login():
    """
    Chama esta função no topo de cada página.
    Se o usuário não estiver autenticado, exibe o formulário de login e
    para a execução (st.stop).  Caso contrário, retorna normalmente.
    """
    if is_authenticated():
        return

    _render_login_page()
    st.stop()


# ──────────────────────────────────────────────────────────────────
#  Renderização da tela de login
# ──────────────────────────────────────────────────────────────────

def _render_login_page():
    logo_path = Path(__file__).resolve().parent / "logo_vazao360.png"
    logo_b64 = _embed_logo_base64(logo_path)

    # ---------- CSS da tela de login ----------
    st.markdown(
        """
        <style>
        /* Ocultar sidebar e barra superior na tela de login */
        [data-testid="stSidebar"]   { display: none !important; }
        [data-testid="stHeader"]    { display: none !important; }
        header                      { visibility: hidden; height: 0; }
        footer                      { visibility: hidden; }
        .block-container            { padding-top: 0 !important; }

        /* Fundo com gradiente Petrobras */
        .stApp {
            background: linear-gradient(160deg, #e8f5e9 0%, #e3f0ff 100%);
        }

        /* Card central */

        .login-card h2 {
            text-align: center;
            color: #004AAD;
            font-size: 1.45rem;
            margin-bottom: 2px;
        }
        .login-subtitle {
            text-align: center;
            color: #64748b;
            font-size: 0.88rem;
            margin-bottom: 28px;
        }
        .login-footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.78rem;
            margin-top: 20px;
        }

        /* Botão de entrada */
        div[data-testid="stForm"] button[kind="primaryFormSubmit"],
        div[data-testid="stForm"] button {
            background: linear-gradient(90deg, #2ECC71 0%, #004AAD 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            padding: 10px 0 !important;
        }
        div[data-testid="stForm"] button:hover {
            opacity: 0.9 !important;
            cursor: pointer !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Centraliza verticalmente com espaçamento
    st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)

    # Usa colunas para centralizar o card
    _, col, _ = st.columns([1, 1.6, 1])

    with col:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)

        # Logo
        if logo_b64:
            st.markdown(
                f"<div style='text-align:center; margin-bottom:14px;'>"
                f"<img src='data:image/png;base64,{logo_b64}' "
                f"style='width:200px; filter: drop-shadow(0 2px 8px rgba(0,74,173,0.15));'/>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<h2>Acesso ao Sistema</h2>", unsafe_allow_html=True)
        st.markdown(
            "<div class='login-subtitle'>Software de Cálculo para Eventos de LOPC</div>",
            unsafe_allow_html=True,
        )

        # ---------- Formulário ----------
        with st.form("login_form", clear_on_submit=False):
            chave = st.text_input(
                "Chave Petrobras",
                placeholder="Ex.: R2D2",
                help="Sua chave de acesso à rede Petrobras.",
            )
            senha = st.text_input(
                "Senha",
                type="password",
                placeholder="Sua senha Petrobras",
            )
            entrar = st.form_submit_button("Entrar →", width="stretch")

        # ---------- Validação ----------
        if entrar:
            if not chave.strip() or not senha.strip():
                st.error("Preencha a Chave Petrobras e a Senha para continuar.")
            else:
                with st.spinner("Verificando credenciais…"):
                    ok, msg = verificar_credenciais(
                        chave.strip(), senha
                    )
                if ok:
                    st.session_state["authenticated"] = True
                    st.session_state["user_key"] = chave.strip().upper()
                    st.rerun()
                else:
                    st.error(msg or "Credenciais inválidas. Tente novamente.")

        st.markdown(
            "<div class='login-footer'>Uso restrito a colaboradores Petrobras</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
