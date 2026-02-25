"""
app/integrations/openai_vision.py
==================================
Integração com OpenAI Vision API (Azure OpenAI / Petrobras Gateway).

Expõe uma única função pública:
    segment_hole_polygon_openai(img_bytes: bytes) -> dict

Retorna:
    {
        "hole_polygon": [{"x": float, "y": float}, ...],  # 0..1
        "confidence":   float,
        "notes":        str,
        "_raw":         str,   # resposta bruta da API para depuração
    }

Se a API não estiver configurada ou falhar, retorna dict com hole_polygon=[].

Configuração via variáveis de ambiente (arquivo .env na raiz do projeto):
    OPENAI_API_KEY         — obrigatório
    DEPLOYMENT             — nome do deployment Azure (ex.: gpt-4o-mini-petrobras)
    MODEL                  — nome do modelo (fallback quando sem deployment)
    API_VERSION            — versão da API Azure (ex.: 2024-06-01)
    AZURE_OPENAI_ENDPOINT  — URL base Azure (omitir para usar OpenAI direto)
    OPENAI_BASE_URL        — URL base customizada (API Gateway)
    CA_CERT_PATH           — caminho do cert CA corporativo (opcional)
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── Carrega .env ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    # Procura .env na raiz do projeto (dois níveis acima deste arquivo)
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass  # dotenv não instalado — usa só vars de ambiente já definidas


# ── Disponibilidade do openai ─────────────────────────────────────────────────
try:
    import openai as _openai_lib
    _OPENAI_SDK_AVAILABLE = True
except ImportError:
    _OPENAI_SDK_AVAILABLE = False


# ── Prompt de segmentação ─────────────────────────────────────────────────────
_SEGMENTATION_PROMPT = """Você é um modelo de visão computacional especializado em inspeção industrial.

Tarefa: identificar e segmentar o CONTORNO EXTERNO da abertura real de um furo em uma superfície metálica (tubulação/equipamento).

IMPORTANTE — NÃO segmente apenas a região mais escura:
  - O furo real inclui o bisel/chanfro metálico e a transição metal → vazio.
  - A borda do furo é definida pela MUDANÇA ESTRUTURAL DA SUPERFÍCIE (metal sólido → abertura vazada), não apenas pela cor.
  - Ignore marcadores ArUco (quadrados preto/branco com bordas regulares — são etiquetas de calibração, NÃO são o furo).

Regras de segmentação:
1) Identifique a abertura física real no metal: interior muito escuro/profundo, bordas irregulares típicas de ruptura, corrosão ou impacto.
2) Expanda radialmente a partir do centro do furo até encontrar:
   - mudança consistente de textura (metal → vazio),
   - borda física do metal (anel/bisel), mesmo que seja marrom, oxidada ou com sombra.
3) Inclua TODO o contorno externo da abertura real (núcleo escuro + penumbra + gradiente de borda).
4) Se o furo for irregular/oblongo/dentado, use pontos suficientes para capturar as reentrâncias reais.

Validação obrigatória antes de responder:
  - O polígono deve tocar a borda do metal em TODOS os lados.
  - Se apenas a área preta interna foi segmentada, EXPANDA até o limite físico do metal.
  - Se o furo for alongado/inclinado, o polígono deve ter a mesma orientação — não force forma circular.

Saída — retorne APENAS JSON válido, sem texto adicional:
{
  "outer_polygon": [{"x": 0.0, "y": 0.0}, ...],
  "confidence": 0.0,
  "notes": "breve observação: forma do furo, onde a borda foi definida"
}

Requisitos de formato:
- Mínimo 24 pontos no outer_polygon.
- Coordenadas normalizadas (0..1): x = esquerda→direita, y = topo→baixo.
- SEMPRE retorne o melhor polígono possível, mesmo que a confiança seja baixa. Só retorne outer_polygon vazio se não houver nenhuma abertura física visível na imagem.
- NÃO retorne texto fora do JSON."""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_client():
    """
    Retorna um cliente OpenAI configurado para Azure ou OpenAI padrão,
    usando variáveis de ambiente e certificado CA corporativo quando disponível.
    """
    if not _OPENAI_SDK_AVAILABLE:
        raise RuntimeError(
            "Biblioteca 'openai' não encontrada. "
            "Instale com: pip install openai"
        )

    api_key = os.getenv("OPENAI_API_KEY", "")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_version = os.getenv("API_VERSION", "2024-06-01")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    ca_cert_raw = os.getenv("CA_CERT_PATH", "").strip()

    # O SDK AzureOpenAI adiciona /openai automaticamente ao azure_endpoint.
    # Remove a cauda /openai se o usuário já a incluiu na URL, evitando duplicação.
    def _strip_openai_suffix(url: str) -> str:
        for suffix in ("/openai/", "/openai"):
            if url.endswith(suffix.rstrip("/")):
                return url[: -len(suffix.rstrip("/"))]
        return url

    if azure_endpoint:
        azure_endpoint = _strip_openai_suffix(azure_endpoint)
    if base_url:
        base_url = _strip_openai_suffix(base_url)

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY não configurada. "
            "Defina a variável de ambiente ou adicione ao arquivo .env."
        )

    # Resolve cert CA e proxy corporativo Petrobras
    _http_client = None
    try:
        import httpx as _httpx
        # Resolve caminho do CA relativo à raiz do projeto
        _project_root = Path(__file__).resolve().parents[2]
        if ca_cert_raw:
            _ca_abs = (
                Path(ca_cert_raw)
                if Path(ca_cert_raw).is_absolute()
                else _project_root / ca_cert_raw
            )
            _verify: bool | str = str(_ca_abs) if _ca_abs.exists() else False
        else:
            _verify = False  # sem CA: desativa verificação (rede corporativa)

        # trust_env=False: ignora HTTP_PROXY/HTTPS_PROXY definidos pelo auth.py
        # (proxy corporativo Petrobras), pois as chamadas à API OpenAI devem ir
        # diretamente à internet, tanto no Streamlit Cloud quanto em rede externa.
        _http_client = _httpx.Client(verify=_verify, timeout=90.0, trust_env=False)
    except ImportError:
        pass  # httpx nao disponível — openai usa urllib

    _common_kwargs: dict = {}
    if _http_client is not None:
        _common_kwargs["http_client"] = _http_client

    if azure_endpoint:
        # Azure OpenAI (endpoint explícito)
        return _openai_lib.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            **_common_kwargs,
        )
    elif base_url:
        # OpenAI com URL customizada (API Gateway)
        return _openai_lib.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
            **_common_kwargs,
        )
    else:
        # OpenAI padrão — deployment usado como model name
        return _openai_lib.OpenAI(
            api_key=api_key,
            **_common_kwargs,
        )


def _resolve_model() -> str:
    """Retorna o nome do deployment/modelo a ser usado."""
    return (
        os.getenv("DEPLOYMENT")
        or os.getenv("MODEL")
        or "gpt-4o-mini"
    )


def _extract_json_from_text(text: str) -> str:
    """
    Extrai o primeiro bloco JSON da resposta (suporta textos com markdown extras).
    """
    # Remove blocos de código markdown
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Localiza substring JSON
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text


def _validate_polygon(polygon: list) -> list:
    """
    Valida e normaliza o polígono retornado pela IA.
    - Garante mínimo de 4 pontos (descarta se menos)
    - Clipa coords a 0..1
    - Fecha o polígono se o primeiro e último ponto forem diferentes
    """
    if not isinstance(polygon, list) or len(polygon) < 4:
        return []
    valid: list = []
    for pt in polygon:
        if not isinstance(pt, dict):
            continue
        try:
            x = max(0.0, min(1.0, float(pt.get("x", pt.get("X", 0)))))
            y = max(0.0, min(1.0, float(pt.get("y", pt.get("Y", 0)))))
            valid.append({"x": x, "y": y})
        except (TypeError, ValueError):
            continue
    if len(valid) < 4:
        return []
    # Fecha o polígono se necessário
    if valid[0] != valid[-1]:
        valid.append(valid[0])
    return valid


# ── API pública ───────────────────────────────────────────────────────────────

def segment_hole_polygon_openai(
    img_bytes: bytes,
) -> dict:
    """
    Envia a imagem para o modelo OpenAI Vision e retorna o polígono do furo.

    Parâmetros
    ----------
    img_bytes   : bytes da imagem (JPEG / PNG)

    Retorna
    -------
    dict com chaves:
        hole_polygon  : lista de pontos {"x", "y"} normalizados 0..1
        confidence    : float 0..1
        notes         : str
        _raw          : str (resposta bruta para depuração)
        _error        : str (preenchido em caso de falha)
    """
    _empty = {
        "outer_polygon": [], "inner_polygon": [], "edge_samples": [],
        "confidence": 0.0, "confidence_outer": 0.0,
        "notes": "", "_raw": "", "_error": "",
    }

    if not _OPENAI_SDK_AVAILABLE:
        _empty["_error"] = (
            "Biblioteca 'openai' não instalada. "
            "Execute: pip install openai"
        )
        return _empty

    if not os.getenv("OPENAI_API_KEY", "").strip():
        _empty["_error"] = (
            "OPENAI_API_KEY não configurada. "
            "Adicione ao arquivo .env na raiz do projeto."
        )
        return _empty

    # Detecta mime type da imagem original
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    mime = "image/jpeg" if img_bytes[:3] == b"\xff\xd8\xff" else "image/png"

    model = _resolve_model()

    # Monta o prompt completo
    prompt_text = _SEGMENTATION_PROMPT

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{img_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        raw_text = response.choices[0].message.content or ""
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        _empty["_error"] = err
        return _empty

    # Parseia JSON
    try:
        json_str = _extract_json_from_text(raw_text)
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as exc:
        _empty["_error"] = f"Falha ao parsear JSON da IA: {exc} | raw: {raw_text[:300]}"
        _empty["_raw"] = raw_text
        return _empty

    outer_poly = _validate_polygon(data.get("outer_polygon", []))
    confidence = float(data.get("confidence", data.get("confidence_outer", 0.0)))
    notes = str(data.get("notes", ""))

    return {
        "outer_polygon":    outer_poly,
        "inner_polygon":    [],   # esquema simplificado — não solicitado à IA
        "edge_samples":     [],   # esquema simplificado — não solicitado à IA
        "confidence":       confidence,
        "confidence_outer": confidence,  # alias para compatibilidade
        "notes":            notes,
        "_raw":             raw_text,
        "_error":           "",
    }


def openai_configured() -> bool:
    """Verdadeiro se OPENAI_API_KEY estiver definida e openai instalado."""
    return _OPENAI_SDK_AVAILABLE and bool(os.getenv("OPENAI_API_KEY", "").strip())


def openai_error_message() -> str:
    """Retorna mensagem de erro descritiva se a OpenAI não estiver configurada."""
    if not _OPENAI_SDK_AVAILABLE:
        return (
            "Biblioteca `openai` não instalada. "
            "Execute no terminal do ambiente correto:\n"
            "`pip install openai python-dotenv`"
        )
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return (
            "OPENAI_API_KEY não encontrada. "
            "Configure o arquivo `.env` na raiz do projeto com:\n"
            "```\nOPENAI_API_KEY=sua_chave\nDEPLOYMENT=gpt-4o-mini-petrobras\nAPI_VERSION=2024-06-01\n```"
        )
    return ""


def test_api_connection() -> tuple[bool, str]:
    """
    Faz uma chamada mínima de chat completions.
    Retorna (True, "") em sucesso ou (False, mensagem_de_erro) em falha.
    """
    if not _OPENAI_SDK_AVAILABLE:
        return False, "Biblioteca 'openai' não instalada."
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return False, "OPENAI_API_KEY não configurada nos Secrets do Streamlit Cloud."
    try:
        client = _get_client()
        dep = os.getenv("DEPLOYMENT") or os.getenv("MODEL", "gpt-4o-mini")
        client.chat.completions.create(
            model=dep,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
