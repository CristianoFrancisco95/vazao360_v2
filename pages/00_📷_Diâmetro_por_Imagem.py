"""
pages/00_📷_Diâmetro_por_Imagem.py
====================================
Pipeline de estimativa de diâmetro com 2 modos:
  1. Polígono manual — usuário clica ~10 pontos ao redor da borda do furo
  2. IA (OpenAI Vision) — segmentação automática

Em ambos os modos, o OpenCV mede a área e converte para mm via ArUco.
"""
import base64
import io

import streamlit as st
from PIL import Image, ImageDraw

from app.ui import inject_global_css, sidebar_brand
from app.auth import require_login

st.set_page_config(
    page_title="Vazão360 — Diâmetro do Furo",
    page_icon="📷",
    layout="wide",
)
require_login()
inject_global_css()
sidebar_brand()

st.title("01 - Estimativa do Diâmetro do Furo por Imagem")
st.caption(
    "**Modo de Uso:** clique ~10 pontos ao redor da borda do furo → OpenCV mede a área.  "
    
)

# ── Imports do pipeline ────────────────────────────────────────────────────────
try:
    from app.image.diameter import process_uploaded_image_openai
    from app.integrations.openai_vision import openai_configured, openai_error_message
    _PIPELINE_OK = True
    _PIPELINE_ERROR = ""
except Exception as _e_pipe:
    _PIPELINE_OK = False
    _PIPELINE_ERROR = str(_e_pipe)

if not _PIPELINE_OK:
    st.error(
        f"Pipeline não disponível: `{_PIPELINE_ERROR}`\n\n"
        "Verifique se **opencv-contrib-python-headless** e **Pillow** estão instalados."
    )
    st.stop()

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    _CLICK_OK = True
except ImportError:
    _CLICK_OK = False


# ── Helper: desenha pontos e polígono sobre a imagem ──────────────────────────
def _draw_polygon_preview(img_bytes: bytes, pts_norm: list) -> bytes:
    """Retorna JPEG com o polígono parcial desenhado (pontos numerados)."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    iw, ih = img.size
    draw = ImageDraw.Draw(img)
    if len(pts_norm) >= 1:
        coords = [(int(p["x"] * iw), int(p["y"] * ih)) for p in pts_norm]
        if len(pts_norm) >= 3:
            draw.polygon(coords, outline=(255, 220, 0), width=3)
        elif len(pts_norm) == 2:
            draw.line(coords, fill=(255, 220, 0), width=3)
        r_pt = max(5, min(10, iw // 100))
        for i, (cx, cy) in enumerate(coords):
            draw.ellipse([cx - r_pt, cy - r_pt, cx + r_pt, cy + r_pt],
                         fill=(255, 80, 0), outline=(255, 255, 255), width=2)
            draw.text((cx + r_pt + 3, cy - r_pt), str(i + 1), fill=(255, 240, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ── Helper: gera PDF com marcadores ArUco em tamanho real ──────────────────
def _generate_aruco_pdf(sizes_mm: list[float] | None = None) -> bytes:
    """
    Gera um PDF de várias páginas (A4, 300 DPI) com marcadores ArUco ID=0
    em tamanho real.  Cada página contém um marcador centralizado com legenda.
    """
    from app.core.vision.hole_detector import generate_aruco_png
    from PIL import ImageFont

    if sizes_mm is None:
        sizes_mm = [5.0, 10.0, 25.0, 50.0, 100.0]

    DPI = 300
    MM_PER_INCH = 25.4
    # A4 em pixels a 300 DPI
    PAGE_W = round(210 / MM_PER_INCH * DPI)   # 2480
    PAGE_H = round(297 / MM_PER_INCH * DPI)   # 3508
    MARGIN = round(15 / MM_PER_INCH * DPI)    # 15 mm de margem

    # Tenta fonte escalonável; senão usa default
    def _font(size: int):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            try:
                return ImageFont.load_default(size=size)
            except Exception:
                return ImageFont.load_default()

    pages: list[Image.Image] = []

    for size_mm in sizes_mm:
        # Pixels necessários para o marcador (inclui borda branca interna)
        # Queremos que a marca preta mede size_mm.
        # generate_aruco_png: inner = size_px - 80, borda de 40 px de cada lado.
        # Logo: inner_px = round(size_mm / MM_PER_INCH * DPI)
        #       size_px   = inner_px + 80
        inner_px    = max(60, round(size_mm / MM_PER_INCH * DPI))
        gen_size_px = inner_px + 80

        # Gera e reescala para garantir a resolução correta
        raw = generate_aruco_png(marker_id=0, dict_name="DICT_4X4_50", size_px=gen_size_px)
        marker_img = Image.open(io.BytesIO(raw)).convert("RGB")
        # Reescala para exatamente gen_size_px (deve já estar certo, mas garante)
        marker_img = marker_img.resize((gen_size_px, gen_size_px), Image.LANCZOS)

        # Página A4 branca
        page = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
        draw = ImageDraw.Draw(page)

        # Centraliza verticalmente deixando espaço para texto
        text_reserve = round(25 / MM_PER_INCH * DPI)   # 25 mm para texto
        avail_h = PAGE_H - 2 * MARGIN - text_reserve
        avail_w = PAGE_W - 2 * MARGIN

        # Se o marcador for maior que a área disponível, reduz (caso do 100 mm)
        if gen_size_px > avail_w or gen_size_px > avail_h:
            scale = min(avail_w / gen_size_px, avail_h / gen_size_px)
            new_sz = max(30, round(gen_size_px * scale))
            marker_img = marker_img.resize((new_sz, new_sz), Image.LANCZOS)
            gen_size_px = new_sz

        x = (PAGE_W - gen_size_px) // 2
        y = MARGIN + (avail_h - gen_size_px) // 2
        page.paste(marker_img, (x, y))

        # Título
        title     = f"Marcador ArUco — {size_mm:g} mm × {size_mm:g} mm (tamanho real)"
        subtitle  = "ArUco DICT_4X4_50, ID = 0  |  Imprima sem redimensionar (100 %)"
        note      = "Verificar: o lado do quadrado preto deve medir exatamente a dimensão indicada."

        fsize_title = round(14 / MM_PER_INCH * DPI * 0.35)   # ~14 pt
        fsize_sub   = round(10 / MM_PER_INCH * DPI * 0.35)   # ~10 pt
        ft = _font(fsize_title)
        fs = _font(fsize_sub)

        text_y = y + gen_size_px + round(5 / MM_PER_INCH * DPI)
        draw.text((PAGE_W // 2, text_y),            title,    fill=(0, 0, 0),       font=ft, anchor="mt")
        draw.text((PAGE_W // 2, text_y + fsize_title + 10), subtitle, fill=(60, 60, 60), font=fs, anchor="mt")
        draw.text((PAGE_W // 2, text_y + fsize_title + fsize_sub + 24), note, fill=(120, 120, 120), font=fs, anchor="mt")

        # Linha de rodapé
        footer = f"Vazão360 — Ferramenta de calibração ArUco  |  {size_mm:g} mm"
        draw.text((PAGE_W // 2, PAGE_H - MARGIN // 2), footer, fill=(160, 160, 160), font=fs, anchor="mb")

        pages.append(page)

    buf = io.BytesIO()
    pages[0].save(
        buf,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=DPI,
    )
    return buf.getvalue()


# ── Configurações ──────────────────────────────────────────────────────────────
with st.expander("⚙️ Configurações", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        _mkr_opts = {"1 mm": 1.0, "5 mm": 5.0, "10 mm": 10.0,
                     "25 mm": 25.0, "50 mm": 50.0, "100 mm": 100.0}
        _mkr_label = st.selectbox(
            "Tamanho real do marcador ArUco",
            options=list(_mkr_opts.keys()),
            index=4,
            key="dbg_marker_size",
        )
        marker_mm = _mkr_opts[_mkr_label]
    with col_b:
        st.metric("Escala esperada se ArUco detectado", f"{marker_mm} mm / lado")
    with col_c:
        st.info(
            "Posicione o marcador ArUco impresso ao lado do furo, "
            "no mesmo plano e perpendicular à câmera."
        )
    # ── Botão download PDF ─────────────────────────────────────────────
    st.markdown("")
    try:
        _aruco_pdf_bytes = _generate_aruco_pdf([5.0, 10.0, 25.0, 50.0, 100.0])
        st.download_button(
            label="📄 Baixar marcadores ArUco em PDF (5 / 10 / 25 / 50 / 100 mm — tamanho real)",
            data=_aruco_pdf_bytes,
            file_name="aruco_marcadores_tamanho_real.pdf",
            mime="application/pdf",
            help="PDF com 5 páginas A4 a 300 DPI. Imprima em 100 % (sem ajuste de página) para obter o tamanho físico correto.",
        )
    except Exception as _e_pdf:
        st.warning(f"Não foi possível gerar o PDF de marcadores: {_e_pdf}")

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📤 Upload da imagem do furo",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    key="dbg_hole_image",
)

# ── Session state ──────────────────────────────────────────────────────────────
_click_pts: list = st.session_state.get("_click_pts", [])
_dbg_result = st.session_state.get("_dbg_result", None)

if uploaded is not None:
    img_bytes = uploaded.read()
    _cur_img_hash = hash(img_bytes)

    # Limpa pontos e resultado se imagem mudou
    if st.session_state.get("_click_img_hash") != _cur_img_hash:
        st.session_state.update({
            "_click_pts": [],
            "_click_last_raw": None,
            "_click_img_hash": _cur_img_hash,
        })
        st.session_state.pop("_dbg_result", None)
        st.session_state.pop("_dbg_img_hash", None)
        _click_pts = []
        _dbg_result = None
        st.rerun()

    pil_img = Image.open(io.BytesIO(img_bytes))
    img_w, img_h = pil_img.size

    # ── Interface de clique ────────────────────────────────────────────────────
    st.markdown("---")
    n_pts = len(_click_pts)
    _hint = "✅ Pronto para processar!" if n_pts >= 3 else f"Adicione mais {3 - n_pts} ponto(s) para habilitar o processamento."
    st.subheader(f"📍 Clique na borda do furo  —  {n_pts} ponto(s) marcado(s)")
    st.caption(
        f"Clique ~10 pontos ao redor da **borda física** do furo (não só a área escura interna).  "
        f"{_hint}"
    )

    # Preview com polígono desenhado — também serve de canvas clicável
    preview_bytes = _draw_polygon_preview(img_bytes, _click_pts)

    if _CLICK_OK:
        coord = streamlit_image_coordinates(Image.open(io.BytesIO(preview_bytes)), key="click_map_dbg")
        if coord is not None:
            last_raw = st.session_state.get("_click_last_raw")
            if coord != last_raw:
                st.session_state["_click_last_raw"] = coord
                new_pts = list(_click_pts) + [{
                    "x": round(coord["x"] / img_w, 6),
                    "y": round(coord["y"] / img_h, 6),
                }]
                st.session_state["_click_pts"] = new_pts
                st.rerun()
    else:
        st.image(preview_bytes, use_container_width=True)
        st.warning(
            "`streamlit-image-coordinates` não instalado. "
            "Execute: `pip install streamlit-image-coordinates`"
        )

    # ── Controles ──────────────────────────────────────────────────────────────
    col_undo, col_clear, col_manual, col_ai = st.columns([1, 1, 2, 2])

    with col_undo:
        if st.button("↩️ Desfazer", disabled=(n_pts == 0), key="dbg_undo"):
            st.session_state["_click_pts"] = _click_pts[:-1]
            st.session_state["_click_last_raw"] = None
            st.rerun()

    with col_clear:
        if st.button("🗑️ Limpar", disabled=(n_pts == 0), key="dbg_clear"):
            st.session_state["_click_pts"] = []
            st.session_state["_click_last_raw"] = None
            st.session_state.pop("_dbg_result", None)
            st.rerun()

    with col_manual:
        _can_process = n_pts >= 3
        if st.button(
            f"📐 Processar área selecionada ({n_pts} pts)" if _can_process
            else f"📐 Manual — marque ≥3 pts ({n_pts}/3)",
            disabled=not _can_process,
            type="primary",
            key="dbg_process_manual",
        ):
            with st.spinner("OpenCV medindo área e diâmetro…"):
                _dbg_result = process_uploaded_image_openai(
                    img_bytes,
                    marker_size_mm=marker_mm,
                    manual_polygon=list(_click_pts),
                )
            st.session_state["_dbg_result"] = _dbg_result
            st.rerun()

    with col_ai:
        # Verifica se a API está acessível (resultado cacheado na sessão)
        _ai_status   = st.session_state.get("_api_test_result")
        _ai_msg      = st.session_state.get("_api_test_msg", "")
        _ai_net_err  = "APIConnectionError" in _ai_msg or "Connection error" in _ai_msg
        _ai_no_key   = not openai_configured()
        _ai_blocked  = _ai_net_err or (_ai_status == "fail" and not _ai_net_err)

        if _ai_no_key:
            st.button(
                "🤖 IA — API não configurada",
                disabled=True,
                key="dbg_process_ai",
                help="Configure OPENAI_API_KEY nos Secrets do Streamlit Cloud.",
            )
        elif _ai_net_err:
            st.button(
                "🤖 IA — indisponível fora da rede Petrobras",
                disabled=True,
                key="dbg_process_ai",
                help=(
                    "O gateway apit.petrobras.com.br só é acessível dentro da rede "
                    "corporativa Petrobras. Use o modo Manual (📐) ou acesse o sistema "
                    "por um servidor interno Petrobras."
                ),
            )
            st.caption(
                "ℹ️ Modo IA disponível apenas na rede Petrobras.",
            )
        else:
            if st.button(
                "🤖 Processar com IA (OpenAI Vision)",
                disabled=False,
                type="secondary",
                key="dbg_process_ai",
                help="Envia a imagem para a IA segmentar o furo automaticamente.",
            ):
                with st.spinner("IA analisando a imagem…"):
                    _dbg_result = process_uploaded_image_openai(
                        img_bytes,
                        marker_size_mm=marker_mm,
                        manual_polygon=None,
                    )
                st.session_state["_dbg_result"] = _dbg_result
                st.rerun()



# ── Resultado ──────────────────────────────────────────────────────────────────
if _dbg_result is not None:
    st.markdown("---")
    if _dbg_result.get("error_message"):
        st.error(f"❌ {_dbg_result['error_message']}")
    if _dbg_result.get("ai_error"):
        st.warning(f"⚠️ {_dbg_result['ai_error']}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        d_mm = _dbg_result.get("diameter_mm")
        st.metric("Diâmetro estimado (mm)", f"{d_mm:.3f}" if d_mm else "—")
    with col2:
        unc = _dbg_result.get("uncertainty_pct")
        st.metric("Incerteza", f"±{unc:.1f}%" if unc else "—")
    with col3:
        conf = _dbg_result.get("confidence")
        st.metric("Confiança", f"{conf * 100:.0f}%" if conf else "—")
    with col4:
        st.metric("Método", _dbg_result.get("method", "—"))

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Área (px²)", f"{_dbg_result.get('area_px', 0):,}")
    with col6:
        a_mm2 = _dbg_result.get("area_mm2")
        st.metric("Área (mm²)", f"{a_mm2:.2f}" if a_mm2 else "—")
    with col7:
        mpp = _dbg_result.get("mm_per_pixel")
        st.metric("mm/pixel", f"{mpp:.5f}" if mpp else "—")
    with col8:
        st.metric("ArUco detectado?", "✅ Sim" if _dbg_result.get("aruco_detected") else "❌ Não")

    col_ov, col_mk = st.columns(2)
    with col_ov:
        if _dbg_result.get("overlay_b64"):
            st.image(
                base64.b64decode(_dbg_result["overlay_b64"]),
                caption="Overlay — máscara + polígono",
                use_container_width=True,
            )
        else:
            st.info("Overlay não disponível.")
    with col_mk:
        if _dbg_result.get("mask_preview_b64"):
            st.image(
                base64.b64decode(_dbg_result["mask_preview_b64"]),
                caption="Máscara binária",
                use_container_width=True,
            )
        else:
            st.info("Máscara não disponível.")

    with st.expander("🔍 Detalhes"):
        st.write(f"**Confiança:** {_dbg_result.get('confidence', 0):.2f}")
        st.write(f"**Notas:** {_dbg_result.get('ai_notes', '—')}")
        if _dbg_result.get("ai_error"):
            st.warning(_dbg_result["ai_error"])
        if _dbg_result.get("ai_raw"):
            st.markdown("**Resposta bruta da IA:**")
            st.code(_dbg_result["ai_raw"], language="json")

    d_min = _dbg_result.get("d_min_mm")
    d_max = _dbg_result.get("d_max_mm")
    if d_min and d_max:
        st.info(
            f"Intervalo de incerteza: **{d_min:.2f} mm** — **{d_max:.2f} mm**  "
            f"(erosão/dilatação ±2 px)"
        )

    st.caption(
        f"Arquivo: {_dbg_result.get('filename')}  |  "
        f"Timestamp: {_dbg_result.get('timestamp')}  |  "
        f"Overlay: {_dbg_result.get('overlay_path', '—')}"
    )

    st.markdown("---")
    if _dbg_result.get("success") and _dbg_result.get("diameter_mm"):
        col_use, col_reset = st.columns([3, 2])
        with col_use:
            if st.button(
                f"✅ Usar este diâmetro ({_dbg_result['diameter_mm']:.2f} mm) na Calculadora",
                type="primary",
            ):
                st.session_state["image_diameter_result_latest"] = _dbg_result
                st.session_state["orifice_image_meta"] = {
                    "diameter_mm":           _dbg_result["diameter_mm"],
                    "diameter_mm_confirmed": _dbg_result["diameter_mm"],
                    "uncertainty_pct":       _dbg_result["uncertainty_pct"],
                    "mm_per_pixel":          _dbg_result["mm_per_pixel"],
                    "method":                _dbg_result["method"],
                    "confidence":            _dbg_result["confidence"],
                    "aruco_detected":        _dbg_result["aruco_detected"],
                    "aruco_ids":             [],
                    "marker_real_mm":        marker_mm,
                    "processing_time_s":     _dbg_result.get("processing_time_s", 0),
                }
                st.success(
                    "Resultado salvo. Vá à **02 — Entrada de Dados** e confirme o "
                    "diâmetro estimado, ou prossiga direto para **03 — Calculadora**."
                )
        with col_reset:
            if st.button(
                "🗑️ Resetar Cálculo de Diâmetro do Furo",
                key="btn_reset_diameter",
                help="Apaga o resultado atual e permite iniciar um novo cálculo.",
            ):
                for _k in (
                    "_dbg_result", "_click_pts", "_click_last_raw",
                    "_click_img_hash", "dbg_hole_image",
                    "orifice_image_meta", "image_diameter_result_latest",
                ):
                    st.session_state.pop(_k, None)
                st.rerun()
