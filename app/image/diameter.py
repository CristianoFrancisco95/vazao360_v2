"""
app/image/diameter.py
======================
Pipeline em 2 estágios para estimativa de diâmetro de furo via imagem.

Estágio 1 — OpenAI Vision:
    A IA retorna um polígono do contorno do furo (pontos normalizados 0..1).
    O polígono é convertido em máscara binária com cv2.fillPoly.

Estágio 2 — OpenCV (medição):
    • ArUco → mm_per_pixel
    • Área da máscara em pixels → diâmetro equivalente
    • d_eq_mm = 2 * sqrt(area_mm2 / π)
    • Incerteza por erosão/dilatação leve

Fallback automático:
    Se a IA não retornar polígono válido, ativa o pipeline OpenCV clássico
    (HoughCircles → Canny+fitEllipse), registrando method="fallback_contour".

API pública principal:
    process_uploaded_image_openai(image_bytes, marker_size_mm) -> dict

Funções auxiliares reutilizáveis:
    detect_aruco_mm_per_px
    build_mask_from_openai
    diameter_from_mask
    make_overlay
    save_overlay_and_b64
"""
from __future__ import annotations

import base64
import io
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image as _PILImage

from app.integrations.openai_vision import (
    segment_hole_polygon_openai,
    openai_configured,
    openai_error_message,
)


# ── Constantes ────────────────────────────────────────────────────────────────
_OVERLAY_DIR = Path(__file__).resolve().parents[2] / "app" / "static" / "overlays"
_MAX_DIM = 1600          # redimensionamento máximo (px)
_MORPH_KERNEL = 5         # kernel morfológico (px)

# Limites de sanidade para a máscara retornada pela IA.
# Furos são vazios físicos → regiões escuras. Se a região marcada for clara
# ou absurdamente grande, a IA aluciou e o resultado é descartado.
_MASK_BRIGHTNESS_MAX = 130  # média de cinza dentro da máscara; acima → não é furo
_MASK_AREA_MAX_RATIO = 0.35 # máscara não pode cobrir mais de 35% da imagem


# ── Helpers de I/O ────────────────────────────────────────────────────────────

def _load_bgr(image_bytes: bytes) -> np.ndarray:
    """Decodifica bytes de imagem → BGR numpy array (≤ _MAX_DIM px no maior eixo)."""
    pil = _PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    w0, h0 = pil.size
    if max(w0, h0) > _MAX_DIM:
        scale = _MAX_DIM / max(w0, h0)
        pil = pil.resize((int(w0 * scale), int(h0 * scale)), _PILImage.LANCZOS)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _encode_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Falha ao codificar overlay como PNG")
    return bytes(buf)


def _bgr_to_b64(bgr: np.ndarray) -> str:
    return base64.b64encode(_encode_png_bytes(bgr)).decode("utf-8")


# ── Detecção de ArUco ─────────────────────────────────────────────────────────

_ARUCO_DICTS = [
    ("DICT_4X4_50",  cv2.aruco.DICT_4X4_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
]


def detect_aruco_mm_per_px(
    img_bgr: np.ndarray,
    marker_size_mm: float = 50.0,
) -> tuple[bool, Optional[float], list, np.ndarray]:
    """
    Detecta marcador(es) ArUco na imagem e calcula mm_per_pixel.

    Retorna
    -------
    (found, mm_per_pixel, corners_list, annotated_bgr)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    annotated = img_bgr.copy()

    for _dict_name, _dict_id in _ARUCO_DICTS:
        aruco_dict = cv2.aruco.getPredefinedDictionary(_dict_id)
        params = cv2.aruco.DetectorParameters()
        try:
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        if ids is not None and len(ids) > 0:
            try:
                cv2.aruco.drawDetectedMarkers(annotated, corners)
            except Exception:
                pass
            # Calcula mm_per_px com base na média dos 4 lados do 1.º marcador
            c = corners[0].reshape(4, 2)
            sides = [
                np.linalg.norm(c[1] - c[0]),
                np.linalg.norm(c[2] - c[1]),
                np.linalg.norm(c[3] - c[2]),
                np.linalg.norm(c[0] - c[3]),
            ]
            side_px = float(np.mean(sides))
            mm_per_pixel = marker_size_mm / side_px
            return True, mm_per_pixel, list(corners), annotated

    return False, None, [], annotated


# ── Máscara via OpenAI Vision ─────────────────────────────────────────────────

def build_mask_from_openai(
    img_bgr: np.ndarray,
    img_bytes: bytes,
) -> tuple[Optional[np.ndarray], dict]:
    """
    Chama a OpenAI Vision e converte o polígono em máscara binária (0/255).

    Retorna
    -------
    (mask_uint8, metadata_dict)
    mask_uint8 : None se falhou
    metadata_dict : {
        "polygon_pts": list,
        "confidence": float,
        "notes": str,
        "ai_error": str,
        "ai_raw": str,
    }
    """
    h, w = img_bgr.shape[:2]
    ai_result = segment_hole_polygon_openai(img_bytes)

    # Usa outer_polygon (contorno físico completo) como primário;
    # cai para inner_polygon apenas se outer não tiver pontos suficientes.
    outer_poly = ai_result.get("outer_polygon", [])
    inner_poly  = ai_result.get("inner_polygon", [])
    polygon = outer_poly if len(outer_poly) >= 4 else inner_poly
    confidence = float(ai_result.get("confidence_outer", ai_result.get("confidence", 0.0)))
    notes = ai_result.get("notes", "")
    ai_error = ai_result.get("_error", "")

    metadata = {
        "polygon_pts":      polygon,
        "outer_polygon":    outer_poly,
        "inner_polygon":    inner_poly,
        "edge_samples":     ai_result.get("edge_samples", []),
        "confidence":       confidence,
        "confidence_inner": float(ai_result.get("confidence_inner", 0.0)),
        "notes":            notes,
        "ai_error":         ai_error,
        "ai_raw":           ai_result.get("_raw", "")[:3000],
    }

    if not polygon or len(polygon) < 4:
        return None, metadata

    # Converte pontos normalizados → pixels
    pts_px = np.array(
        [[int(pt["x"] * w), int(pt["y"] * h)] for pt in polygon],
        dtype=np.int32,
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_px], 255)

    # Pós-processamento morfológico:
    # 1) CLOSE fecha falhas internas criadas pela IA (bordas não-contínuas)
    # 2) DILATE expande para capturar bisel/chanfro que a IA subcobertura
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_MORPH_KERNEL, _MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Mantém apenas o maior componente conexo
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, np.uint8(255), np.uint8(0))

    # Preenche buracos internos (flood fill na borda)
    flood = mask.copy()
    flood_padded = cv2.copyMakeBorder(flood, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(flood_padded, None, (0, 0), 255)
    flood_inner = cv2.bitwise_not(flood_padded[1:-1, 1:-1])
    mask = cv2.bitwise_or(mask, flood_inner)

    # ── Sanidade: máscara não pode cobrir mais de 35% da imagem ─────────────
    # (a única rejeição segura — o brilho da borda do furo varia muito)
    mask_area = int(np.count_nonzero(mask))
    img_area  = h * w
    if mask_area > 0:
        area_ratio = mask_area / img_area
        if area_ratio > _MASK_AREA_MAX_RATIO:
            metadata["ai_error"] = (
                f"Máscara cobre {area_ratio:.0%} da imagem "
                f"(limite={_MASK_AREA_MAX_RATIO:.0%}) — alucinação da IA."
            )
            return None, metadata

    metadata["polygon_pts"] = pts_px.tolist()
    return mask, metadata

# ── Máscara a partir de polígono manual (cliques do usuário) ────────────────────────

def build_mask_from_click_points(
    img_bgr: np.ndarray,
    pts_normalized: list,
) -> tuple[Optional[np.ndarray], dict]:
    """
    Constrói máscara binária a partir de pontos clicados pelo usuário.

    pts_normalized : lista de {"x": 0..1, "y": 0..1}
    Retorna (mask_uint8, metadata_dict) — mesma interface de build_mask_from_openai.
    """
    h, w = img_bgr.shape[:2]
    metadata = {
        "polygon_pts":  [],
        "outer_polygon": pts_normalized,
        "inner_polygon": [],
        "confidence":    1.0,
        "notes":         f"Polígono manual ({len(pts_normalized)} pontos).",
        "ai_error":      "",
        "ai_raw":        "",
    }

    if not pts_normalized or len(pts_normalized) < 3:
        metadata["ai_error"] = "Polígono manual requer pelo menos 3 pontos."
        return None, metadata

    pts_px = np.array(
        [[int(pt["x"] * w), int(pt["y"] * h)] for pt in pts_normalized],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_px], 255)

    # Fecha buracos e suaviza bordas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_MORPH_KERNEL, _MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Preenche buracos internos (flood fill na borda)
    flood = mask.copy()
    flood_padded = cv2.copyMakeBorder(flood, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(flood_padded, None, (0, 0), 255)
    flood_inner = cv2.bitwise_not(flood_padded[1:-1, 1:-1])
    mask = cv2.bitwise_or(mask, flood_inner)

    metadata["polygon_pts"] = pts_px.tolist()
    return mask, metadata

# ── Diâmetro a partir da máscara ──────────────────────────────────────────────

def diameter_from_mask(
    mask: np.ndarray,
    mm_per_pixel: Optional[float],
) -> dict:
    """
    Calcula diâmetro equivalente de círculo a partir da máscara binária.

    Retorna
    -------
    dict com:
        d_eq_mm, d_eq_px, area_px, area_mm2, r_eq_px, cx, cy,
        d_min_mm, d_max_mm, uncertainty_mm, uncertainty_pct
    """
    area_px = int(np.count_nonzero(mask))
    r_eq_px = math.sqrt(area_px / math.pi) if area_px > 0 else 0.0
    d_eq_px = 2.0 * r_eq_px

    # Centro de massa
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        h, w = mask.shape[:2]
        cx, cy = w // 2, h // 2

    # Conversão para mm
    area_mm2: Optional[float] = None
    d_eq_mm: Optional[float] = None
    d_min_mm: Optional[float] = None
    d_max_mm: Optional[float] = None
    uncertainty_mm: Optional[float] = None
    uncertainty_pct: Optional[float] = None

    if mm_per_pixel is not None and mm_per_pixel > 0:
        area_mm2 = area_px * (mm_per_pixel ** 2)
        d_eq_mm = 2.0 * math.sqrt(area_mm2 / math.pi)

        # Incerteza: erosão/dilatação ±2 px
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_ero = cv2.erode(mask, kernel2, iterations=2)
        mask_dil = cv2.dilate(mask, kernel2, iterations=2)
        a_min = int(np.count_nonzero(mask_ero))
        a_max = int(np.count_nonzero(mask_dil))
        d_min_mm = 2.0 * math.sqrt(a_min * (mm_per_pixel ** 2) / math.pi) if a_min > 0 else d_eq_mm
        d_max_mm = 2.0 * math.sqrt(a_max * (mm_per_pixel ** 2) / math.pi) if a_max > 0 else d_eq_mm
        uncertainty_mm = (d_max_mm - d_min_mm) / 2.0
        uncertainty_pct = (uncertainty_mm / d_eq_mm * 100.0) if d_eq_mm > 0 else None

    return {
        "d_eq_mm":       d_eq_mm,
        "d_eq_px":       d_eq_px,
        "area_px":       area_px,
        "area_mm2":      area_mm2,
        "r_eq_px":       r_eq_px,
        "cx":            cx,
        "cy":            cy,
        "d_min_mm":      d_min_mm,
        "d_max_mm":      d_max_mm,
        "uncertainty_mm":  uncertainty_mm,
        "uncertainty_pct": uncertainty_pct,
    }


# ── Overlay visual ────────────────────────────────────────────────────────────

def make_overlay(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    cx: int,
    cy: int,
    r_eq_px: float,
    d_eq_mm: Optional[float],
    uncertainty_pct: Optional[float],
    method: str,
    extra_text: str = "",
    outer_pts: Optional[list] = None,
    inner_pts: Optional[list] = None,
) -> np.ndarray:
    """
    Cria overlay visual sobre a imagem original com:
      - Preenchimento semi-transparente na máscara (ciano-verde)
      - Contorno externo da máscara (ciano vivo, espesso)
      - outer_polygon da IA (branco+verde)
      - inner_polygon da IA (laranja)
      - Legenda de cores
      - Texto com diâmetro e incerteza

    Retorna imagem BGR anotada.
    """
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    # ── 1. Preenchimento semi-transparente da máscara ─────────────────────
    if mask is not None:
        color_layer = np.zeros_like(out)
        color_layer[mask > 0] = (200, 230, 0)   # ciano-esverdeado vivido
        cv2.addWeighted(color_layer, 0.45, out, 0.55, 0, out)
        # Contorno externo da máscara — linha grossa ciano
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 230, 200), 3)

    # ── 2. outer_polygon da IA (branco por baixo + verde por cima) ────────
    if outer_pts and len(outer_pts) >= 4:
        pts = np.array(
            [[int(p["x"] * w), int(p["y"] * h)] for p in outer_pts], dtype=np.int32
        )
        cv2.polylines(out, [pts], isClosed=True, color=(255, 255, 255), thickness=4)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 80), thickness=2)

    # ── 3. inner_polygon da IA (laranja) ──────────────────────────────────
    if inner_pts and len(inner_pts) >= 4:
        pts_i = np.array(
            [[int(p["x"] * w), int(p["y"] * h)] for p in inner_pts], dtype=np.int32
        )
        cv2.polylines(out, [pts_i], isClosed=True, color=(0, 140, 255), thickness=2)

    # ── 4. Legenda de cores ───────────────────────────────────────────────
    legend = [
        ("Area detectada (mas.)", (200, 230,   0)),
        ("Borda outer (IA)",      (  0, 255,  80)),
        ("Borda inner (IA)",      (  0, 140, 255)),
    ]
    _lx, _ly = w - 10, h - 10
    _fs_leg, _lh_leg = 0.45, 18
    _box_w_leg = 220
    _box_h_leg = _lh_leg * len(legend) + 8
    _lx0 = _lx - _box_w_leg
    cv2.rectangle(out, (_lx0, _ly - _box_h_leg), (_lx, _ly), (20, 20, 20), -1)
    cv2.rectangle(out, (_lx0, _ly - _box_h_leg), (_lx, _ly), (120, 120, 120), 1)
    for i, (label, color) in enumerate(legend):
        _y = _ly - _box_h_leg + 4 + _lh_leg * (i + 1) - 3
        cv2.rectangle(out, (_lx0 + 4, _y - 9), (_lx0 + 16, _y + 2), color, -1)
        cv2.putText(out, label, (_lx0 + 20, _y), cv2.FONT_HERSHEY_SIMPLEX, _fs_leg, (220, 220, 220), 1)

    # ── 6. Texto de métricas ──────────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines: list[str] = []
    if d_eq_mm is not None:
        unc_str = f" ±{uncertainty_pct:.0f}%" if uncertainty_pct is not None else ""
        lines.append(f"d = {d_eq_mm:.2f} mm{unc_str}")
    else:
        lines.append(f"d = {r_eq_px * 2:.0f} px  (sem escala)")
    lines.append(f"Metodo: {method}")
    if extra_text:
        lines.append(extra_text)

    _fs, _th = 0.85, 2
    _pad = 8
    _lh = 30
    _box_h = _lh * len(lines) + _pad * 2
    _box_w = max(cv2.getTextSize(l, font, _fs, _th)[0][0] for l in lines) + _pad * 2
    cv2.rectangle(out, (8, 8), (8 + _box_w, 8 + _box_h), (30, 30, 30), -1)
    cv2.rectangle(out, (8, 8), (8 + _box_w, 8 + _box_h), (200, 200, 200), 1)
    for i, line in enumerate(lines):
        cy_text = 8 + _pad + _lh * (i + 1) - 6
        cv2.putText(out, line, (12, cy_text), font, _fs, (0, 230, 255), _th)

    return out


# ── Salvar overlay ────────────────────────────────────────────────────────────

def save_overlay_and_b64(
    overlay_bgr: np.ndarray,
    basename: str = "overlay",
) -> tuple[str, str]:
    """
    Salva overlay em disco (app/static/overlays/) e retorna (path_str, base64_str).
    """
    _OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{basename}_{ts}.png"
    fpath = _OVERLAY_DIR / fname
    ok, buf = cv2.imencode(".png", overlay_bgr)
    if ok:
        fpath.write_bytes(bytes(buf))
    overlay_b64 = base64.b64encode(_encode_png_bytes(overlay_bgr)).decode("utf-8")
    return str(fpath), overlay_b64


# ── Fallback OpenCV clássico ─────────────────────────────────────────────────

def _fallback_opencv(
    img_bgr: np.ndarray,
) -> tuple[Optional[np.ndarray], str, float]:
    """
    Pipeline OpenCV clássico (HoughCircles → Canny+fitEllipse).
    Retorna (mask, method, confidence).
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    min_dim = min(h, w)

    # Tenta HoughCircles
    circles = cv2.HoughCircles(
        enhanced, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(int(min_dim * 0.06), 10),
        param1=80.0, param2=28.0,
        minRadius=int(min_dim * 0.01),
        maxRadius=int(min_dim * 0.45),
    )

    if circles is not None:
        best = np.uint16(np.around(circles))[0][0]
        cx, cy, r = int(best[0]), int(best[1]), int(best[2])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        return mask, "fallback_HoughCircles", 0.55

    # Tenta Canny + fitEllipse
    edges = cv2.Canny(enhanced, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue
        perim = cv2.arcLength(cnt, True)
        circ = 4 * math.pi * area / (perim ** 2) if perim > 0 else 0
        if circ > 0.20 and area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is not None:
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(best_cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(best_cnt)
                cv2.ellipse(mask, ellipse, 255, -1)
                return mask, "fallback_Canny+fitEllipse", 0.40
            except cv2.error:
                pass
        cv2.drawContours(mask, [best_cnt], -1, 255, -1)
        return mask, "fallback_Canny+contour", 0.30

    return None, "fallback_no_detection", 0.0


# ── Pipeline principal ────────────────────────────────────────────────────────

def process_uploaded_image_openai(
    image_input,
    marker_size_mm: float = 50.0,
    manual_polygon: Optional[list] = None,
) -> dict:
    """
    Pipeline completo 2 estágios para estimativa de diâmetro via imagem.

    Parâmetros
    ----------
    image_input   : bytes ou file-like object com a imagem
    marker_size_mm: tamanho real do lado do marcador ArUco em mm

    Retorna
    -------
    dict compatível com image_diameter_result_latest no session_state:
    {
        "filename":             str,
        "method":               str,
        "diameter_mm":          float | None,
        "diameter_px_equiv":    float,
        "area_px":              int,
        "area_mm2":             float | None,
        "mm_per_pixel":         float | None,
        "aruco_detected":       bool,
        "aruco_marker_size_mm": float,
        "confidence":           float | None,
        "uncertainty_mm":       float | None,
        "uncertainty_pct":      float | None,
        "d_min_mm":             float | None,
        "d_max_mm":             float | None,
        "timestamp":            str,
        "overlay_path":         str,
        "overlay_b64":          str,
        "mask_preview_b64":     str,
        "ai_confidence":        float,
        "ai_notes":             str,
        "ai_error":             str,
        "success":              bool,
        "error_message":        str,
    }
    """
    t0 = time.time()
    ts_str = datetime.now().isoformat()
    result_base = {
        "filename": getattr(image_input, "name", "imagem") if not isinstance(image_input, (bytes, bytearray)) else "imagem",
        "method": "",
        "diameter_mm": None,
        "diameter_px_equiv": 0.0,
        "area_px": 0,
        "area_mm2": None,
        "mm_per_pixel": None,
        "aruco_detected": False,
        "aruco_marker_size_mm": marker_size_mm,
        "confidence": None,
        "uncertainty_mm": None,
        "uncertainty_pct": None,
        "d_min_mm": None,
        "d_max_mm": None,
        "timestamp": ts_str,
        "overlay_path": "",
        "overlay_b64": "",
        "mask_preview_b64": "",
        "ai_confidence": 0.0,
        "ai_notes": "",
        "ai_error": "",
        "ai_raw": "",
        "success": False,
        "error_message": "",
    }

    # 1. Lê bytes
    try:
        if isinstance(image_input, (bytes, bytearray)):
            img_bytes = bytes(image_input)
        else:
            img_bytes = image_input.read()
    except Exception as exc:
        result_base["error_message"] = f"Falha ao ler imagem: {exc}"
        return result_base

    # 2. Carrega BGR
    try:
        img_bgr = _load_bgr(img_bytes)
    except Exception as exc:
        result_base["error_message"] = f"Falha ao decodificar imagem: {exc}"
        return result_base

    h, w = img_bgr.shape[:2]

    # 3. Detecta ArUco
    aruco_found, mm_per_px, _aruco_corners, img_bgr_aruco = detect_aruco_mm_per_px(
        img_bgr, marker_size_mm
    )
    result_base["aruco_detected"] = aruco_found
    result_base["mm_per_pixel"] = mm_per_px
    # Usa imagem com anotação ArUco como base para overlay
    working_bgr = img_bgr_aruco.copy()

    # 4. Determina máscara: polígono manual (prioridade) > OpenAI Vision > fallback OpenCV
    mask: Optional[np.ndarray] = None
    method = ""
    ai_confidence = 0.0
    ai_notes = ""
    ai_error = ""
    ai_meta: dict = {}

    if manual_polygon and len(manual_polygon) >= 3:
        # Modo manual: usa os pontos clicados pelo usuário, sem chamar a IA
        mask, ai_meta = build_mask_from_click_points(img_bgr, manual_polygon)
        ai_confidence = 1.0
        ai_notes = ai_meta.get("notes", "")
        ai_error = ai_meta.get("ai_error", "")
        if mask is not None and np.count_nonzero(mask) > 100:
            method = "manual_polygon_area_equiv"
        else:
            mask = None
    elif openai_configured():
        mask, ai_meta = build_mask_from_openai(img_bgr, img_bytes)
        ai_confidence = ai_meta.get("confidence", 0.0)
        ai_notes = ai_meta.get("notes", "")
        ai_error = ai_meta.get("ai_error", "")
        if mask is not None and np.count_nonzero(mask) > 100:
            method = "openai_polygon_mask_area_equiv"
        else:
            mask = None
            ai_error = ai_error or "IA não retornou polígono válido — usando fallback OpenCV"
    else:
        ai_error = openai_error_message() or "OpenAI não configurada"

    # 5. Fallback OpenCV se necessário
    fallback_confidence = 0.0
    if mask is None:
        mask_fb, method_fb, fallback_confidence = _fallback_opencv(img_bgr)
        mask = mask_fb
        method = method_fb

    result_base["ai_confidence"] = ai_confidence
    result_base["ai_notes"] = ai_notes
    result_base["ai_error"] = ai_error
    result_base["ai_raw"] = ai_meta.get("ai_raw", "")
    result_base["method"] = method

    if mask is None or np.count_nonzero(mask) < 4:
        result_base["error_message"] = (
            "Nenhum furo detectado. "
            "Verifique: contraste da imagem, presença do furo visível, iluminação."
        )
        # Overlay mesmo sem detecção
        overlay = working_bgr.copy()
        cv2.putText(overlay, "Furo nao detectado", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        _, b64 = save_overlay_and_b64(overlay, "overlay_fail")
        result_base["overlay_b64"] = b64
        return result_base

    # 6. Calcula diâmetro
    meas = diameter_from_mask(mask, mm_per_px)
    confidence = ai_confidence if ai_confidence > 0 else fallback_confidence

    # 7. Máscara preview: original + máscara sobreposta em vermelho vivo (ajuda o usuário a ver a área)
    mask_preview = working_bgr.copy()
    if mask is not None:
        red_layer = np.zeros_like(mask_preview)
        red_layer[mask > 0] = (0, 60, 255)  # vermelho vivo
        cv2.addWeighted(red_layer, 0.55, mask_preview, 0.45, 0, mask_preview)
        # Contorno branco espesso para destacar o limite
        c_prev, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_preview, c_prev, -1, (255, 255, 255), 3)
    mask_b64 = _bgr_to_b64(mask_preview)

    # 8. Overlay com métricas + polígonos da IA
    overlay = make_overlay(
        working_bgr,
        mask,
        cx=meas["cx"],
        cy=meas["cy"],
        r_eq_px=meas["r_eq_px"],
        d_eq_mm=meas["d_eq_mm"],
        uncertainty_pct=meas["uncertainty_pct"],
        method=method,
        extra_text=f"ArUco: {'OK' if aruco_found else 'N/D'}  {mm_per_px:.4f} mm/px" if mm_per_px else "",
        outer_pts=ai_meta.get("outer_polygon", []) if ai_meta else [],
        inner_pts=ai_meta.get("inner_polygon", []) if ai_meta else [],
    )
    overlay_path, overlay_b64 = save_overlay_and_b64(overlay, "overlay")

    elapsed = time.time() - t0
    result_base.update({
        "method":            method,
        "diameter_mm":       meas["d_eq_mm"],
        "diameter_px_equiv": meas["d_eq_px"],
        "area_px":           meas["area_px"],
        "area_mm2":          meas["area_mm2"],
        "confidence":        confidence,
        "uncertainty_mm":    meas["uncertainty_mm"],
        "uncertainty_pct":   meas["uncertainty_pct"],
        "d_min_mm":          meas["d_min_mm"],
        "d_max_mm":          meas["d_max_mm"],
        "overlay_path":      overlay_path,
        "overlay_b64":       overlay_b64,
        "mask_preview_b64":  mask_b64,
        "success":           meas["d_eq_mm"] is not None,
        "processing_time_s": elapsed,
    })
    return result_base
