"""
app/core/vision/hole_detector.py
=================================
Pipeline OpenCV clássico para estimativa de diâmetro de furo mediante imagem.

Estratégia
----------
1. Detectar marcador ArUco (escala) → mm_per_pixel
2. Corrigir perspectiva se necessário (opcional, via homografia)
3. Detectar furo via HoughCircles (furo circular) ou Canny + findContours + fitEllipse (forma arbitrária)
4. Converter diâmetro pixels → mm usando mm_per_pixel
5. Estimar incerteza e confiança
6. Retornar overlay anotado + metadados completos para auditoria

Requerimentos
-------------
  opencv-contrib-python-headless >= 4.7   (módulo aruco incluído)
  Pillow >= 9.0
  numpy >= 1.24

Notas de compatibilidade ArUco
-------------------------------
  OpenCV 4.7+  → cv2.aruco.ArucoDetector  (nova API)
  OpenCV < 4.7 → cv2.aruco.detectMarkers  (API legada; mantida como fallback)
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Resultado da detecção
# ---------------------------------------------------------------------------

@dataclass
class HoleDetectionResult:
    """Resultado completo do pipeline de detecção de furo."""

    success: bool
    """True se o diâmetro em mm foi estimado com sucesso."""

    diameter_mm: Optional[float] = None
    """Diâmetro estimado do furo em milímetros (None se sem escala ou falha)."""

    diameter_px: Optional[float] = None
    """Diâmetro estimado do furo em pixels."""

    uncertainty_pct: Optional[float] = None
    """Incerteza estimada (±%) do diâmetro_mm."""

    mm_per_pixel: Optional[float] = None
    """Fator de escala calculado a partir do marcador ArUco."""

    method: str = ""
    """Método de detecção do furo: 'HoughCircles' | 'Canny+fitEllipse' | 'Canny+minEnclosingCircle' | 'none'."""

    aruco_detected: bool = False
    """Se um marcador ArUco foi detectado na imagem."""

    aruco_ids: list = field(default_factory=list)
    """IDs dos marcadores ArUco detectados."""

    aruco_dict_used: str = ""
    """Dicionário ArUco utilizado."""

    confidence: float = 0.0
    """Confiança da estimativa (0–1)."""

    annotated_image_bytes: Optional[bytes] = None
    """Imagem PNG anotada com overlay (marcador, contorno, texto)."""

    warnings: list[str] = field(default_factory=list)
    """Lista de advertências do processamento."""

    processing_time_s: float = 0.0
    """Tempo de processamento em segundos."""


# ---------------------------------------------------------------------------
# Dicionários ArUco suportados (tentados nesta ordem)
# ---------------------------------------------------------------------------

_ARUCO_DICT_NAMES = [
    ("DICT_4X4_50",  cv2.aruco.DICT_4X4_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
]


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _detect_aruco(
    gray: np.ndarray,
) -> tuple[list, list, str]:
    """
    Tenta detectar marcadores ArUco nos dicionários suportados.

    Retorna (corners, ids, dict_name). corners e ids estão vazios se nada detectado.
    Compatível com OpenCV < 4.7 (detectMarkers) e >= 4.7 (ArucoDetector).
    """
    for dict_name, dict_id in _ARUCO_DICT_NAMES:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()

        try:
            # Nova API (OpenCV >= 4.7)
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            # Fallback API legada (OpenCV < 4.7)
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=params
            )

        if ids is not None and len(ids) > 0:
            return list(corners), ids.flatten().tolist(), dict_name

    return [], [], ""


def _marker_side_px(corners: list) -> float:
    """
    Calcula o comprimento médio dos quatro lados do primeiro marcador (pixels).
    Usa a média dos 4 lados para robustez a perspectiva leve.
    """
    c = corners[0].reshape(4, 2)
    sides = [
        np.linalg.norm(c[1] - c[0]),
        np.linalg.norm(c[2] - c[1]),
        np.linalg.norm(c[3] - c[2]),
        np.linalg.norm(c[0] - c[3]),
    ]
    return float(np.mean(sides))


def _encode_png(bgr: np.ndarray) -> bytes:
    """Converte array BGR para bytes PNG."""
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode falhou ao gerar PNG")
    return bytes(buf)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def detect_hole(
    image_bytes: bytes,
    marker_real_mm: float = 50.0,
    hough_dp: float = 1.2,
    hough_min_dist_factor: float = 0.06,
    hough_param1: float = 80.0,
    hough_param2: float = 28.0,
) -> HoleDetectionResult:
    """
    Estima o diâmetro do furo a partir de bytes de imagem.

    Parâmetros
    ----------
    image_bytes            : bytes da imagem (JPEG, PNG, etc.)
    marker_real_mm         : tamanho real (lado) do marcador ArUco em mm
    hough_dp               : razão resolução acumulador/imagem (HoughCircles)
    hough_min_dist_factor  : distância mínima entre centros como fração da menor dimensão
    hough_param1           : limiar superior Canny para HoughCircles
    hough_param2           : limiar acumulador HoughCircles (menor = detecta mais, menos preciso)
    """
    t0 = time.time()

    # ── 1. Carregar imagem ────────────────────────────────────────────────
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Limita resolução para ≤ 2 MP (acelera processamento sem perda de precisão significativa)
        max_dim = 1600
        w0, h0 = pil_img.size
        if max(w0, h0) > max_dim:
            scale = max_dim / max(w0, h0)
            pil_img = pil_img.resize(
                (int(w0 * scale), int(h0 * scale)), Image.LANCZOS
            )
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        return HoleDetectionResult(
            success=False,
            warnings=[f"Falha ao carregar imagem: {exc}"],
        )

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    annotated = img_bgr.copy()
    warnings: list[str] = []

    # ── 2. Detectar marcador ArUco ────────────────────────────────────────
    aruco_corners, aruco_ids, aruco_dict_name = _detect_aruco(gray)
    mm_per_pixel: Optional[float] = None
    aruco_detected = len(aruco_ids) > 0

    if aruco_detected:
        side_px = _marker_side_px(aruco_corners)
        mm_per_pixel = marker_real_mm / side_px
        # Desenha contornos e IDs do marcador
        try:
            cv2.aruco.drawDetectedMarkers(annotated, aruco_corners)
        except Exception:
            pass
    else:
        warnings.append(
            "Marcador ArUco NÃO detectado — sem referência de escala. "
            "Posicione um marcador impresso ao lado do furo e reenvie a imagem."
        )

    # ── 3. Pré-processamento ─────────────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # ── 4. Detecção do furo — HoughCircles ───────────────────────────────
    min_dim = min(h, w)
    min_dist = max(int(min_dim * hough_min_dist_factor), 10)

    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=hough_dp,
        minDist=min_dist,
        param1=hough_param1,
        param2=hough_param2,
        minRadius=int(min_dim * 0.01),
        maxRadius=int(min_dim * 0.45),
    )

    diameter_px: Optional[float] = None
    cx_hole, cy_hole = None, None
    detection_method = ""

    if circles is not None:
        circles_r = np.uint16(np.around(circles))
        # OpenCV ordena por acumulador (maior = mais confiável)
        best = circles_r[0][0]
        cx_hole, cy_hole, radius_px = int(best[0]), int(best[1]), int(best[2])
        diameter_px = float(radius_px * 2)
        detection_method = "HoughCircles"
        cv2.circle(annotated, (cx_hole, cy_hole), radius_px, (0, 255, 0), 3)
        cv2.circle(annotated, (cx_hole, cy_hole), 4, (0, 0, 255), -1)

    # ── 5. Fallback — Canny + findContours + fitEllipse ──────────────────
    if diameter_px is None:
        edges = cv2.Canny(enhanced, 50, 150)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_cnt = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.25 and area > best_area:
                best_area = area
                best_cnt = cnt

        if best_cnt is not None:
            if len(best_cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(best_cnt)
                    (ex, ey), (ma, mi), _ = ellipse
                    diameter_px = float(max(ma, mi))
                    cx_hole, cy_hole = int(ex), int(ey)
                    detection_method = "Canny+fitEllipse"
                    cv2.ellipse(annotated, ellipse, (0, 255, 0), 3)
                    cv2.circle(annotated, (cx_hole, cy_hole), 4, (0, 0, 255), -1)
                except cv2.error:
                    (fx, fy), fr = cv2.minEnclosingCircle(best_cnt)
                    diameter_px = float(fr * 2)
                    cx_hole, cy_hole = int(fx), int(fy)
                    detection_method = "Canny+minEnclosingCircle"
                    cv2.circle(annotated, (cx_hole, cy_hole), int(fr), (0, 255, 0), 3)
            else:
                (fx, fy), fr = cv2.minEnclosingCircle(best_cnt)
                diameter_px = float(fr * 2)
                cx_hole, cy_hole = int(fx), int(fy)
                detection_method = "Canny+minEnclosingCircle"
                cv2.circle(annotated, (cx_hole, cy_hole), int(fr), (0, 255, 0), 3)

    # ── 6. Nenhum furo detectado ─────────────────────────────────────────
    if diameter_px is None or diameter_px <= 0:
        dt = time.time() - t0
        return HoleDetectionResult(
            success=False,
            aruco_detected=aruco_detected,
            aruco_ids=aruco_ids,
            aruco_dict_used=aruco_dict_name,
            mm_per_pixel=mm_per_pixel,
            method="none",
            annotated_image_bytes=_encode_png(annotated),
            warnings=warnings
            + [
                "Nenhum furo detectado. Tente: melhor iluminação, fundo contrastante, "
                "enquadramento mais próximo ou ajuste manual."
            ],
            processing_time_s=dt,
        )

    # ── 7. Converter pixels → mm ─────────────────────────────────────────
    diameter_mm: Optional[float] = None
    uncertainty_pct: Optional[float] = None
    confidence = 0.0

    if mm_per_pixel is not None:
        diameter_mm = diameter_px * mm_per_pixel

        # Incerteza estimada
        # Base: ±5% (erro de localização de bordas + imprecisão do marcador impresso)
        base_unc = 5.0
        if detection_method.startswith("Canny"):
            base_unc += 3.0  # contornos menos estáveis que circulos de Hough
        if len(aruco_ids) > 1:
            base_unc -= 1.0  # múltiplos marcadores → melhor média
        uncertainty_pct = max(2.0, base_unc)

        confidence = 0.70 if detection_method == "HoughCircles" else 0.50
        if len(aruco_ids) > 1:
            confidence = min(0.90, confidence + 0.10)

        label = f"d = {diameter_mm:.1f} mm  ±{uncertainty_pct:.0f}%"
        cv2.putText(
            annotated, label, (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 255), 3,
        )
        cv2.putText(
            annotated, f"Escala: {mm_per_pixel:.4f} mm/px", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2,
        )
        cv2.putText(
            annotated, f"Metodo: {detection_method}", (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (180, 180, 180), 2,
        )
    else:
        # Sem escala: reporta apenas pixels
        warnings.append(
            f"Diâmetro medido em pixels: {diameter_px:.1f} px. "
            "Sem marcador ArUco não é possível converter para mm."
        )
        confidence = 0.15
        cv2.putText(
            annotated, f"d = {diameter_px:.0f} px  (sem escala)", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 3,
        )
        cv2.putText(
            annotated, f"Metodo: {detection_method}", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (180, 180, 180), 2,
        )

    dt = time.time() - t0
    return HoleDetectionResult(
        success=(diameter_mm is not None),
        diameter_mm=diameter_mm,
        diameter_px=diameter_px,
        uncertainty_pct=uncertainty_pct,
        mm_per_pixel=mm_per_pixel,
        method=detection_method,
        aruco_detected=aruco_detected,
        aruco_ids=aruco_ids,
        aruco_dict_used=aruco_dict_name,
        confidence=confidence,
        annotated_image_bytes=_encode_png(annotated),
        warnings=warnings,
        processing_time_s=dt,
    )


# ---------------------------------------------------------------------------
# Gerador de marcador ArUco para impressão
# ---------------------------------------------------------------------------

def generate_aruco_png(
    marker_id: int = 0,
    dict_name: str = "DICT_4X4_50",
    size_px: int = 400,
) -> bytes:
    """
    Gera um marcador ArUco como PNG (bytes) para impressão.

    Parâmetros
    ----------
    marker_id  : ID do marcador (0–49 para DICT_4X4_50)
    dict_name  : nome do dicionário ArUco
    size_px    : tamanho da imagem gerada em pixels (quadrada)

    Retorna
    -------
    bytes do PNG com borda branca para facilitar impressão e detecção.
    """
    dict_map = {
        "DICT_4X4_50":  cv2.aruco.DICT_4X4_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    }
    dict_id = dict_map.get(dict_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    inner = size_px - 80  # deixa borda branca ao redor
    marker_img = np.zeros((inner, inner), dtype=np.uint8)

    try:
        # OpenCV >= 4.7
        cv2.aruco.generateImageMarker(aruco_dict, marker_id, inner, marker_img, 1)
    except AttributeError:
        # Fallback legado
        marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, inner)

    # Borda branca
    canvas = np.ones((size_px, size_px), dtype=np.uint8) * 255
    offset = 40
    canvas[offset : offset + inner, offset : offset + inner] = marker_img

    # Converte para BGR para adicionar texto colorido
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        canvas_bgr,
        f"ArUco {dict_name}  ID={marker_id}",
        (10, size_px - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 0, 180), 1,
    )

    return _encode_png(canvas_bgr)
