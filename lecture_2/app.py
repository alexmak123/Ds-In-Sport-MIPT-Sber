from __future__ import annotations

import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

Point = tuple[float, float]


def _available_homography_methods() -> dict[str, int]:
    methods: dict[str, int] = {"DLT (0)": 0}
    candidates = [
        ("RANSAC", "RANSAC"),
        ("LMEDS", "LMEDS"),
        ("RHO", "RHO"),
        ("USAC_DEFAULT", "USAC_DEFAULT"),
        ("USAC_PARALLEL", "USAC_PARALLEL"),
        ("USAC_FAST", "USAC_FAST"),
        ("USAC_ACCURATE", "USAC_ACCURATE"),
        ("USAC_PROSAC", "USAC_PROSAC"),
        ("USAC_MAGSAC", "USAC_MAGSAC"),
    ]
    for label, attr in candidates:
        value = getattr(cv2, attr, None)
        if isinstance(value, int):
            methods[label] = value
    return methods


def _read_image(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _blend_overlay(dst_raw: np.ndarray | None, warped: np.ndarray | None, alpha: float) -> np.ndarray | None:
    if dst_raw is None or warped is None:
        return None
    if dst_raw.shape[:2] != warped.shape[:2]:
        return None
    a = float(np.clip(alpha, 0.0, 1.0))
    base = dst_raw.astype(np.float32)
    top = warped.astype(np.float32)
    out = base.copy()
    mask = np.any(warped > 0, axis=2)
    if np.any(mask):
        out[mask] = base[mask] * (1.0 - a) + top[mask] * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_points(image: np.ndarray | None, points: list[Point]) -> np.ndarray | None:
    if image is None:
        return None
    out = image.copy()
    h, w = out.shape[:2]
    r = max(14, int(min(h, w) * 0.02))
    font = max(0.8, min(1.5, min(h, w) / 650))
    for i, (x, y) in enumerate(points, start=1):
        xi = int(round(x))
        yi = int(round(y))
        cv2.circle(out, (xi, yi), r, (255, 80, 80), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (xi, yi), r, (255, 255, 255), 3, lineType=cv2.LINE_AA)
        pos = (xi + r + 6, yi - r - 6)
        cv2.putText(out, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(out, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, font, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def _next_hint(src_points: list[Point], dst_points: list[Point]) -> str:
    if len(src_points) == len(dst_points):
        return "Ставьте следующую точку на ФОТО."
    if len(src_points) == len(dst_points) + 1:
        return "Ставьте соответствующую точку на МАКЕТЕ."
    if len(dst_points) == len(src_points) + 1:
        return "Ставьте соответствующую точку на ФОТО."
    return "Количество точек сильно расходится. Ставьте пары по очереди."


def _status(src_points: list[Point], dst_points: list[Point], matrix: np.ndarray | None = None) -> str:
    lines = [
        "Инструкция:",
        "1. Загрузите фото и макет кнопками загрузки.",
        "2. Кликайте по большим изображениям ниже.",
        "3. Повторный клик по точке удаляет ее.",
        "4. Нажмите 'Построить гомографию'.",
        "",
        f"Фото: {len(src_points)} точек",
        f"Макет: {len(dst_points)} точек",
        _next_hint(src_points, dst_points),
    ]
    if len(src_points) != len(dst_points):
        lines.append("Внимание: число точек не совпадает.")
    if min(len(src_points), len(dst_points)) < 4:
        lines.append("Нужно минимум 4 пары точек.")
    if matrix is not None:
        lines.append("")
        lines.append("H =")
        lines.append(np.array2string(matrix, precision=4, suppress_small=True))
    return "\n".join(lines)


def _extract_xy(evt: gr.SelectData) -> Point | None:
    for value in (getattr(evt, "index", None), getattr(evt, "value", None)):
        if isinstance(value, dict):
            if "x" in value and "y" in value:
                return float(value["x"]), float(value["y"])
            coord = value.get("coord")
            if isinstance(coord, (tuple, list)) and len(coord) >= 2:
                return float(coord[0]), float(coord[1])
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            return float(value[0]), float(value[1])
    return None


def _hit_point(points: list[Point], x: float, y: float, image: np.ndarray) -> int | None:
    h, w = image.shape[:2]
    hit = max(20, int(min(h, w) * 0.03))
    hit2 = hit * hit
    for i, (px, py) in enumerate(points):
        dx = x - px
        dy = y - py
        if dx * dx + dy * dy <= hit2:
            return i
    return None


def _zoom(image: np.ndarray | None, center: Point | None, points: list[Point]) -> np.ndarray | None:
    if image is None or center is None:
        return None
    h, w = image.shape[:2]
    x, y = int(round(center[0])), int(round(center[1]))
    half = 70
    x0, x1 = max(0, x - half), min(w, x + half)
    y0, y1 = max(0, y - half), min(h, y + half)
    crop = image[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return None
    local = [(px - x0, py - y0) for px, py in points if x0 <= px < x1 and y0 <= py < y1]
    crop = _draw_points(crop, local)
    zoomed = cv2.resize(crop, (crop.shape[1] * 4, crop.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    zh, zw = zoomed.shape[:2]
    cx = int(round((x - x0) * 4))
    cy = int(round((y - y0) * 4))
    cv2.line(zoomed, (max(0, cx - 25), cy), (min(zw - 1, cx + 25), cy), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(zoomed, (cx, max(0, cy - 25)), (cx, min(zh - 1, cy + 25)), (255, 255, 0), 2, cv2.LINE_AA)
    return zoomed


def _load_source(path: str | None, dst_points: list[Point] | None):
    img = _read_image(path)
    src_points: list[Point] = []
    dst_points = list(dst_points or [])
    return img, src_points, _draw_points(img, src_points), None, None, None, _status(src_points, dst_points)


def _load_target(path: str | None, src_points: list[Point] | None):
    img = _read_image(path)
    dst_points: list[Point] = []
    src_points = list(src_points or [])
    return img, dst_points, _draw_points(img, dst_points), None, None, None, _status(src_points, dst_points)


def _click_src(src_raw: np.ndarray | None, src_points: list[Point] | None, dst_points: list[Point] | None, evt: gr.SelectData):
    src_points = list(src_points or [])
    dst_points = list(dst_points or [])
    if src_raw is None:
        return gr.update(), src_points, _status(src_points, dst_points)
    xy = _extract_xy(evt)
    if xy is None:
        return _draw_points(src_raw, src_points), src_points, _status(src_points, dst_points)
    x = float(np.clip(xy[0], 0, src_raw.shape[1] - 1))
    y = float(np.clip(xy[1], 0, src_raw.shape[0] - 1))
    hit = _hit_point(src_points, x, y, src_raw)
    if hit is None:
        src_points.append((x, y))
    else:
        src_points.pop(hit)
    return _draw_points(src_raw, src_points), src_points, _status(src_points, dst_points)


def _click_dst(dst_raw: np.ndarray | None, src_points: list[Point] | None, dst_points: list[Point] | None, evt: gr.SelectData):
    src_points = list(src_points or [])
    dst_points = list(dst_points or [])
    if dst_raw is None:
        return gr.update(), dst_points, _status(src_points, dst_points)
    xy = _extract_xy(evt)
    if xy is None:
        return _draw_points(dst_raw, dst_points), dst_points, _status(src_points, dst_points)
    x = float(np.clip(xy[0], 0, dst_raw.shape[1] - 1))
    y = float(np.clip(xy[1], 0, dst_raw.shape[0] - 1))
    hit = _hit_point(dst_points, x, y, dst_raw)
    if hit is None:
        dst_points.append((x, y))
    else:
        dst_points.pop(hit)
    return _draw_points(dst_raw, dst_points), dst_points, _status(src_points, dst_points)


def _undo_src(src_raw: np.ndarray | None, src_points: list[Point] | None, dst_points: list[Point] | None):
    src_points = list(src_points or [])
    dst_points = list(dst_points or [])
    if src_points:
        src_points.pop()
    return _draw_points(src_raw, src_points), src_points, _status(src_points, dst_points)


def _undo_dst(dst_raw: np.ndarray | None, src_points: list[Point] | None, dst_points: list[Point] | None):
    src_points = list(src_points or [])
    dst_points = list(dst_points or [])
    if dst_points:
        dst_points.pop()
    return _draw_points(dst_raw, dst_points), dst_points, _status(src_points, dst_points)


def _clear(src_raw: np.ndarray | None, dst_raw: np.ndarray | None):
    src_points: list[Point] = []
    dst_points: list[Point] = []
    return (
        _draw_points(src_raw, src_points),
        _draw_points(dst_raw, dst_points),
        src_points,
        dst_points,
        None,
        None,
        None,
        _status(src_points, dst_points),
    )


def _compute(
    src_raw: np.ndarray | None,
    dst_raw: np.ndarray | None,
    src_points: list[Point] | None,
    dst_points: list[Point] | None,
    overlay_alpha: float,
    homography_method_name: str,
    ransac_reproj_threshold: float,
):
    src_points = list(src_points or [])
    dst_points = list(dst_points or [])
    if src_raw is None or dst_raw is None:
        return None, None, None, _status(src_points, dst_points) + "\n\nОшибка: загрузите оба изображения."
    if len(src_points) != len(dst_points):
        return None, None, None, _status(src_points, dst_points) + "\n\nОшибка: количество точек должно совпадать."
    if len(src_points) < 4:
        return None, None, None, _status(src_points, dst_points) + "\n\nОшибка: нужно минимум 4 пары точек."
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)
    method_map = _available_homography_methods()
    method_code = method_map.get(homography_method_name, 0)
    h, mask = cv2.findHomography(
        src,
        dst,
        method=method_code,
        ransacReprojThreshold=float(ransac_reproj_threshold),
    )
    if h is None:
        return None, None, None, _status(src_points, dst_points) + "\n\nОшибка: гомография не посчиталась."
    warped = cv2.warpPerspective(src_raw, h, (dst_raw.shape[1], dst_raw.shape[0]))
    overlay = _blend_overlay(dst_raw, warped, overlay_alpha)
    info = f"\n\nМетод: {homography_method_name}\nRANSAC threshold: {float(ransac_reproj_threshold):.2f}"
    if mask is not None:
        inliers = int(mask.ravel().sum())
        info += f"\nInliers: {inliers}/{len(src_points)}"
    return warped, overlay, warped, _status(src_points, dst_points, h) + info


def _update_overlay(dst_raw: np.ndarray | None, warped_state: np.ndarray | None, overlay_alpha: float):
    return _blend_overlay(dst_raw, warped_state, overlay_alpha)

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Homography Tool") as demo:
        gr.Markdown("# Homography Tool (Football Field)")

        src_raw_state = gr.State(None)
        dst_raw_state = gr.State(None)
        src_points_state = gr.State([])
        dst_points_state = gr.State([])
        warped_state = gr.State(None)
        homography_methods = _available_homography_methods()
        gr.HTML(
            """
<style>
.lens-host { position: relative; }
.lens-overlay {
  position: absolute;
  width: 170px;
  height: 170px;
  border: 2px solid #ffd54f;
  border-radius: 50%;
  box-shadow: 0 4px 18px rgba(0,0,0,.25);
  pointer-events: none;
  display: none;
  z-index: 20;
  background-repeat: no-repeat;
  background-color: rgba(255,255,255,0.9);
}
.lens-overlay::after {
  content: "";
  position: absolute;
  left: 50%;
  top: 50%;
  width: 24px;
  height: 24px;
  transform: translate(-50%, -50%);
  border: 1px solid rgba(255,255,0,.9);
  border-radius: 50%;
}
</style>
<script>
(() => {
  const ZOOM = 2.6;
  const LENS = 170;

  function attachLens(rootId) {
    const root = document.getElementById(rootId);
    if (!root) return;
    const wrap = root.querySelector('.image-container, .wrap, .svelte-1ipelgc') || root;
    const img = root.querySelector('img');
    if (!img) return;

    if (img.dataset.lensBound === '1') return;
    img.dataset.lensBound = '1';

    root.classList.add('lens-host');
    let lens = root.querySelector('.lens-overlay');
    if (!lens) {
      lens = document.createElement('div');
      lens.className = 'lens-overlay';
      root.appendChild(lens);
    }

    function updateLens(ev) {
      const rect = img.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
        lens.style.display = 'none';
        return;
      }
      lens.style.display = 'block';
      lens.style.left = `${x - LENS/2}px`;
      lens.style.top = `${y - LENS/2}px`;
      lens.style.backgroundImage = `url("${img.src}")`;
      lens.style.backgroundSize = `${rect.width * ZOOM}px ${rect.height * ZOOM}px`;
      lens.style.backgroundPosition = `${-(x * ZOOM - LENS/2)}px ${-(y * ZOOM - LENS/2)}px`;
    }

    img.addEventListener('mousemove', updateLens);
    img.addEventListener('mouseenter', updateLens);
    img.addEventListener('mouseleave', () => { lens.style.display = 'none'; });
    root.addEventListener('mouseleave', () => { lens.style.display = 'none'; });
  }

  function boot() {
    attachLens('src-view');
    attachLens('dst-view');
  }

  boot();
  setInterval(boot, 1000);
})();
</script>
"""
        )

        with gr.Row():
            src_file = gr.File(label="Загрузка фото поля", file_types=["image"], type="filepath")
            dst_file = gr.File(label="Загрузка макета поля", file_types=["image"], type="filepath")

        with gr.Row():
            with gr.Column():
                src_view = gr.Image(
                    label="Фото поля (клик = добавить/удалить точку)",
                    type="numpy",
                    interactive=False,
                    height=520,
                    elem_id="src-view",
                )
                undo_src = gr.Button("Отменить точку (фото)")
            with gr.Column():
                dst_view = gr.Image(
                    label="Макет поля (клик = добавить/удалить точку)",
                    type="numpy",
                    interactive=False,
                    height=520,
                    elem_id="dst-view",
                )
                undo_dst = gr.Button("Отменить точку (макет)")
            with gr.Column():
                result_view = gr.Image(label="Результат гомографии", type="numpy", interactive=False, height=260)
                overlay_view = gr.Image(label="Наложение на макет", type="numpy", interactive=False, height=260)
                homography_method = gr.Dropdown(
                    label="Метод поиска гомографии",
                    choices=list(homography_methods.keys()),
                    value="RANSAC" if "RANSAC" in homography_methods else "DLT (0)",
                )
                ransac_thresh = gr.Slider(
                    minimum=0.5,
                    maximum=20.0,
                    value=3.0,
                    step=0.1,
                    label="ransacReprojThreshold",
                )
                overlay_alpha = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    label="Прозрачность наложения (0 = макет, 1 = warped фото)",
                )
                compute_btn = gr.Button("Построить гомографию", variant="primary")
                clear_btn = gr.Button("Очистить точки")

        status_box = gr.Textbox(label="Статус", value=_status([], []), lines=12, interactive=False)

        src_file.change(
            fn=_load_source,
            inputs=[src_file, dst_points_state],
            outputs=[src_raw_state, src_points_state, src_view, result_view, overlay_view, warped_state, status_box],
            queue=False,
        )
        dst_file.change(
            fn=_load_target,
            inputs=[dst_file, src_points_state],
            outputs=[dst_raw_state, dst_points_state, dst_view, result_view, overlay_view, warped_state, status_box],
            queue=False,
        )

        src_view.select(
            fn=_click_src,
            inputs=[src_raw_state, src_points_state, dst_points_state],
            outputs=[src_view, src_points_state, status_box],
            queue=False,
        )
        dst_view.select(
            fn=_click_dst,
            inputs=[dst_raw_state, src_points_state, dst_points_state],
            outputs=[dst_view, dst_points_state, status_box],
            queue=False,
        )

        undo_src.click(
            fn=_undo_src,
            inputs=[src_raw_state, src_points_state, dst_points_state],
            outputs=[src_view, src_points_state, status_box],
            queue=False,
        )
        undo_dst.click(
            fn=_undo_dst,
            inputs=[dst_raw_state, src_points_state, dst_points_state],
            outputs=[dst_view, dst_points_state, status_box],
            queue=False,
        )

        clear_btn.click(
            fn=_clear,
            inputs=[src_raw_state, dst_raw_state],
            outputs=[src_view, dst_view, src_points_state, dst_points_state, result_view, overlay_view, warped_state, status_box],
            queue=False,
        )

        compute_btn.click(
            fn=_compute,
            inputs=[
                src_raw_state,
                dst_raw_state,
                src_points_state,
                dst_points_state,
                overlay_alpha,
                homography_method,
                ransac_thresh,
            ],
            outputs=[result_view, overlay_view, warped_state, status_box],
            queue=False,
        )

        overlay_alpha.change(
            fn=_update_overlay,
            inputs=[dst_raw_state, warped_state, overlay_alpha],
            outputs=[overlay_view],
            queue=False,
        )

    return demo


if __name__ == "__main__":
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    no_proxy = os.environ.get("NO_PROXY", "")
    parts = {p.strip() for p in no_proxy.split(",") if p.strip()}
    parts.update({"127.0.0.1", "localhost"})
    os.environ["NO_PROXY"] = ",".join(sorted(parts))
    build_app().launch(server_name="127.0.0.1", share=False)
