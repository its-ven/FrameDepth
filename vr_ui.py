"""
Floating settings panel for depth3d_vr. Drawn with Pillow into a texture and operated
with VR controller laser pointers (trigger = click / drag).
"""
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PW, PH = 1280, 800                      # panel texture size in pixels
PANEL_W_M = 1.6                         # physical size in the VR world (metres)
PANEL_H_M = PANEL_W_M * PH / PW

COL_BG = (16, 18, 26, 240)
COL_TEXT = (235, 238, 245, 255)
COL_DIM = (150, 156, 172, 255)
COL_ACCENT = (64, 156, 255, 255)
COL_WIDGET = (48, 52, 68, 255)
COL_HOVER = (72, 78, 100, 255)
COL_WARN = (255, 190, 80, 255)

TABS = ["3D", "Screen", "Environment"]

_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key in _fonts:
        return _fonts[key]
    names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
    f = None
    for n in names:
        try:
            f = ImageFont.truetype(n, size)
            break
        except OSError:
            continue
    if f is None:
        f = ImageFont.load_default(size)
    _fonts[key] = f
    return f


def ray_panel_hit(origin, direction, inv_model):
    """Intersect a world-space ray with the panel. Returns (t, px, py) or None."""
    o = inv_model @ np.append(origin, 1.0)
    d = inv_model[:3, :3] @ direction
    if abs(d[2]) < 1e-6:
        return None
    t = -o[2] / d[2]
    if t <= 0:
        return None
    x, y = o[0] + t * d[0], o[1] + t * d[1]
    if abs(x) > PANEL_W_M / 2 or abs(y) > PANEL_H_M / 2:
        return None
    return t, (x / PANEL_W_M + 0.5) * PW, (0.5 - y / PANEL_H_M) * PH


class Panel:
    def __init__(self, st, on_action):
        self.st = st
        self.on_action = on_action
        self.tab = 0
        self.info = dict(env_name="", env_layout="", env_video=False, env_status="", ipd=0.063)
        self.widgets = []
        self.hover = None
        self.active = None
        self._down = False
        self.dirty = True
        self._toast = ("", 0.0)
        self._layout()

    # ------------------------------------------------------------------ state
    def set_info(self, **kw):
        if any(self.info.get(k) != v for k, v in kw.items()):
            self.info.update(kw)
            self.dirty = True
            self._layout()

    def toast(self, text, seconds=2.5):
        self._toast = (text, time.time() + seconds)
        self.dirty = True

    def invalidate(self):
        self.dirty = True
        self._layout()

    # ------------------------------------------------------------------ layout
    def _layout(self):
        W = []
        for i, name in enumerate(TABS):
            W.append(dict(t="tab", idx=i, label=name, rect=(60 + i * 230, 92, 60 + i * 230 + 210, 148)))

        y = 190
        row = 84

        def slider(key, label, lo, hi, fmt):
            nonlocal y
            W.append(dict(t="slider", key=key, label=label, lo=lo, hi=hi, fmt=fmt,
                          rect=(500, y - 28, 1010, y + 28), track=(520, 990), y=y))
            y += row

        def toggle(key, label):
            nonlocal y
            W.append(dict(t="toggle", key=key, label=label, rect=(500, y - 28, 640, y + 28), y=y))
            y += row

        def button(action, label, x0=60, x1=480, big=False):
            W.append(dict(t="button", action=action, label=label, rect=(x0, y - 30, x1, y + 30), y=y))
            return y

        if self.tab == 0:
            slider("depth_cm", "3D depth", 0.0, 30.0, "{:.1f} cm")
            slider("conv", "Convergence", 0.0, 1.0, "{:.2f}")
            toggle("stereo3d", "3D enabled")
            toggle("swap", "Swap eyes")
            toggle("debug", "Show depth map")
        elif self.tab == 1:
            slider("fov", "Screen size", 20.0, 180.0, "{:.0f}\u00b0")
            slider("distance", "Distance", 1.0, 10.0, "{:.1f} m")
            slider("height", "Height", -1.5, 1.5, "{:+.2f} m")
            slider("sharpen", "Text sharpening", 0.0, 1.0, "{:.2f}")
            button("recenter", "Recenter screen", 500, 900)
            y += row
        else:
            W.append(dict(t="button", action="env_prev", label="\u25c0", rect=(60, y - 30, 160, y + 30), y=y))
            W.append(dict(t="button", action="env_next", label="\u25b6", rect=(1120, y - 30, 1220, y + 30), y=y))
            y += row
            W.append(dict(t="button", action="env_layout", label="Layout: " + (self.info["env_layout"] or "-"),
                          rect=(60, y - 30, 600, y + 30), y=y))
            y += row
            slider("env_brightness", "Brightness", 0.0, 1.5, "{:.2f}")
            slider("env_yaw", "Rotation", -180.0, 180.0, "{:+.0f}\u00b0")
            if self.info["env_video"]:
                toggle("env_play", "Play video")

        fy = 742
        for i, (label, action) in enumerate((("Save", "save"), ("Reset", "reset"), ("Close", "close"))):
            W.append(dict(t="button", action=action, label=label,
                          rect=(60 + i * 240, fy - 32, 60 + i * 240 + 210, fy + 32), y=fy))
        self.widgets = W
        self.dirty = True

    # ------------------------------------------------------------------ interaction
    def _widget_at(self, px, py):
        for w in self.widgets:
            x0, y0, x1, y1 = w["rect"]
            if x0 <= px <= x1 and y0 <= py <= y1:
                return w
        return None

    def _set_slider(self, w, px):
        t0, t1 = w["track"]
        t = min(1.0, max(0.0, (px - t0) / (t1 - t0)))
        self.st[w["key"]] = w["lo"] + t * (w["hi"] - w["lo"])
        self.dirty = True

    def pointer(self, px, py, down):
        """Feed the pointer position in panel pixels (None = not pointing at the panel)."""
        press = down and not self._down
        self._down = down
        if px is None:
            if self.hover is not None:
                self.hover = None
                self.dirty = True
            if not down:
                self.active = None
            return
        w = self._widget_at(px, py)
        if w is not self.hover:
            self.hover = w
            self.dirty = True
        if not down:
            self.active = None
            return
        if self.active is not None:
            self._set_slider(self.active, px)
            return
        if not press or w is None:
            return
        if w["t"] == "slider":
            self.active = w
            self._set_slider(w, px)
        elif w["t"] == "toggle":
            self.st[w["key"]] = not self.st[w["key"]]
            self.dirty = True
            self.on_action("toggled:" + w["key"])
        elif w["t"] == "button":
            self.on_action(w["action"])
        elif w["t"] == "tab":
            self.tab = w["idx"]
            self.hover = None
            self._layout()

    # ------------------------------------------------------------------ drawing
    def render_if_dirty(self):
        if self._toast[0] and time.time() > self._toast[1]:
            self._toast = ("", 0.0)
            self.dirty = True
        if not self.dirty:
            return None
        self.dirty = False
        return self.render()

    def render(self):
        img = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, PW - 1, PH - 1), radius=40, fill=COL_BG)
        d.text((60, 46), "Depth 3D", font=font(44, True), fill=COL_TEXT, anchor="lm")
        d.text((PW - 60, 46), "trigger: click / drag", font=font(24), fill=COL_DIM, anchor="rm")

        f_label, f_val = font(32), font(32, True)
        for w in self.widgets:
            hov = w is self.hover or w is self.active
            x0, y0, x1, y1 = w["rect"]
            if w["t"] == "tab":
                sel = w["idx"] == self.tab
                d.rounded_rectangle(w["rect"], radius=18,
                                    fill=COL_ACCENT if sel else (COL_HOVER if hov else COL_WIDGET))
                d.text(((x0 + x1) / 2, (y0 + y1) / 2), w["label"], font=font(30, True),
                       fill=COL_TEXT, anchor="mm")
            elif w["t"] == "slider":
                d.text((60, w["y"]), w["label"], font=f_label, fill=COL_TEXT, anchor="lm")
                t0, t1 = w["track"]
                val = float(self.st[w["key"]])
                frac = min(1.0, max(0.0, (val - w["lo"]) / (w["hi"] - w["lo"])))
                kx = t0 + frac * (t1 - t0)
                d.rounded_rectangle((t0, w["y"] - 6, t1, w["y"] + 6), radius=6, fill=COL_WIDGET)
                d.rounded_rectangle((t0, w["y"] - 6, kx, w["y"] + 6), radius=6, fill=COL_ACCENT)
                d.ellipse((kx - 20, w["y"] - 20, kx + 20, w["y"] + 20),
                          fill=(255, 255, 255, 255) if hov else COL_TEXT)
                d.text((1040, w["y"]), w["fmt"].format(val), font=f_val, fill=COL_TEXT, anchor="lm")
            elif w["t"] == "toggle":
                d.text((60, w["y"]), w["label"], font=f_label, fill=COL_TEXT, anchor="lm")
                on = bool(self.st[w["key"]])
                d.rounded_rectangle(w["rect"], radius=28,
                                    fill=COL_ACCENT if on else (COL_HOVER if hov else COL_WIDGET))
                kx = x1 - 28 if on else x0 + 28
                d.ellipse((kx - 22, w["y"] - 22, kx + 22, w["y"] + 22), fill=COL_TEXT)
            elif w["t"] == "button":
                d.rounded_rectangle(w["rect"], radius=20, fill=COL_HOVER if hov else COL_WIDGET)
                d.text(((x0 + x1) / 2, (y0 + y1) / 2), w["label"], font=font(30, True),
                       fill=COL_TEXT, anchor="mm")

        if self.tab == 0:
            far_cm = self.st["depth_cm"] * self.st["conv"]
            if far_cm / 100.0 > self.info["ipd"]:
                d.text((60, 640), "Far parts exceed your eye spacing - lower depth or convergence.",
                       font=font(26), fill=COL_WARN, anchor="lm")
            else:
                d.text((60, 640), "Tip: keep depth x convergence under about 6 for comfort.",
                       font=font(26), fill=COL_DIM, anchor="lm")
        elif self.tab == 2:
            name = self.info["env_name"] or "None (black void)"
            d.text((640, 190), name, font=font(32, True), fill=COL_TEXT, anchor="mm")
            status = self.info["env_status"]
            if status:
                d.text((60, 640), status, font=font(26), fill=COL_WARN, anchor="lm")
            else:
                d.text((60, 640), "Put 360/180 images or videos in the 'environments' folder.",
                       font=font(26), fill=COL_DIM, anchor="lm")

        if self._toast[0]:
            d.text((PW - 60, 742), self._toast[0], font=font(30, True), fill=COL_ACCENT, anchor="rm")
        return img.tobytes()
