"""
depth3d.py - screen capture + depth estimation core, and a simple side-by-side (SBS) window.

Two roles:
  * Library: `Grabber` (Windows Graphics Capture) and `DepthWorker` (Depth Anything V2 on the
    GPU, scene-cut aware) are used by depth3d_vr.py, the main SteamVR player.
  * Standalone: `python depth3d.py --source-monitor 1` opens a plain SBS window that you can
    feed to a 3D-capable viewer (Desktop+ HSBS, Bigscreen...). Simpler and less capable than
    the VR player: older warp shader, no menu, no environments.

Controls of the standalone SBS window:
  Up / Down      depth strength
  Left / Right   convergence (which depth sits at the screen plane)
  X              swap eyes
  D              toggle depth-map debug view
  F              toggle fullscreen
  Esc            quit
"""
import argparse
import threading
import time

import glfw
import moderngl
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation
from windows_capture import Frame, InternalCaptureControl, WindowsCapture

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


# --------------------------------------------------------------------------
# Window listing helper
# --------------------------------------------------------------------------
def list_windows():
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    titles = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                titles.append(buf.value)
        return True

    user32.EnumWindows(cb, 0)
    return titles


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------
class Grabber:
    """Keeps only the newest captured frame (BGRA, HxWx4 uint8)."""

    def __init__(self, window_name=None, monitor_index=None, cursor=True):
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        kwargs = dict(cursor_capture=cursor, draw_border=False)
        if window_name:
            kwargs["window_name"] = window_name
        else:
            kwargs["monitor_index"] = monitor_index  # 1-based
        cap = WindowsCapture(**kwargs)

        @cap.event
        def on_frame_arrived(frame: Frame, ctrl: InternalCaptureControl):
            buf = np.array(frame.frame_buffer, copy=True)
            with self.lock:
                self.frame = buf
                self.seq += 1

        @cap.event
        def on_closed():
            print("Capture session closed.")

        self.control = cap.start_free_threaded()

    def latest(self):
        with self.lock:
            return self.frame, self.seq

    def stop(self):
        try:
            self.control.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Depth estimation worker
# --------------------------------------------------------------------------
class DepthWorker(threading.Thread):
    def __init__(self, grabber, width, smooth, model_id, cut_threshold=0.12):
        super().__init__(daemon=True)
        self.grabber = grabber
        self.width = max(112, (width // 14) * 14)
        self.smooth = smooth
        self.cut_threshold = cut_threshold
        self.active = True         # set False to idle the model (e.g. when 3D is switched off)
        self.frame = None          # the captured frame the current depth belongs to
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.depth = None          # float32 HxW in 0..1 (1 = near)
        self.seq = 0
        self.fps = 0.0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.device == "cpu":
            print("WARNING: CUDA not available, running on CPU (will be slow).")
        print(f"Loading {model_id} on {self.device} ...")
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(model_id)
            .to(self.device, self.dtype)
            .eval()
        )
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def latest(self):
        with self.lock:
            return self.depth, self.seq

    def latest_pair(self):
        """Depth plus the exact captured frame it was computed from (keeps them in sync)."""
        with self.lock:
            return self.depth, self.frame, self.seq

    def run(self):
        last_seq = -1
        lo = hi = None
        prev = None
        prev_thumb = None
        n, t0 = 0, time.time()
        while not self.stop_evt.is_set():
            if not self.active:
                time.sleep(0.02)
                continue
            frame, seq = self.grabber.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq

            with torch.inference_mode():
                h, w = frame.shape[:2]
                mw = self.width
                mh = max(14, int(round(mw * h / w / 14)) * 14)

                t = torch.from_numpy(frame).to(self.device, non_blocking=True)
                t = t[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0
                t = F.interpolate(t, size=(mh, mw), mode="bilinear",
                                  antialias=True, align_corners=False)

                # Tiny luminance thumbnail -> scene-cut / motion detection
                thumb = F.adaptive_avg_pool2d(t, (18, 32)).mean(1)
                if prev_thumb is None:
                    diff, cut = 1.0, True
                else:
                    diff = (thumb - prev_thumb).abs().mean().item()
                    cut = diff > self.cut_threshold
                prev_thumb = thumb
                if cut:                      # hard cut: forget everything from the old shot
                    lo = hi = None
                    prev = None

                t = ((t - self.mean) / self.std).to(self.dtype)

                d = self.model(pixel_values=t).predicted_depth[0].float()  # HxW, larger = nearer

                # Robust, temporally smoothed normalisation (avoids pumping/flicker)
                sample = d.flatten()[::7]
                q = torch.quantile(sample, torch.tensor([0.02, 0.98], device=self.device))
                l, hgh = q[0].item(), q[1].item()
                if lo is None:
                    lo, hi = l, hgh
                else:
                    lo += 0.08 * (l - lo)
                    hi += 0.08 * (hgh - hi)
                dn = ((d - lo) / max(hi - lo, 1e-6)).clamp(0, 1)

                if prev is not None and prev.shape == dn.shape:
                    # calm shot -> smooth heavily; lots of motion -> trust the new frame
                    alpha = self.smooth + (1 - self.smooth) * min(1.0, diff / 0.06)
                    dn = prev * (1 - alpha) + dn * alpha
                prev = dn
                out = dn.cpu().numpy().astype(np.float32)

            with self.lock:
                self.depth = out
                self.frame = frame
                self.seq += 1

            n += 1
            if time.time() - t0 >= 2.0:
                self.fps = n / (time.time() - t0)
                n, t0 = 0, time.time()


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
VERT = """
#version 330
in vec2 pos;
out vec2 uv;
void main() {
    uv = vec2(pos.x * 0.5 + 0.5, 1.0 - (pos.y * 0.5 + 0.5));
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

FRAG = """
#version 330
uniform sampler2D color_tex;
uniform sampler2D depth_tex;
uniform float strength;     // max horizontal shift per eye, as fraction of width
uniform float convergence;  // depth value (0..1) that sits at the screen plane
uniform float eye;          // -1 = left, +1 = right
uniform int steps;
uniform int show_depth;
in vec2 uv;
out vec4 fragColor;
void main() {
    // Backward warp via fixed-point iteration: find the source pixel whose
    // own depth-based shift lands on this output pixel.
    vec2 src = uv;
    for (int i = 0; i < steps; i++) {
        float d = texture(depth_tex, src).r;
        src.x = uv.x + eye * strength * (d - convergence);
    }
    if (show_depth == 1) {
        fragColor = vec4(vec3(texture(depth_tex, uv).r), 1.0);
    } else {
        fragColor = vec4(texture(color_tex, src).rgb, 1.0);
    }
}
"""


def make_tex(ctx, w, h, comps, dtype, data=None):
    tex = ctx.texture((w, h), comps, data, dtype=dtype)
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


def main():
    ap = argparse.ArgumentParser(description="Window -> depth -> SBS 3D")
    ap.add_argument("--window", help="Title (or part of it) of the window to capture")
    ap.add_argument("--source-monitor", type=int, default=None,
                    help="Capture a whole monitor instead of a window (1 = first monitor)")
    ap.add_argument("--list", action="store_true",
                    help="List visible window titles and monitors, then exit")
    ap.add_argument("--model", choices=list(MODELS), default="base",
                    help="Depth model size (small = fastest, large = best quality)")
    ap.add_argument("--depth-width", type=int, default=700,
                    help="Model input width (lower = faster). Try 518 or 392.")
    ap.add_argument("--strength", type=float, default=0.025)
    ap.add_argument("--convergence", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=4, help="Warp iterations (quality/cost)")
    ap.add_argument("--smooth", type=float, default=0.5,
                    help="Depth temporal blend, 1 = no smoothing, lower = smoother")
    ap.add_argument("--out-monitor", type=int, default=None,
                    help="Start the SBS output fullscreen on this monitor (0-based). "
                         "Must NOT be the monitor you are capturing.")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()

    if args.list:
        print("Windows:")
        for t in sorted(set(list_windows())):
            print("  ", t)
        if glfw.init():
            print("Monitors (--source-monitor is 1-based, --out-monitor is 0-based):")
            for i, m in enumerate(glfw.get_monitors()):
                vm = glfw.get_video_mode(m)
                x, y = glfw.get_monitor_pos(m)
                print(f"   out={i}  source={i + 1}  {glfw.get_monitor_name(m).decode(errors='ignore')}"
                      f"  {vm.size.width}x{vm.size.height} at ({x},{y})")
            glfw.terminate()
        return
    if not args.window and args.source_monitor is None:
        ap.error("give --window TITLE or --source-monitor N (use --list to see options)")
    if args.window and args.source_monitor is not None:
        ap.error("use either --window or --source-monitor, not both")
    if args.source_monitor is not None and args.out_monitor is not None \
            and args.source_monitor - 1 == args.out_monitor:
        ap.error("output monitor and captured monitor are the same: that would feed back on itself")

    grabber = Grabber(args.window, args.source_monitor)
    worker = DepthWorker(grabber, args.depth_width, args.smooth, MODELS[args.model])
    worker.start()

    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    monitors = glfw.get_monitors()
    mon = None
    w, h = args.width, args.height
    if args.out_monitor is not None and args.out_monitor < len(monitors):
        mon = monitors[args.out_monitor]
        mode = glfw.get_video_mode(mon)
        w, h = mode.size.width, mode.size.height
    win = glfw.create_window(w, h, "depth3d SBS", mon, None)
    glfw.make_context_current(win)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "2f", "pos")])

    state = {
        "strength": args.strength,
        "conv": args.convergence,
        "swap": False,
        "debug": False,
        "fullscreen": mon is not None,
        "windowed_geom": (100, 100, args.width, args.height),
    }

    def on_key(window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_UP:
            state["strength"] = min(0.1, state["strength"] + 0.0025)
        elif key == glfw.KEY_DOWN:
            state["strength"] = max(0.0, state["strength"] - 0.0025)
        elif key == glfw.KEY_RIGHT:
            state["conv"] = min(1.0, state["conv"] + 0.02)
        elif key == glfw.KEY_LEFT:
            state["conv"] = max(0.0, state["conv"] - 0.02)
        elif key == glfw.KEY_X and action == glfw.PRESS:
            state["swap"] = not state["swap"]
        elif key == glfw.KEY_D and action == glfw.PRESS:
            state["debug"] = not state["debug"]
        elif key == glfw.KEY_F and action == glfw.PRESS:
            if state["fullscreen"]:
                x, y, ww, hh = state["windowed_geom"]
                glfw.set_window_monitor(window, None, x, y, ww, hh, 0)
            else:
                x, y = glfw.get_window_pos(window)
                ww, hh = glfw.get_window_size(window)
                state["windowed_geom"] = (x, y, ww, hh)
                m = glfw.get_primary_monitor()
                vm = glfw.get_video_mode(m)
                glfw.set_window_monitor(window, m, 0, 0, vm.size.width, vm.size.height,
                                        vm.refresh_rate)
            state["fullscreen"] = not state["fullscreen"]

    glfw.set_key_callback(win, on_key)

    color_tex = None
    color_size = None
    depth_tex = make_tex(ctx, 2, 2, 1, "f4", np.full((2, 2), 0.5, dtype="f4").tobytes())
    depth_size = (2, 2)
    last_color_seq = -1
    last_depth_seq = -1
    last_title = 0.0

    prog["color_tex"].value = 0
    prog["depth_tex"].value = 1
    prog["steps"].value = args.steps

    try:
        while not glfw.window_should_close(win):
            glfw.poll_events()

            frame, cseq = grabber.latest()
            if frame is not None and cseq != last_color_seq:
                last_color_seq = cseq
                fh, fw = frame.shape[:2]
                if color_size != (fw, fh):
                    if color_tex is not None:
                        color_tex.release()
                    color_tex = make_tex(ctx, fw, fh, 4, "f1")
                    color_tex.swizzle = "BGRA"
                    color_size = (fw, fh)
                color_tex.write(np.ascontiguousarray(frame))

            depth, dseq = worker.latest()
            if depth is not None and dseq != last_depth_seq:
                last_depth_seq = dseq
                dh, dw = depth.shape
                if depth_size != (dw, dh):
                    depth_tex.release()
                    depth_tex = make_tex(ctx, dw, dh, 1, "f4")
                    depth_size = (dw, dh)
                depth_tex.write(depth.tobytes())

            fbw, fbh = glfw.get_framebuffer_size(win)
            ctx.viewport = (0, 0, fbw, fbh)
            ctx.clear(0.0, 0.0, 0.0)

            if color_tex is not None:
                color_tex.use(0)
                depth_tex.use(1)
                prog["strength"].value = state["strength"]
                prog["convergence"].value = state["conv"]
                prog["show_depth"].value = 1 if state["debug"] else 0
                half = fbw // 2
                sign = -1.0 if state["swap"] else 1.0
                for i, eye in enumerate((-1.0 * sign, 1.0 * sign)):
                    ctx.viewport = (i * half, 0, half, fbh)
                    prog["eye"].value = eye
                    vao.render(moderngl.TRIANGLE_STRIP)

            glfw.swap_buffers(win)

            now = time.time()
            if now - last_title > 1.0:
                last_title = now
                glfw.set_window_title(
                    win,
                    f"depth3d SBS | depth {worker.fps:.0f} fps | "
                    f"strength {state['strength']:.3f} | conv {state['conv']:.2f}",
                )
    finally:
        worker.stop_evt.set()
        grabber.stop()
        glfw.terminate()


if __name__ == "__main__":
    main()
