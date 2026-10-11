"""
depth3d_vr.py - SteamVR 2D -> 3D player with an in-VR settings panel and immersive environments.

Captures a monitor (or window), estimates depth with Depth Anything V2, and shows it as a
curved screen in SteamVR with real per-eye depth. It is a normal SteamVR scene app, so it
streams to the Steam Frame like any VR game.

In VR:   press MENU or A on a controller to open the settings panel, point with the laser,
         trigger = click / drag. Settings are saved to settings.json (Save button, and
         automatically on exit).
Hotkeys (hold Ctrl+Alt, work even when another window has focus):
  Space          open / close settings panel      Enter    click (fallback if trigger fails)
  Up / Down      depth strength                   PgUp/PgDn  screen size
  Left / Right   convergence                      Home     recenter
  End            swap eyes                        Insert   depth-map debug view
  Delete         3D on / off
"""
import argparse
import copy
import ctypes
import json
import math
import time
from pathlib import Path

import glfw
import moderngl
import numpy as np
import openvr

from depth3d import MODELS, DepthWorker, Grabber, list_windows, make_tex
from vr_env import LAYOUTS, EnvManager
from vr_shaders import (DILATE_FRAG, FLAT_FRAG, FLAT_VERT, FS_VERT, PANEL_FRAG, PANEL_VERT,
                        REFINE_FRAG, SCREEN_FRAG, SCREEN_VERT)
from vr_ui import PANEL_H_M, PANEL_W_M, PH, PW, Panel, ray_panel_hit

HERE = Path(__file__).resolve().parent
SETTINGS_PATH = HERE / "settings.json"
ENV_DIR = HERE / "environments"

SAVE_KEYS = ["depth_cm", "conv", "fov", "distance", "height", "sharpen", "stereo3d", "swap",
             "env_name", "env_layouts", "env_brightness", "env_yaw", "env_play", "pointer_tilt"]

# Hotkey limits and adjustment speeds (units per second while held)
LIMITS = dict(depth_cm=(0.0, 30.0), conv=(0.0, 1.0), fov=(20.0, 180.0))
RATES = dict(depth_cm=5.0, conv=0.3, fov=35.0)

UI_DISTANCE = 1.4          # metres in front of you when the panel opens


# ---------------------------------------------------------------- math helpers
def m34(m):
    a = np.eye(4, dtype=np.float64)
    for r in range(3):
        for c in range(4):
            a[r, c] = m[r][c]
    return a


def m44(m):
    return np.array([[m[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float64)


def translate(x, y, z):
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = (x, y, z)
    return m


def head_yaw(head):
    """Yaw (about +Y) that points -Z at where the head is looking, ignoring pitch/roll."""
    fwd = -head[:3, 2]
    return math.atan2(-fwd[0], -fwd[2])


def build_screen(aspect, dist, fov_deg, segs=96):
    """Curved (cylindrical) screen centred on the viewer. Returns (vertex array, width in m)."""
    phi = math.radians(fov_deg)
    width = phi * dist
    height = width / aspect
    v = []
    for i in range(segs + 1):
        t = i / segs
        a = (t - 0.5) * phi
        x, z = dist * math.sin(a), -dist * math.cos(a)
        v += [x, height / 2, z, t, 0.0]
        v += [x, -height / 2, z, t, 1.0]
    return np.array(v, dtype="f4"), width


def pointer_ray(pose, tilt_deg):
    t = math.radians(tilt_deg)
    d = pose[:3, :3] @ np.array([0.0, math.sin(t), -math.cos(t)])
    return pose[:3, 3].copy(), d / np.linalg.norm(d)


# ---------------------------------------------------------------- settings
def load_saved():
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except Exception:                                   # noqa: BLE001
        return {}


def save_settings(st):
    SETTINGS_PATH.write_text(json.dumps({k: st[k] for k in SAVE_KEYS}, indent=2))


# ---------------------------------------------------------------- keyboard (global)
user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None
VK = dict(CTRL=0x11, ALT=0x12, SPACE=0x20, PGUP=0x21, PGDN=0x22, END=0x23, HOME=0x24,
          LEFT=0x25, UP=0x26, RIGHT=0x27, DOWN=0x28, INS=0x2D, DEL=0x2E, RETURN=0x0D)


def down(name):
    return user32 is not None and bool(user32.GetAsyncKeyState(VK[name]) & 0x8000)


# ---------------------------------------------------------------- controllers
class Controllers:
    """Legacy controller polling: pose + trigger + buttons for every connected controller."""

    def __init__(self, vrsys, debug):
        self.vrsys = vrsys
        self.debug = debug
        self.devs = []
        self.frame = 0
        self.prev = {}

    def poll(self, poses):
        if self.frame % 45 == 0:
            self.devs = [i for i in range(openvr.k_unMaxTrackedDeviceCount)
                         if self.vrsys.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller]
        self.frame += 1
        out = []
        for i in self.devs:
            p = poses[i]
            if not (p.bDeviceIsConnected and p.bPoseIsValid):
                continue
            ok, state = self.vrsys.getControllerState(i)
            mask = int(state.ulButtonPressed) if ok else 0
            trigger = bool(mask & (1 << openvr.k_EButton_SteamVR_Trigger)) or state.rAxis[1].x > 0.6
            new = mask & ~self.prev.get(i, 0)
            self.prev[i] = mask
            if self.debug and new:
                bits = [b for b in range(64) if new >> b & 1]
                print(f"\n[input] controller {i} role {self.vrsys.getControllerRoleForTrackedDeviceIndex(i)}"
                      f" pressed button ids {bits}")
            out.append(dict(index=i, role=self.vrsys.getControllerRoleForTrackedDeviceIndex(i),
                            pose=m34(p.mDeviceToAbsoluteTracking), trigger=trigger, mask=mask, new=new))
        return out


# ---------------------------------------------------------------- the app
class App:
    def __init__(self, args):
        self.args = args
        self.base = dict(depth_cm=args.depth_cm, conv=args.convergence, fov=args.fov,
                         distance=args.distance, height=-args.drop, sharpen=args.sharpen,
                         stereo3d=True, swap=False, debug=False, env_name="", env_layouts={},
                         env_brightness=1.0, env_yaw=0.0, env_play=True, pointer_tilt=args.pointer_tilt)
        self.st = copy.deepcopy(self.base)
        self.st.update({k: v for k, v in load_saved().items() if k in self.base})

        self.grabber = Grabber(args.window, args.source_monitor, cursor=not args.no_cursor)
        self.worker = DepthWorker(self.grabber, args.depth_width, args.smooth, MODELS[args.model],
                                  args.cut_threshold)
        self.worker.start()

        self._init_gl()
        self._init_vr()
        self._init_render()

        self.env = EnvManager(self.ctx, ENV_DIR)
        self.env.paused = not self.st["env_play"]
        if self.st["env_name"] in self.env.names():
            self.env.select(self.st["env_name"])
        else:
            self.st["env_name"] = ""

        self.panel = Panel(self.st, self.on_action)
        self.ui_tex = self.ctx.texture((PW, PH), 4)
        self.ui_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.ui_open = False
        self.panel_model = np.eye(4)
        self.panel_inv = np.eye(4)
        self.rays = []

        self.head = np.eye(4)
        self.anchor_pos = np.zeros(3)
        self.anchor_yaw = 0.0
        self.need_recenter = True
        self.prev_keys = {}
        self.dirty = True
        self.running = True

    # ------------------------------------------------------------------ setup
    def _init_gl(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.win = glfw.create_window(64, 64, "depth3d_vr", None, None)
        glfw.make_context_current(self.win)
        glfw.swap_interval(0)
        self.ctx = moderngl.create_context()

    def _init_vr(self):
        self.vrsys = openvr.init(openvr.VRApplication_Scene)
        self.comp = openvr.VRCompositor()
        try:
            ipd = float(self.vrsys.getFloatTrackedDeviceProperty(
                openvr.k_unTrackedDeviceIndex_Hmd, openvr.Prop_UserIpdMeters_Float))
        except Exception:                               # noqa: BLE001
            ipd = 0.063
        self.ipd = ipd if 0.04 < ipd < 0.09 else 0.063

        tw, th = self.vrsys.getRecommendedRenderTargetSize()
        self.tw, self.th = int(tw * self.args.render_scale), int(th * self.args.render_scale)
        print(f"Per-eye render target: {self.tw}x{self.th}")
        self.targets = []
        for _ in range(2):
            tex = self.ctx.texture((self.tw, self.th), 4)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.targets.append((tex, self.ctx.framebuffer(color_attachments=[tex])))

        eyes = (openvr.Eye_Left, openvr.Eye_Right)
        self.proj = [m44(self.vrsys.getProjectionMatrix(e, 0.1, 200.0)) for e in eyes]
        self.inv_e2h = [np.linalg.inv(m34(self.vrsys.getEyeToHeadTransform(e))) for e in eyes]
        self.controllers = Controllers(self.vrsys, self.args.debug_input)

    def _init_render(self):
        ctx = self.ctx
        self.prog = ctx.program(vertex_shader=SCREEN_VERT, fragment_shader=SCREEN_FRAG)
        self.prog["color_tex"].value = 0
        self.prog["depth_tex"].value = 1
        self.prog["samples"].value = max(4, self.args.samples)
        verts, self.screen_w = build_screen(16 / 9, self.st["distance"], self.st["fov"])
        self.vbo = ctx.buffer(verts.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "3f 2f", "pos", "tc")])
        self.mesh_key = None

        # depth refinement passes
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self.prog_ref = ctx.program(vertex_shader=FS_VERT, fragment_shader=REFINE_FRAG)
        self.prog_dil = ctx.program(vertex_shader=FS_VERT, fragment_shader=DILATE_FRAG)
        self.vao_ref = ctx.vertex_array(self.prog_ref, [(quad, "2f", "pos")])
        self.vao_dil = ctx.vertex_array(self.prog_dil, [(quad, "2f", "pos")])
        self.prog_ref["color_tex"].value = 0
        self.prog_ref["depth_tex"].value = 1
        self.prog_ref["sigma_c"].value = self.args.edge_sigma
        self.prog_dil["src"].value = 2
        self.prog_dil["radius"].value = self.args.dilate

        self.S = dict(color_tex=None, color_size=None, depth_size=(2, 2), ref=None)
        self.S["depth_tex"] = make_tex(ctx, 2, 2, 1, "f4", np.full((2, 2), 0.5, dtype="f4").tobytes())
        self.S["final_depth"] = self.S["depth_tex"]
        self.last_cseq = self.last_dseq = -1

        # settings panel + laser pointers
        self.prog_panel = ctx.program(vertex_shader=PANEL_VERT, fragment_shader=PANEL_FRAG)
        self.prog_panel["ui_tex"].value = 4
        w, h = PANEL_W_M / 2, PANEL_H_M / 2
        pv = np.array([-w, h, 0, 0, 0, -w, -h, 0, 0, 1, w, h, 0, 1, 0, w, -h, 0, 1, 1], dtype="f4")
        self.vao_panel = ctx.vertex_array(self.prog_panel,
                                          [(ctx.buffer(pv.tobytes()), "3f 2f", "pos", "tc")])
        self.prog_flat = ctx.program(vertex_shader=FLAT_VERT, fragment_shader=FLAT_FRAG)
        self.dyn = ctx.buffer(reserve=2048, dynamic=True)
        self.vao_flat = ctx.vertex_array(self.prog_flat, [(self.dyn, "3f", "pos")])

    # ------------------------------------------------------------------ capture textures
    def _make_refine_targets(self, cw, ch):
        if self.S["ref"] is not None:
            for obj in self.S["ref"]:
                obj.release()
            self.S["ref"] = None
        if self.args.refine_width <= 0:
            return
        rw = min(cw, self.args.refine_width)
        rh = max(1, int(round(ch * rw / cw)))
        ta, tb = make_tex(self.ctx, rw, rh, 1, "f4"), make_tex(self.ctx, rw, rh, 1, "f4")
        self.S["ref"] = (ta, self.ctx.framebuffer(color_attachments=[ta]),
                         tb, self.ctx.framebuffer(color_attachments=[tb]))
        self.prog_dil["texel"].value = (1.0 / rw, 1.0 / rh)

    def upload_color(self, frame):
        S = self.S
        fh, fw = frame.shape[:2]
        if S["color_size"] != (fw, fh):
            if S["color_tex"] is not None:
                S["color_tex"].release()
            t = make_tex(self.ctx, fw, fh, 4, "f1")
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)   # crisp text when minified
            t.anisotropy = 16.0
            t.swizzle = "BGRA"
            S["color_tex"], S["color_size"] = t, (fw, fh)
            self._make_refine_targets(fw, fh)
        S["color_tex"].write(np.ascontiguousarray(frame))
        S["color_tex"].build_mipmaps()

    def upload_depth(self, depth):
        S = self.S
        dh, dw = depth.shape
        if S["depth_size"] != (dw, dh):
            S["depth_tex"].release()
            S["depth_tex"] = make_tex(self.ctx, dw, dh, 1, "f4")
            S["depth_size"] = (dw, dh)
            self.prog_ref["depth_texel"].value = (1.0 / dw, 1.0 / dh)
        S["depth_tex"].write(depth.tobytes())

    def refine(self):
        S = self.S
        if S["ref"] is None:
            return S["depth_tex"]
        ta, fa, tb, fb = S["ref"]
        S["color_tex"].use(0)
        S["depth_tex"].use(1)
        fa.use()
        self.vao_ref.render(moderngl.TRIANGLE_STRIP)
        if self.args.dilate > 0:
            ta.use(2)
            fb.use()
            self.vao_dil.render(moderngl.TRIANGLE_STRIP)
            return tb
        return ta

    # ------------------------------------------------------------------ actions / UI
    def layout_for_env(self):
        name = self.st["env_name"]
        return self.st["env_layouts"].get(name) or self.env.guessed

    def cycle_env(self, step):
        names = self.env.names()
        i = names.index(self.st["env_name"]) if self.st["env_name"] in names else 0
        self.st["env_name"] = names[(i + step) % len(names)]
        self.env.select(self.st["env_name"])
        self.panel.invalidate()

    def on_action(self, name):
        if name == "save":
            save_settings(self.st)
            self.panel.toast("Saved")
        elif name == "reset":
            self.st.clear()
            self.st.update(copy.deepcopy(self.base))
            self.env.select("")
            self.panel.toast("Reset to launcher defaults (Save to keep)")
            self.panel.invalidate()
        elif name == "close":
            self.ui_open = False
        elif name == "recenter":
            self.need_recenter = True
        elif name == "env_prev":
            self.cycle_env(-1)
        elif name == "env_next":
            self.cycle_env(+1)
        elif name == "env_layout":
            cur = self.layout_for_env()
            nxt = LAYOUTS[(LAYOUTS.index(cur) + 1) % len(LAYOUTS)] if cur in LAYOUTS else LAYOUTS[0]
            self.st["env_layouts"][self.st["env_name"]] = nxt
            self.panel.invalidate()
        elif name == "toggled:env_play":
            self.env.paused = not self.st["env_play"]
        self.dirty = True

    def toggle_ui(self):
        self.ui_open = not self.ui_open
        if self.ui_open:
            yaw = head_yaw(self.head)
            fwd = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
            pos = self.head[:3, 3] + fwd * UI_DISTANCE + np.array([0.0, -0.15, 0.0])
            self.panel_model = translate(*pos) @ rot_y(yaw)
            self.panel_inv = np.linalg.inv(self.panel_model)
            self.panel.invalidate()

    def update_ui(self, ctrls):
        kb_click = down("CTRL") and down("ALT") and down("RETURN")
        best = None
        self.rays = []
        for c in sorted(ctrls, key=lambda c: c["role"] != openvr.TrackedControllerRole_RightHand):
            o, d = pointer_ray(c["pose"], self.st["pointer_tilt"])
            trig = c["trigger"] or (kb_click and best is None)
            hit = ray_panel_hit(o, d, self.panel_inv)
            self.rays.append((o, d, hit[0] if hit else None, trig))
            if hit and (best is None or (trig and not best[1])):
                best = (hit, trig)
        if best:
            (_, px, py), trig = best
            self.panel.pointer(px, py, trig)
        else:
            self.panel.pointer(None, None, False)
        data = self.panel.render_if_dirty()
        if data is not None:
            self.ui_tex.write(data)

    # ------------------------------------------------------------------ hotkeys
    def handle_keys(self, dt):
        st = self.st
        chord = down("CTRL") and down("ALT")
        if chord:
            def adjust(name, sign):
                lo, hi = LIMITS[name]
                st[name] = min(hi, max(lo, st[name] + sign * RATES[name] * dt))
                self.dirty = True
            for key, name, sign in (("UP", "depth_cm", 1), ("DOWN", "depth_cm", -1),
                                    ("RIGHT", "conv", 1), ("LEFT", "conv", -1),
                                    ("PGUP", "fov", 1), ("PGDN", "fov", -1)):
                if down(key):
                    adjust(name, sign)
                    if self.ui_open:
                        self.panel.invalidate()
        for key, action in (("HOME", "recenter"), ("END", "swap"), ("INS", "debug"),
                            ("SPACE", "ui"), ("DEL", "3d")):
            cur = chord and down(key)
            if cur and not self.prev_keys.get(key):
                if action == "recenter":
                    self.need_recenter = True
                elif action == "ui":
                    self.toggle_ui()
                elif action == "3d":
                    st["stereo3d"] = not st["stereo3d"]
                else:
                    st[action] = not st[action]
                self.dirty = True
                if self.ui_open:
                    self.panel.invalidate()
            self.prev_keys[key] = cur

    # ------------------------------------------------------------------ rendering
    def draw_lasers(self, vp):
        if not self.rays:
            return
        head_pos = self.head[:3, 3]
        right, up = self.panel_model[:3, 0], self.panel_model[:3, 1]
        normal = self.panel_model[:3, 2]
        for o, d, t, trig in self.rays:
            length = t if t is not None else 1.2
            e = o + d * length
            side = np.cross(d, head_pos - o)
            n = np.linalg.norm(side)
            if n < 1e-6:
                continue
            side = side / n * 0.003
            quad = np.array([o + side, o - side, e + side, e - side], dtype="f4")
            self.prog_flat["mvp"].write(vp.T.astype("f4").tobytes())
            self.prog_flat["color"].value = (1.0, 0.85, 0.3, 0.9) if trig else (0.4, 0.75, 1.0, 0.75)
            self.dyn.write(quad.tobytes())
            self.vao_flat.render(moderngl.TRIANGLE_STRIP, vertices=4)
            if t is not None:
                c = e + normal * 0.004
                ring = [c] + [c + (math.cos(a) * right + math.sin(a) * up) * 0.013
                              for a in np.linspace(0, 2 * math.pi, 15)]
                self.prog_flat["color"].value = (1.0, 1.0, 1.0, 0.95)
                self.dyn.write(np.array(ring, dtype="f4").tobytes())
                self.vao_flat.render(moderngl.TRIANGLE_FAN, vertices=len(ring))

    def render_eyes(self):
        st, S, env = self.st, self.S, self.env
        have_color = S["color_tex"] is not None
        if have_color:
            S["color_tex"].use(0)
            S["final_depth"].use(1)
        sign = -1.0 if st["swap"] else 1.0
        strength = (0.5 * (st["depth_cm"] / 100.0) / self.screen_w) if st["stereo3d"] else 0.0
        self.prog["strength"].value = strength
        self.prog["convergence"].value = st["conv"]
        self.prog["sharpen"].value = st["sharpen"]
        self.prog["show_depth"].value = 1 if (st["debug"] and st["stereo3d"]) else 0

        screen_model = (translate(self.anchor_pos[0], self.anchor_pos[1] + st["height"], self.anchor_pos[2])
                        @ rot_y(self.anchor_yaw))
        view_base = np.linalg.inv(self.head)
        head_rot_inv = np.linalg.inv(self._head_rot())
        env_yaw = self.anchor_yaw + math.radians(st["env_yaw"])
        env_layout = self.layout_for_env()

        for i in range(2):
            tex, fbo = self.targets[i]
            fbo.use()
            self.ctx.viewport = (0, 0, self.tw, self.th)
            self.ctx.clear(0.0, 0.0, 0.0)
            eye = -1.0 if i == 0 else 1.0
            if env.ready:
                env.draw(self.proj[i] @ self.inv_e2h[i] @ head_rot_inv, eye, env_layout,
                         env_yaw, st["env_brightness"])
            vp = self.proj[i] @ self.inv_e2h[i] @ view_base
            if have_color:
                self.prog["mvp"].write((vp @ screen_model).T.astype("f4").tobytes())
                self.prog["eye"].value = eye * sign
                self.vao.render(moderngl.TRIANGLE_STRIP)
            if self.ui_open:
                self.ctx.enable(moderngl.BLEND)
                self.ui_tex.use(4)
                self.prog_panel["mvp"].write((vp @ self.panel_model).T.astype("f4").tobytes())
                self.vao_panel.render(moderngl.TRIANGLE_STRIP)
                self.draw_lasers(vp)
                self.ctx.disable(moderngl.BLEND)
        self.ctx.finish()
        for i, eye in enumerate((openvr.Eye_Left, openvr.Eye_Right)):
            t = openvr.Texture_t()
            t.handle = int(self.targets[i][0].glo)
            t.eType = openvr.TextureType_OpenGL
            t.eColorSpace = openvr.ColorSpace_Gamma
            self.comp.submit(eye, t)

    def _head_rot(self):
        r = self.head.copy()
        r[:3, 3] = 0.0
        return r

    # ------------------------------------------------------------------ main loop
    def run(self):
        args, st, S = self.args, self.st, self.S
        poses = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
        event = openvr.VREvent_t()
        last_t, last_print, mesh_key = time.time(), 0.0, None
        print("\nRunning. Press MENU or A on a controller (or hold Ctrl+Alt and press Space) "
              "to open the settings panel.")
        while self.running:
            self.comp.waitGetPoses(poses, None)
            while self.vrsys.pollNextEvent(event):
                if event.eventType == openvr.VREvent_Quit:
                    self.running = False

            now = time.time()
            dt = min(now - last_t, 0.1)
            last_t = now

            hp = poses[openvr.k_unTrackedDeviceIndex_Hmd]
            if hp.bPoseIsValid:
                self.head = m34(hp.mDeviceToAbsoluteTracking)
                if self.need_recenter:
                    self.anchor_pos = self.head[:3, 3].copy()
                    self.anchor_yaw = head_yaw(self.head)
                    self.need_recenter = False

            self.handle_keys(dt)
            ctrls = self.controllers.poll(poses)
            for c in ctrls:
                if any(c["new"] >> b & 1 for b in (openvr.k_EButton_ApplicationMenu, openvr.k_EButton_A)):
                    self.toggle_ui()
                    break
            if self.ui_open:
                self.panel.set_info(env_name=st["env_name"], env_layout=self.layout_for_env(),
                                    env_video=self.env.is_video,
                                    env_status="Loading..." if self.env.loading else self.env.error,
                                    ipd=self.ipd)
                self.update_ui(ctrls)

            self.worker.active = st["stereo3d"]
            self.env.update()

            # ---- upload newest color / depth
            updated = False
            if args.sync and st["stereo3d"]:
                depth, frame, dseq = self.worker.latest_pair()
                if depth is not None and frame is not None and dseq != self.last_dseq:
                    self.last_dseq = dseq
                    self.upload_color(frame)
                    self.upload_depth(depth)
                    updated = True
            else:
                frame, cseq = self.grabber.latest()
                if frame is not None and cseq != self.last_cseq:
                    self.last_cseq = cseq
                    self.upload_color(frame)
                    updated = True
                depth, dseq = self.worker.latest()
                if st["stereo3d"] and depth is not None and dseq != self.last_dseq:
                    self.last_dseq = dseq
                    self.upload_depth(depth)
                    updated = True
            if updated and S["color_tex"] is not None and st["stereo3d"]:
                S["final_depth"] = self.refine()

            # ---- screen mesh (rebuild if aspect / size / distance changed)
            aspect = (S["color_size"][0] / S["color_size"][1]) if S["color_size"] else 16 / 9
            key = (round(aspect, 4), round(st["fov"], 1), round(st["distance"], 2))
            if key != mesh_key:
                mesh_key = key
                verts, self.screen_w = build_screen(aspect, st["distance"], st["fov"])
                self.vbo.write(verts.tobytes())

            self.render_eyes()

            if self.dirty and now - last_print > 0.4:
                last_print, self.dirty = now, False
                far_cm = st["depth_cm"] * st["conv"]
                warn = ("  !! far parts exceed eye spacing - lower depth or convergence"
                        if far_cm / 100.0 > self.ipd and st["stereo3d"] else "")
                print(f"\r3D {'on ' if st['stereo3d'] else 'off'} | depth {st['depth_cm']:.1f} cm | "
                      f"conv {st['conv']:.2f} | screen {st['fov']:.0f} deg | "
                      f"depth fps {self.worker.fps:.0f}{warn}        ", end="", flush=True)

    def shutdown(self):
        print("\nShutting down...")
        if not self.args.no_autosave:
            try:
                save_settings(self.st)
            except Exception as e:                       # noqa: BLE001
                print("Could not save settings:", e)
        self.env.stop()
        self.worker.stop_evt.set()
        self.grabber.stop()
        openvr.shutdown()
        glfw.terminate()


def main():
    ap = argparse.ArgumentParser(description="SteamVR 2D->3D depth player")
    ap.add_argument("--window", help="Title (or part of it) of the window to capture")
    ap.add_argument("--source-monitor", type=int, default=None,
                    help="Capture a whole monitor (1 = main monitor)")
    ap.add_argument("--list", action="store_true", help="List window titles and exit")
    ap.add_argument("--model", choices=list(MODELS), default="large")
    ap.add_argument("--depth-width", type=int, default=700)
    ap.add_argument("--smooth", type=float, default=0.5,
                    help="Depth temporal blend in calm shots (1 = off, lower = smoother)")
    ap.add_argument("--cut-threshold", type=float, default=0.12,
                    help="Scene-cut sensitivity (lower = detects more cuts)")
    ap.add_argument("--sync", action=argparse.BooleanOptionalAction, default=True,
                    help="Show each video frame together with its own depth map (best for movies)")
    ap.add_argument("--samples", type=int, default=24,
                    help="Warp search samples per pixel (quality vs GPU cost)")
    ap.add_argument("--refine-width", type=int, default=1920, help="Depth refinement resolution (0 = off)")
    ap.add_argument("--edge-sigma", type=float, default=0.10, help="Edge-snap strictness")
    ap.add_argument("--dilate", type=float, default=1.0, help="Grow foreground depth by N pixels (0 = off)")
    ap.add_argument("--sharpen", type=float, default=0.5, help="Text sharpening 0..1 (initial value)")
    ap.add_argument("--depth-cm", type=float, default=6.0, help="Total disparity range in cm (initial)")
    ap.add_argument("--convergence", type=float, default=0.4, help="(initial)")
    ap.add_argument("--distance", type=float, default=3.0, help="Screen distance in metres (initial)")
    ap.add_argument("--fov", type=float, default=70.0, help="Screen width in degrees (initial)")
    ap.add_argument("--drop", type=float, default=0.0, help="Lower the screen below eye level, metres")
    ap.add_argument("--render-scale", type=float, default=1.0)
    ap.add_argument("--pointer-tilt", type=float, default=0.0,
                    help="Tilt the controller laser up(+)/down(-) in degrees if it points off")
    ap.add_argument("--no-cursor", action="store_true", help="Do not capture the mouse cursor")
    ap.add_argument("--no-autosave", action="store_true", help="Do not save settings on exit")
    ap.add_argument("--debug-input", action="store_true", help="Print controller button ids as you press them")
    args = ap.parse_args()

    if args.list:
        print("Windows:")
        for t in sorted(set(list_windows())):
            print("  ", t)
        if glfw.init():
            print("Monitors (use the 'source' number with --source-monitor):")
            for i, m in enumerate(glfw.get_monitors()):
                vm = glfw.get_video_mode(m)
                x, y = glfw.get_monitor_pos(m)
                print(f"   source={i + 1}  {glfw.get_monitor_name(m).decode(errors='ignore')}"
                      f"  {vm.size.width}x{vm.size.height} at ({x},{y})")
            glfw.terminate()
        return
    if not args.window and args.source_monitor is None:
        ap.error("give --window TITLE or --source-monitor N")

    try:
        app = App(args)
    except Exception as e:                               # noqa: BLE001
        print("\nCould not start:", e)
        print("Start SteamVR and connect the Frame (Steam Link -> PC VR) first.")
        raise
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
