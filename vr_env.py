"""
Immersive environments for depth3d_vr: 360/180 degree images and videos,
mono or stereo (side-by-side / over-under), shown on a big sphere around you.

Drop files into the `environments` folder next to the script. Name hints
(optional, the in-VR menu can also cycle the layout):
    "..._180_sbs.mp4"  VR180 side-by-side        "..._360_ou.jpg"  360 over-under
    "..._360.jpg"      360 mono                  "..._tb", "_lr", "_3dh", "_3dv" also work
"""
import re
import threading
import time
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from vr_shaders import ENV_FRAG, ENV_VERT

Image.MAX_IMAGE_PIXELS = None

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}
LAYOUTS = ["360 mono", "360 OU", "360 SBS", "180 mono", "180 SBS", "180 OU"]
MAX_IMAGE_DIM = 8192
MAX_VIDEO_WIDTH = 4096


def scan(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT | VIDEO_EXT]
    return sorted(files, key=lambda p: p.name.lower())


def _tag(name, *words):
    pat = r"(^|[^a-z0-9])(" + "|".join(words) + r")([^a-z0-9]|$)"
    return re.search(pat, name) is not None


def guess_layout(name, w, h):
    """Best-effort layout from the file name, falling back to the aspect ratio."""
    n = name.lower()
    fov = 180 if "180" in n else (360 if "360" in n else None)
    if _tag(n, "sbs", "lr", "3dh", "lr180", "sbs180"):
        stereo = "SBS"
    elif _tag(n, "ou", "tb", "3dv", "overunder", "over_under", "topbottom"):
        stereo = "OU"
    elif _tag(n, "mono"):
        stereo = "mono"
    else:
        stereo = None
    a = w / max(h, 1)
    if fov is None:
        fov = 360
    if stereo is None:
        if fov == 180:
            stereo = "SBS" if a > 1.5 else "mono"
        else:
            stereo = "OU" if 0.75 < a < 1.35 else ("SBS" if a > 3.0 else "mono")
    return f"{fov} {stereo}"


def parse_layout(layout):
    fov180 = 1 if layout.startswith("180") else 0
    stereo = 1 if "SBS" in layout else (2 if "OU" in layout else 0)
    return fov180, stereo


def build_sphere(lat_n=48, lon_n=96):
    verts = []
    for i in range(lat_n + 1):
        lat = np.pi * (i / lat_n - 0.5)
        for j in range(lon_n + 1):
            lon = 2 * np.pi * (j / lon_n - 0.5)
            verts.append((np.cos(lat) * np.sin(lon), np.sin(lat), -np.cos(lat) * np.cos(lon)))
    idx = []
    for i in range(lat_n):
        for j in range(lon_n):
            a = i * (lon_n + 1) + j
            b = a + lon_n + 1
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return np.array(verts, "f4"), np.array(idx, "i4")


class EnvManager:
    def __init__(self, ctx, folder):
        self.ctx = ctx
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=True)
        self.files = scan(self.folder)

        self.prog = ctx.program(vertex_shader=ENV_VERT, fragment_shader=ENV_FRAG)
        v, i = build_sphere()
        self.vao = ctx.vertex_array(self.prog, [(ctx.buffer(v.tobytes()), "3f", "pos")],
                                    index_buffer=ctx.buffer(i.tobytes()))
        self.prog["env_tex"].value = 3

        self.tex = None
        self.size = None
        self.name = ""
        self.is_video = False
        self.guessed = "360 mono"
        self.loading = False
        self.error = ""
        self.paused = False

        self._lock = threading.Lock()
        self._pending = None
        self._pending_info = None
        self._gen = 0

    # ---------------------------------------------------------------- selection
    def names(self):
        return [""] + [p.name for p in self.files]

    def rescan(self):
        self.files = scan(self.folder)

    def select(self, name):
        """Start loading `name` ('' = no environment). Loading happens on a worker thread."""
        self._gen += 1
        gen = self._gen
        self.error = ""
        with self._lock:
            self._pending = None
        if not name:
            self.name = ""
            self.loading = False
            self.is_video = False
            self._release()
            return
        path = self.folder / name
        if not path.exists():
            self.name = ""
            self.error = f"{name} not found"
            return
        self.name = name
        self.loading = True
        self.is_video = path.suffix.lower() in VIDEO_EXT
        target = self._video_loop if self.is_video else self._load_image
        threading.Thread(target=target, args=(path, gen), daemon=True).start()

    def _release(self):
        if self.tex is not None:
            self.tex.release()
        self.tex = None
        self.size = None

    # ---------------------------------------------------------------- worker threads
    def _load_image(self, path, gen):
        try:
            img = Image.open(path)
            img.load()
            img = img.convert("RGB")
            w, h = img.size
            s = min(1.0, MAX_IMAGE_DIM / max(w, h))
            if s < 1.0:
                img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
            arr = np.ascontiguousarray(np.asarray(img))
            with self._lock:
                if gen == self._gen:
                    self._pending = arr
        except Exception as e:                      # noqa: BLE001
            if gen == self._gen:
                self.error = f"Could not load: {e}"
                self.loading = False

    def _video_loop(self, path, gen):
        try:
            import av
        except ImportError:
            self.error = "Videos need the 'av' package (run setup.bat)"
            self.loading = False
            return
        try:
            while gen == self._gen:
                container = av.open(str(path))
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                fps = float(stream.average_rate or 30)
                t0, n = time.time(), 0
                for frame in container.decode(stream):
                    if gen != self._gen:
                        break
                    while self.paused and gen == self._gen:
                        time.sleep(0.05)
                        t0 = time.time() - n / fps
                    w, h = frame.width, frame.height
                    s = min(1.0, MAX_VIDEO_WIDTH / w)
                    arr = frame.reformat(width=int(w * s) // 2 * 2, height=int(h * s) // 2 * 2,
                                         format="rgb24").to_ndarray()
                    with self._lock:
                        if gen == self._gen:
                            self._pending = arr
                    n += 1
                    delay = t0 + n / fps - time.time()
                    if delay > 0:
                        time.sleep(delay)
                    elif delay < -1.0:              # fell far behind: resync, don't spiral
                        t0, n = time.time(), 0
                container.close()
        except Exception as e:                      # noqa: BLE001
            if gen == self._gen:
                self.error = f"Video error: {e}"
                self.loading = False

    # ---------------------------------------------------------------- GL thread
    def update(self):
        """Upload a newly decoded image / video frame (call once per VR frame)."""
        with self._lock:
            arr, self._pending = self._pending, None
        if arr is None:
            return
        h, w = arr.shape[:2]
        if self.size != (w, h):
            self._release()
            self.tex = self.ctx.texture((w, h), 3, dtype="f1")
            self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.tex.repeat_x = True
            self.tex.repeat_y = False
            self.tex.anisotropy = 8.0
            self.size = (w, h)
            self.guessed = guess_layout(self.name, w, h)
        self.tex.write(arr)
        self.tex.build_mipmaps()
        self.loading = False

    @property
    def ready(self):
        return self.tex is not None and bool(self.name)

    def draw(self, mvp, eye, layout, yaw, brightness):
        fov180, stereo = parse_layout(layout)
        self.tex.use(3)
        self.prog["mvp"].write(mvp.T.astype("f4").tobytes())
        self.prog["stereo"].value = stereo
        self.prog["fov180"].value = fov180
        self.prog["eye"].value = eye
        self.prog["yaw"].value = yaw
        self.prog["brightness"].value = brightness
        self.vao.render()

    def stop(self):
        self._gen += 1
