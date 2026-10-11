"""GLSL shaders for depth3d_vr.py (kept separate so they can be tested on their own)."""

# ---------------------------------------------------------------- the 3D screen
SCREEN_VERT = """
#version 330
uniform mat4 mvp;
in vec3 pos;
in vec2 tc;
out vec2 uv;
void main() {
    uv = tc;
    gl_Position = mvp * vec4(pos, 1.0);
}
"""

# For every output pixel, SEARCH along the scan line for the source pixel that lands here
# (nearest one wins, like a z-buffer). Revealed gaps are filled from the far side.
# Mip-mapped + anisotropic sampling keeps text readable; optional sharpening on top.
SCREEN_FRAG = """
#version 330
uniform sampler2D color_tex;
uniform sampler2D depth_tex;
uniform float strength;      // per-eye uv shift per unit of (depth - convergence)
uniform float convergence;
uniform float eye;           // -1 left, +1 right
uniform float sharpen;
uniform int samples;
uniform int show_depth;
in vec2 uv;
out vec4 fragColor;
void main() {
    vec2 gx = dFdx(uv);      // footprint of one output pixel -> picks the right mip level
    vec2 gy = dFdy(uv);
    if (show_depth == 1) {
        fragColor = vec4(vec3(textureLod(depth_tex, uv, 0.0).r), 1.0);
        return;
    }
    float sxs = uv.x;
    if (strength > 1e-7) {
        float R = strength * max(convergence, 1.0 - convergence);
        float stp = 2.0 * R / float(samples - 1);
        float tol = 0.75 * stp;
        float gapmax = strength;
        float best_d = -1.0, fb_score = 1e9, fb_sx = uv.x;
        for (int i = 0; i < samples; i++) {
            float sx = uv.x - R + stp * float(i);
            float d = textureLod(depth_tex, vec2(sx, uv.y), 0.0).r;
            float r = abs(sx - eye * strength * (d - convergence) - uv.x);
            if (r <= tol && d > best_d) best_d = d;
            float score = r / gapmax + 2.0 * d;        // in a gap: nearest FAR sample
            if (r <= gapmax && score < fb_score) { fb_score = score; fb_sx = sx; }
        }
        sxs = (best_d >= 0.0) ? uv.x + eye * strength * (best_d - convergence) : fb_sx;
    }
    vec2 p = vec2(sxs, uv.y);
    vec3 col = textureGrad(color_tex, p, gx, gy).rgb;
    if (sharpen > 0.0) {
        vec3 blur = 0.25 * (textureGrad(color_tex, p + gx, gx, gy).rgb +
                            textureGrad(color_tex, p - gx, gx, gy).rgb +
                            textureGrad(color_tex, p + gy, gx, gy).rgb +
                            textureGrad(color_tex, p - gy, gx, gy).rgb);
        col = clamp(col + sharpen * (col - blur), 0.0, 1.0);
    }
    fragColor = vec4(col, 1.0);
}
"""

# ---------------------------------------------------------------- depth refinement passes
FS_VERT = """
#version 330
in vec2 pos;
out vec2 uv;
void main() {
    uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

# Joint-bilateral upsampling: snaps soft low-res depth edges onto the real colour edges.
REFINE_FRAG = """
#version 330
uniform sampler2D color_tex;
uniform sampler2D depth_tex;
uniform vec2 depth_texel;   // 1 / raw depth size
uniform float sigma_c;      // colour similarity (smaller = harder edges)
in vec2 uv;
out vec4 fragColor;
void main() {
    vec3 c0 = texture(color_tex, uv).rgb;
    float acc = 0.0, wsum = 0.0;
    for (int j = -2; j <= 2; j++) {
        for (int i = -2; i <= 2; i++) {
            vec2 p = uv + vec2(float(i), float(j)) * depth_texel;
            vec3 dc = texture(color_tex, p).rgb - c0;
            float ws = exp(-float(i * i + j * j) / 4.5);
            float wc = exp(-dot(dc, dc) / (2.0 * sigma_c * sigma_c));
            float w = ws * wc;
            acc += w * texture(depth_tex, p).r;
            wsum += w;
        }
    }
    fragColor = vec4(acc / max(wsum, 1e-5), 0.0, 0.0, 1.0);
}
"""

# Tiny foreground dilation so the nearest depth fully covers an object's rim.
DILATE_FRAG = """
#version 330
uniform sampler2D src;
uniform vec2 texel;
uniform float radius;       // in pixels of the refined map
in vec2 uv;
out vec4 fragColor;
void main() {
    float m = texture(src, uv).r;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            m = max(m, texture(src, uv + vec2(float(i), float(j)) * texel * radius).r);
        }
    }
    fragColor = vec4(m, 0.0, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------- immersive environment
ENV_VERT = """
#version 330
uniform mat4 mvp;
in vec3 pos;
out vec3 dir;
void main() {
    dir = pos;
    gl_Position = mvp * vec4(pos * 80.0, 1.0);
}
"""

# Equirectangular 360 / 180 viewer, mono or stereo (side-by-side / over-under).
ENV_FRAG = """
#version 330
uniform sampler2D env_tex;
uniform int stereo;          // 0 mono, 1 side-by-side, 2 over-under
uniform int fov180;          // 0 = 360, 1 = 180 (front half only)
uniform float eye;           // -1 left, +1 right
uniform float yaw;           // radians: rotation of the environment about the vertical axis
uniform float brightness;
in vec3 dir;
out vec4 fragColor;
const float PI = 3.14159265358979;
void main() {
    vec3 d = normalize(dir);
    float c = cos(yaw), s = sin(yaw);
    vec3 e = vec3(d.x * c - d.z * s, d.y, d.x * s + d.z * c);   // world -> environment frame
    float lon = atan(e.x, -e.z);
    float lat = asin(clamp(e.y, -1.0, 1.0));
    float v = 0.5 - lat / PI;
    vec2 uvA = vec2(lon / (2.0 * PI) + 0.5, v);
    vec2 uvB = vec2(fract(lon / (2.0 * PI)), v);
    vec2 gx = dFdx(uvA), gy = dFdy(uvA);
    vec2 gxb = dFdx(uvB), gyb = dFdy(uvB);
    if (abs(gxb.x) + abs(gyb.x) < abs(gx.x) + abs(gy.x)) { gx = gxb; gy = gyb; }  // hide the wrap seam
    vec2 uv = uvA;
    float inside = 1.0;
    if (fov180 == 1) {
        uv.x = lon / PI + 0.5;
        gx.x *= 2.0; gy.x *= 2.0;
        inside = 1.0 - smoothstep(0.5 * PI - 0.03, 0.5 * PI, abs(lon));
    }
    vec2 scale = vec2(1.0), off = vec2(0.0);
    float right = step(0.0, eye);
    if (stereo == 1) { scale.x = 0.5; off.x = 0.5 * right; }
    if (stereo == 2) { scale.y = 0.5; off.y = 0.5 * right; }
    vec3 col = textureGrad(env_tex, uv * scale + off, gx * scale, gy * scale).rgb;
    fragColor = vec4(col * brightness * inside, 1.0);
}
"""

# ---------------------------------------------------------------- settings panel + laser pointers
PANEL_VERT = """
#version 330
uniform mat4 mvp;
in vec3 pos;
in vec2 tc;
out vec2 uv;
void main() {
    uv = tc;
    gl_Position = mvp * vec4(pos, 1.0);
}
"""

PANEL_FRAG = """
#version 330
uniform sampler2D ui_tex;
in vec2 uv;
out vec4 fragColor;
void main() {
    fragColor = texture(ui_tex, uv);
}
"""

FLAT_VERT = """
#version 330
uniform mat4 mvp;
in vec3 pos;
void main() {
    gl_Position = mvp * vec4(pos, 1.0);
}
"""

FLAT_FRAG = """
#version 330
uniform vec4 color;
out vec4 fragColor;
void main() {
    fragColor = color;
}
"""
