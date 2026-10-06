"""Headless Blender multiplane: stacked cel layers, one moving camera.

Each layer is a plane sized to fill the camera's view exactly at its own
distance, so frame 0 reproduces the flat panel pixel for pixel. When the
camera moves, near planes slide further than far ones - parallax, the whole
effect. Nothing is redrawn: a face can tear at a layer edge but can never
change, which is the opposite failure to a video model's.

Colour is kept exact: Standard view transform, emission-only shading. Blender's
default AgX transform would shift every colour of the art.

usage:
  blender -b --factory-startup --python multiplane.py -- <layersDir> <out.mp4>
          [--seconds 4] [--fps 24] [--scale 0.75] [--move push|pan|drift]
"""
import argparse, json, math, sys, time
from pathlib import Path
import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("layers"); ap.add_argument("out")
ap.add_argument("--seconds", type=float, default=4.0)
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--scale", type=float, default=0.75, help="render size as a fraction of the panel")
ap.add_argument("--move", default="push")
ap.add_argument("--overscan", type=float, default=1.12)
ap.add_argument("--near", type=float, default=4.0); ap.add_argument("--far", type=float, default=12.0)
a = ap.parse_args(argv)

src = Path(a.layers).resolve()  # Blender resolves a relative image path against the .blend, not the shell
meta = json.loads((src / "layers.json").read_text())
W, H = meta["w"], meta["h"]

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.resolution_x = int(round(W * a.scale / 2)) * 2
sc.render.resolution_y = int(round(H * a.scale / 2)) * 2
sc.render.fps = a.fps
sc.frame_start, sc.frame_end = 1, int(round(a.seconds * a.fps))
for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
    try:
        sc.render.engine = eng
        break
    except TypeError:
        continue
sc.view_settings.view_transform = "Standard"
sc.view_settings.look = "None"
sc.world = bpy.data.worlds.new("w"); sc.world.color = (1, 1, 1)

cam_data = bpy.data.cameras.new("cam"); cam_data.lens = 50; cam_data.sensor_fit = "VERTICAL"; cam_data.sensor_height = 24
cam = bpy.data.objects.new("cam", cam_data); sc.collection.objects.link(cam); sc.camera = cam
fov_y = 2 * math.atan(cam_data.sensor_height / 2 / cam_data.lens)
aspect = W / H

# Camera moves, kept small: a displaced cut-out tears if pushed far past its own edge.
moves = {
    "push":  ((0, 0, 0), (0.0, 0.0, -0.9)),
    "pan":   ((-0.35, 0, 0), (0.35, 0, 0)),
    "drift": ((-0.2, -0.08, 0), (0.2, 0.08, -0.5)),
    # a figure near the top edge: rise with her while pushing in, or the push crops the face
    # (v2 opening: Lirazel's head left the frame by 3 s on "push")
    "rise":  ((0, -0.05, 0), (0, 0.14, -0.45)),
}
p0, p1 = moves[a.move]

n = len(meta["layers"])
zs = [1 / ((1 - t) / a.far + t / a.near) for t in (i / max(1, n - 1) for i in range(n))]
# v2: overscan. A plane that exactly fills the view at rest shows its edge as soon as the
# camera moves (v1: grey bars on the pan and the drift). Multiplane backgrounds were always
# painted wider than the frame; frame 0 is now a slight crop of the panel, not all of it.
# v3 (2026-09-29, whole chapter): a flat 12% was not enough for a PAN on a wide panel - the
# nearest plane's right edge showed as a pale strip at the end of 01_c01_erl. Raising the crop
# to what the move needs cost up to 31% of a tall panel (faces sit near its edges). So the crop
# stays at --overscan, and each plane's GEOMETRY is extended by what the move still needs at
# the nearest plane (+3%), with the image mapped back to its original size and the texture set
# to MIRROR: a revealed edge shows the picture reflected (a short seam) instead of a gap.
# One factor for every plane, so frame 0 stays registered.
# v3b (same day): the mirror is right for SCENERY and wrong for a FIGURE cut off by the frame -
# the dragon's layer touched the edge in 22_f07_claws and 27_f12b_standoff, and the drift showed
# its mirror image: a ghost second dragon. So: if any non-backdrop layer is opaque along an edge
# the camera moves toward, that edge must never be revealed - crop up to CAP instead, and if even
# that is not enough, shrink the move. Otherwise keep the gentle crop + mirrored extension.
import numpy as np  # bundled with Blender

CAP = 1.20


def touches(path, axis):
    im = bpy.data.images.load(str(path)); w_, h_ = im.size
    px = np.empty(w_ * h_ * 4, dtype=np.float32); im.pixels.foreach_get(px)
    alpha = px.reshape(h_, w_, 4)[:, :, 3]; bpy.data.images.remove(im)
    k = max(2, int(0.02 * (w_ if axis == "x" else h_)))
    bands = (alpha[:, :k], alpha[:, -k:]) if axis == "x" else (alpha[:k, :], alpha[-k:, :])
    return any((b > 0.5).mean() > 0.02 for b in bands)


dx = max(abs(p0[0]), abs(p1[0])); dy = max(abs(p0[1]), abs(p1[1]))
half_h = min(zs) * math.tan(fov_y / 2); half_w = half_h * aspect
need = max(1.03 * (1 + dx / half_w), 1.03 * (1 + dy / half_h))
edge_fig = any(touches(src / L["file"], ax) for L in meta["layers"][1:] for ax, d in (("x", dx), ("y", dy)) if d > 0)
if edge_fig and need > a.overscan:
    crop = min(need, CAP)
    f = min([1.0] + [(CAP / 1.03 - 1) / (d / hw) for d, hw in ((dx, half_w), (dy, half_h)) if d > 0])
    p0, p1 = tuple(v * f for v in p0), tuple(v * f for v in p1)
    geo = 1.0
    print(f"MULTIPLANE a figure touches the frame edge: crop {crop:.3f}, move x{f:.2f}, no extension")
else:
    crop, geo = a.overscan, max(1.0, need / a.overscan)
    print(f"MULTIPLANE crop {crop:.3f}, mirrored edge extension x{geo:.3f} for move {a.move}")
for i, L in enumerate(meta["layers"]):
    # far layer at `far`, near layer at `near`, spaced in inverse depth (what the model predicts)
    z = zs[i]
    h = 2 * z * math.tan(fov_y / 2) * crop * geo; w = h * aspect
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, -z))
    pl = bpy.context.active_object; pl.name = f"layer_{i}"; pl.scale = (w, h, 1)
    mat = bpy.data.materials.new(f"m{i}"); mat.use_nodes = True
    try:
        mat.surface_render_method = "BLENDED"
    except AttributeError:
        mat.blend_method = "BLEND"
    nt = mat.node_tree; nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = bpy.data.images.load(str(src / L["file"])); tex.interpolation = "Cubic"
    # MIRROR, not EXTEND: a revealed edge then continues the picture (more hedge) instead of
    # repeating one column of pixels as stripes (01_c01_erl test, 2026-09-29)
    try:
        tex.extension = "MIRROR"
    except TypeError:
        tex.extension = "EXTEND"
    if geo > 1.0:  # uv' = (uv - 0.5) * geo + 0.5: the image keeps its size on the bigger plane
        tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (geo, geo, 1)
        mp.inputs["Location"].default_value = (0.5 * (1 - geo), 0.5 * (1 - geo), 0)
        nt.links.new(tc.outputs["UV"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
    emi = nt.nodes.new("ShaderNodeEmission"); tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader"); outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], emi.inputs["Color"])
    nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1]); nt.links.new(emi.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], outn.inputs["Surface"])
    pl.data.materials.append(mat)

for f, p in ((sc.frame_start, p0), (sc.frame_end, p1)):
    cam.location = p; cam.keyframe_insert("location", frame=f)
# Blender 5 moved keyframes into layered actions; ease via the default Bezier interpolation.

out = Path(a.out).resolve()
try:
    sc.render.image_settings.media_type = "VIDEO"
except (AttributeError, TypeError):
    pass
sc.render.image_settings.file_format = "FFMPEG"
sc.render.ffmpeg.format = "MPEG4"; sc.render.ffmpeg.codec = "H264"; sc.render.ffmpeg.constant_rate_factor = "HIGH"
sc.render.filepath = str(out)
t0 = time.perf_counter()
bpy.ops.render.render(animation=True)
secs = time.perf_counter() - t0
frames = sc.frame_end - sc.frame_start + 1
rec = {"out": str(out), "move": a.move, "crop": round(crop, 3), "edgeExtension": round(geo, 3), "figureAtEdge": bool(edge_fig), "moveScale": round(p1[0] / moves[a.move][1][0], 3) if moves[a.move][1][0] else (round(p1[2] / moves[a.move][1][2], 3) if moves[a.move][1][2] else 1.0), "frames": frames, "fps": a.fps, "res": [sc.render.resolution_x, sc.render.resolution_y],
       "renderSec": round(secs, 1), "secPerVideoSec": round(secs / (frames / a.fps), 2), "engine": sc.render.engine}
print("MULTIPLANE", json.dumps(rec))
(out.parent / (out.stem + ".json")).write_text(json.dumps(rec, indent=1))
