"""Annotate a simulator part, or locate its contact point with a VLM fallback.

Point/verify/track adapted from showlab/Show-Harness at 101bd59d, consolidated here.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from string import Template

import numpy as np
from PIL import Image, ImageDraw


# These stages do not move toward a fixed scene point.
NON_SPATIAL_MOTIONS = frozenset(
    {"LIFT", "RETREAT", "WAIT", "RELEASE", "OPEN", "DONE", "REASON", "STILL", "PRESENT"}
)
CALL_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}, "reasoning": {"type": "string"}},
    "required": ["code", "reasoning"], "additionalProperties": False,
}
GRID_POINT_SCHEMA = {
    "type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
    "minItems": 2, "maxItems": 2,
}
POINT_SCHEMA = {
    "type": "object",
    "properties": {"points": {
        "type": "array", "maxItems": 1, "items": {
            "type": "object",
            "properties": {"part": {"type": "string"}, "why": {"type": "string"},
                           "point": GRID_POINT_SCHEMA},
            "required": ["part", "why", "point"], "additionalProperties": False,
        },
    }},
    "required": ["points"], "additionalProperties": False,
}
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"present": {"type": "boolean"}, "on_target": {"type": "boolean"},
                   "why": {"type": "string"}, "point": GRID_POINT_SCHEMA},
    "required": ["present", "on_target", "why", "point"], "additionalProperties": False,
}
VIEWS = (("agentview", "agentview_image"), ("robot0_eye_in_hand", "wrist_image"))


def execute_part_query(api, code):
    """Interpret one read-only API expression, without exec/eval or arbitrary access."""
    try:
        node = ast.parse(code, mode="eval").body
    except (SyntaxError, TypeError) as exc:
        raise ValueError("Expected a single sim.get_object_part(...) expression or None") from exc
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "sim"
            and node.func.attr == "get_object_part" and not node.args
            and len(node.keywords) == 2
            and {k.arg for k in node.keywords} == {"object_name", "part_name"}
            and all(isinstance(k.value, ast.Constant) and isinstance(k.value.value, str)
                    for k in node.keywords)):
        raise ValueError("Only sim.get_object_part(object_name='...', part_name='...') is supported")
    return api.get_object_part(**{k.arg: k.value.value for k in node.keywords})


def build(context):
    if context["plugins"].get("affordance") is not True:
        return ""
    record = context.get("affordance")
    if record is None:
        record = guide(context)
        context["affordance"] = record
    if record is None:
        return ""
    if record.get("source") == "vlm" and not any(record["points"].values()):
        return _text("none")
    views = [_text(name) for name, xy in record["pixels_xy"].items() if xy is not None]
    names = (Template(_text("both_views")).substitute(first=views[0], second=views[1])
             if len(views) == 2 else views[0] if views else "")
    visibility = Template(_text("visible")).substitute(views=names) if views else _text("outside")
    target = "grasp" if context["subgoal"].get("motion", "").upper() == "GRASP" else "target"
    if record.get("source") == "vlm":
        return Template(_text("vlm")).substitute(
            visibility=visibility, target=_text(target), part=record["part_name"],
        )
    part = record["part"]
    return Template(_text()).substitute(
        visibility=visibility, target=_text(target),
        object_name=part["object_name"], part_name=part["part_name"],
    )


def guide(context):
    """Select once per stage; update geometry or VLM points for each observation."""
    if context["plugins"].get("affordance") is not True:
        return None
    state = context["plugin_state"].setdefault("affordance", {})
    if context["subgoal"].get("motion", "").upper() in NON_SPATIAL_MOTIONS:
        state.clear()
        return None
    api = context.get("affordance_api") or state.get("api")
    if api is None:
        api = LiberoAffordanceAPI(context["env"], context)
    state["api"] = api
    if context["stage_key"] != state.get("stage_key"):
        prompt = Template(_text("selection")).substitute(
            task=context["task"], subgoal=json.dumps(context["subgoal"], sort_keys=True),
            catalog=json.dumps(api.catalog()),
        )
        query = _request(context, prompt, context["agentview_image"], CALL_SCHEMA,
                         operation="selection", reference=context["wrist_image"])
        try:
            selection = execute_part_query(api, query.get("code")) if query else None
            result = {"query": query, "result": selection}
        except ValueError as exc:
            selection = None
            result = {"query": query, "error": str(exc), "result": None}
        state.update(stage_key=context["stage_key"], selection=selection, query=query,
                     vlm_points={}, vlm_part="")
        trace = context.get("trace")
        if trace is not None:
            trace.event("Affordance API call", result)
    selection = state["selection"]
    if selection is None:
        return _guide_vlm(context, state)
    part = api.get_object_part(selection["object_name"], selection["part_name"])
    pixels = {}
    for name, field in VIEWS:
        frame = context[field]
        xy = api.project(part["world_position"], name, frame.shape) if frame is not None else None
        pixels[name] = xy
        if xy is None:
            continue
        context[field] = _draw_point(frame, xy, context)
    return {"source": "simulator", "query": state["query"], "part": part, "pixels_xy": pixels}


def _guide_vlm(context, state):
    """Track both views: Agentview can move too, so cached pixels need rechecking."""
    points, pixels, reference = {}, {}, None
    for name, field in VIEWS:
        frame = context[field]
        previous = state["vlm_points"].get(name)
        point = None
        if frame is not None:
            # Repeated identical confirmations can be the model echoing its own dot.
            if previous and previous["unchanged"] < 3:
                point = _verify(context, frame, name, previous, reference, operation="track")
                if point is not None:
                    point["unchanged"] = (previous["unchanged"] + 1
                                          if point["point"] == previous["point"] else 0)
            else:
                point = _locate(context, frame, name, state["vlm_part"], reference)
        points[name] = point
        pixels[name] = _pixels(point["point"], frame.shape) if point else None
        if point is not None:
            if not state["vlm_part"]:
                state["vlm_part"] = point["part"]
            context[field] = _draw_point(frame, pixels[name], context)
            if name == "agentview":
                reference = context[field]
    state["vlm_points"] = points
    record = {"source": "vlm", "query": state["query"], "part_name": state["vlm_part"],
              "points": points, "pixels_xy": pixels}
    if context.get("trace") is not None:
        context["trace"].event("Affordance VLM fallback", record)
    return record


def _locate(context, frame, view, part, reference):
    prompt = Template(_text("point")).substitute(
        task=context["task"], subgoal=json.dumps(context["subgoal"], sort_keys=True), view=_text(view),
        part=part or context["subgoal"].get("affordance", ""),
        reference=_text("reference") if reference is not None else "",
    )
    payload = _request(context, prompt, frame, POINT_SCHEMA, operation="locate",
                       view=view, reference=reference)
    items = payload.get("points") if payload is not None else None
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        return None
    candidate = items[0]
    grid = _grid_point(candidate.get("point"))
    chosen_part = candidate.get("part")
    if grid is None or not isinstance(chosen_part, str) or not chosen_part.strip():
        return None
    point = {"part": part or chosen_part.strip(), "point": grid, "unchanged": 0}
    return _verify(context, frame, view, point, reference, operation="verify")


def _verify(context, frame, view, point, reference, *, operation):
    prompt = Template(_text("verify")).substitute(
        task=context["task"], subgoal=json.dumps(context["subgoal"], sort_keys=True), view=_text(view),
        part=point["part"], reference=_text("reference") if reference is not None else "",
    )
    annotated = _draw_point(frame, _pixels(point["point"], frame.shape), context)
    payload = _request(context, prompt, annotated, VERIFY_SCHEMA, operation=operation,
                       view=view, reference=reference)
    if payload is None or payload.get("present") is not True:
        return None
    on_target = payload.get("on_target")
    if not isinstance(on_target, bool):
        return None
    grid = point["point"] if on_target else _grid_point(payload.get("point"))
    if grid is None:
        return None
    return {**point, "point": grid, "verified": on_target, "why": payload.get("why", "")}


def _request(context, prompt, image, schema, *, operation, view=None, reference=None):
    """Record the sub-role and restore the controller's trace even on model failure."""
    trace = context.get("trace")
    saved_trace = trace.context if trace is not None else None
    try:
        if trace is not None:
            trace.context = {**saved_trace, "role": "affordance", "operation": operation, "view": view}
        response = context["client"].complete_json(
            prompt, image, wrist_image=reference, schema=schema, temperature=0.0,
            debug=context.get("debug", False), context=context,
        )
        payload = response.payload.get("json")
        if not isinstance(payload, dict):
            raise ValueError("Affordance response must be a JSON object")
        return payload
    except (RuntimeError, ValueError, ConnectionError) as exc:
        if trace is not None:
            trace.event("Affordance model error", {**trace.context, "error": str(exc)})
        print(f"[affordance] {operation} failed: {' '.join(str(exc).split())[:160]}")
        return None
    finally:
        if trace is not None:
            trace.context = saved_trace


def _grid_point(value):
    """Reject malformed/out-of-image coordinates rather than pinning them to an edge."""
    if not isinstance(value, list) or len(value) != 2:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not 0 <= v <= 1000 for v in value):
        return None
    return [int(round(v)) for v in value]


def _pixels(point_yx, shape):
    y, x = point_yx
    height, width = shape[:2]
    return [int(round(x / 1000 * (width - 1))), int(round(y / 1000 * (height - 1)))]


def _draw_point(frame, xy, context):
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    x, y = xy
    r = context["harness"]["affordance_dot_radius_px"]
    draw.ellipse((x-r-1, y-r-1, x+r+1, y+r+1), fill="white")
    draw.ellipse((x-r, y-r, x+r, y+r), fill=(255, 0, 0))
    return np.asarray(image)


class LiberoAffordanceAPI:
    """Expose actual loaded objects, semantic sites, and named visual mesh parts.

    Locations are site/geometry/body reference centers, not computed grasp poses.
    Every lookup reads current MuJoCo data, including articulated and moved parts.
    """

    def __init__(self, env, cfg):
        self.env, self.cfg, self.sim = env, cfg, env.sim
        self._parts = {}
        model = self.sim.model
        objects = {**env.env.objects_dict, **env.env.fixtures_dict}
        for name, obj in objects.items():
            if name not in env.env.obj_body_id:
                continue
            self._parts[(name, "body")] = ("body", env.env.obj_body_id[name], "object body reference center")
            prefix = obj.naming_prefix
            for idx, site in enumerate(model.site_names):
                if site and site.startswith(prefix) and not site.endswith("_site"):
                    self._parts[(name, "site:" + site[len(prefix):])] = ("site", idx, "named region center")
            # LIBERO often gives handle meshes semantic names but only numeric geom
            # names. Read the mesh labels instead of inventing a handle coordinate.
            for idx, geom in enumerate(model.geom_names):
                if not geom or not geom.startswith(prefix) or model.geom_group[idx] != 1:
                    continue
                mesh_id = int(model.geom_dataid[idx])
                if int(model.geom_type[idx]) != 7 or mesh_id < 0:  # mjGEOM_MESH
                    continue
                mesh = model.mesh_id2name(mesh_id)
                if not mesh:
                    continue
                label = mesh[len(prefix):] if mesh.startswith(prefix) else mesh
                part = "mesh:" + label
                if (name, part) in self._parts:
                    part += ":" + geom[len(prefix):]
                self._parts[(name, part)] = ("geom", idx, "named visual mesh center")

    def catalog(self):
        return [{"object_name": obj, "part_name": part, "reference": spec[2]}
                for (obj, part), spec in sorted(self._parts.items())]

    def get_object_part(self, object_name: str, part_name: str):
        key = (object_name, part_name)
        if key not in self._parts:
            raise ValueError(f"Unknown simulator object part: {key!r}; choose an exact catalog entry")
        kind, idx, reference = self._parts[key]
        position = getattr(self.sim.data, kind + "_xpos")[idx].copy()
        return {"object_name": object_name, "part_name": part_name,
                "world_position": position.tolist(), "reference": reference,
                "simulator_id": idx, "simulator_kind": kind}

    def project(self, world_position, camera_name, image_shape):
        """Project without clipping off-screen points onto an image edge.

        Raw MuJoCo images have rows along camera +Y. Match robosuite's image
        convention and then the exact np.rot90 used by observation_images().
        """
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

        model, data = self.sim.model, self.sim.data
        camera = model.camera_name2id(camera_name)
        rotation = self.cfg["libero"]["agentview_rotation" if camera_name == "agentview" else "wrist_rotation"]
        height, width = image_shape[:2]
        if rotation in (90, 270):
            height, width = width, height
        local = data.cam_xmat[camera].reshape(3, 3).T @ (np.asarray(world_position) - data.cam_xpos[camera])
        depth = -local[2]
        if depth <= model.vis.map.znear * model.stat.extent:
            return None
        focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[camera]) / 2))
        x = (width - 1) / 2 + focal * local[0] / depth
        y = (height - 1) / 2 + focal * local[1] / depth
        if IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION] == -1:
            y = height - 1 - y
        for _ in range(rotation // 90):
            x, y = y, width - 1 - x
            width, height = height, width
        if not (0 <= x < width and 0 <= y < height):
            return None
        return [float(x), float(y)]


def _text(section="prompt"):
    text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
    if section != "prompt":
        _, marker, text = text.partition(f"\n[{section}]\n")
        if not marker:
            raise ValueError(f"Missing [{section}] in {__file__}'s prompt.txt")
    return text.partition("\n[")[0].strip()
