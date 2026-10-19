---
name: sim-affordance
description: Select simulator object parts and write read-only API queries for ground-truth affordance markers in Show-Harness LIBERO runs. Use for grasp-part or placement-region guidance from simulator geometry.
---

# Ground-truth affordance queries

Choose the object instance and physical part that serve the CURRENT task stage.
The images establish which object is meant; the supplied simulator catalog defines
the exact available object and part names. Match object identity as well as shape:
an alphabet-soup can is not a milk carton. For grasping a handle, prefer a named
handle mesh. For placement, prefer the destination's named region. Use a body
reference only when the whole object is an appropriate target. Never substitute a
whole body for a requested handle that is absent from the catalog; abstain instead.

## Live API selection

Write one Python API expression in the `code` field of a JSON object:

```json
{"code": "sim.get_object_part(object_name='EXACT_OBJECT_NAME', part_name='EXACT_PART_NAME')", "reasoning": "This catalog entry is the part needed by the current stage."}
```

Replace both names with exact values from the catalog. If no catalogued part fits,
return `{"code": "None", "reasoning": "Explain the missing part or ambiguity."}`.
The runtime interprets only this read-only call with two literal string keyword
arguments. Do not generate imports, assignments, other calls, coordinates, or pixel
annotations. The API returns the current MuJoCo site/mesh/body reference center.
The renderer projects it into both cameras each step, including their configured
image rotations. No VLM re-annotation or verification calls are needed for tracking.

The controller receives: "The affordance is marked with a RED point ... This is the
object part that you should grasp." Non-grasp stages receive a target-region cue.
The marker is privileged simulator guidance and may overlay an occluded reference
center. It is not a collision-free approach, contact pose, or proof of grasp success.

## Writing a simulator-side adapter or an offline example

Use `plugins.affordance.build.LiberoAffordanceAPI(env, context)`. Enumerate `catalog()`
before selecting names; call `get_object_part(object_name=..., part_name=...)` after
each physics update. Use `project(world_position, camera_name, image.shape)` on the
same processed images sent to the controller. A `None` projection means off-screen
or behind the camera; do not clamp it to an image edge.

Keep simulator-derived coordinates out of the VLM selection response. If an object
needs a new contact anchor, derive it from its actual mesh/site in the adapter and
name its meaning explicitly, rather than inserting a guessed world/pixel offset.
Record the query, returned object/part, world position, and projected pixels beside
the clean and marked views. The live implementation is in
`plugins/affordance/build.py`, including the simulator API and model selection.
Selection wording and controller marker guidance both live in
`plugins/affordance/prompt.txt`.
