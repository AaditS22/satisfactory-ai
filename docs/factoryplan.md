# FactoryPlan schema

## Version 0

### Units and axes
- Metres. The mod converts to Unreal centimetres in exactly one place.
- Axes are Unreal's: x forward, y right, z up (left-handed).
- `yaw` in degrees, multiples of 90 in v0. Yaw 0 means items flow along +y.

### Top level
| Field | Type | Meaning |
|---|---|---|
| schema | string | Always "satai.factoryplan" |
| version | int | 0 |
| name | string | Human-readable label |
| entities | list | See below. Order does not matter; the mod sorts by kind. |

### Entity (all kinds)
| Field | Type | Meaning |
|---|---|---|
| id | string | Unique within the plan; belts refer to it |
| kind | string | "foundation", "machine" or "belt" |
| class | string | Docs class name, e.g. "Build_SmelterMk1_C" |

### foundation, machine
| Field | Type | Meaning |
|---|---|---|
| pos | [x, y, z] | Actor origin, metres, plan-local |
| yaw | number | Degrees |
| recipe | string | Machines only, e.g. "Recipe_IngotIron_C" |

### belt
| Field | Type | Meaning |
|---|---|---|
| from | string | id of the source entity (its free output is used) |
| to | string | id of the target entity (its free input is used) |

### Build request (POST /build)
{"plan": <FactoryPlan>, "site": {"origin": [x, y, z], "yaw": 0}}
`site.origin` is in world metres. Plan-local positions are rotated by
`site.yaw`, then offset by `site.origin`.

### Placement rules (data/rules.json, measured in data/catalog.json)
Plan-local grid: 1 m cells, foundations every 8 m starting at the plan origin.

- Foundations: centre at x, y = 8k + 4. z = floor - thickness/2, so the top
  is exactly the floor (clearance boxes are +-0.5 / +-1 / +-2 m; the extra 6.5 cm
  in the mesh bounds is decorative trim).
- Machines: origin z = floor. Origin x, y = whole metre + snap offset,
  given at yaw 0 in building-local axes; rotate the offset with the building
  (at yaw 90/270 the x and y offsets swap). This puts every port on a cell
  centre across the flow, and the front/back clearance edges on whole metres.
  Even-width machines (constructor, foundry, manufacturer) then cover one
  extra cell of padding.
- Footprint: clearance_min/clearance_max from the catalog, rounded
  outward to whole cells. Soft boxes may overlap other buildings.
- Belt height: every machine port is 1 m above the floor.
  Splitters, mergers and lifts (on_belt) have their origin at their port,
  so their z = floor + 1 m, centred on a cell; their ports are the centres
  of the 4 neighbouring cells.
- Lifts: bottom port at the origin, top port at (0, 0, H),
  4 m <= H <= 48 m in 1 m steps (wiki). Facing of the top port is still to
  be measured once lifts can be spawned with a height (the catalog's lift is
  zero-height, so its ports overlap).
- Miners: position comes from the resource node (Phase 2), not the grid.
  Mesh bounds include the drill head (z from -1.8 m to +18.3 m); the
  clearance starts at z = 0. Output port at (0, 8, 1).
- Floor height (Phase 3): use the mesh top (bounds_max z), not the
  clearance top, which is lower. Tallest: manufacturer 14.5 m.