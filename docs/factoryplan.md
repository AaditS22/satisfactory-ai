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

### Known v0 limits
No clock speeds, splitters, mergers, lifts, port names or floors.
Belts pick the first free port by direction.