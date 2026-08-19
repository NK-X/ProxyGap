# Approved fixed quadrant terrain V2

Approval date: 2026-08-18

Status: frozen user-approved training and evaluation map.

The map must not be regenerated or edited in place. Any terrain change requires a new versioned directory and new approval. Training code may create separate task XML files that change only the Ant initial pose; the frozen height array, hfield payload, contour texture, friction and terrain geometry must remain byte-identical.

## Frozen task geometry

- Map dimensions: 80 m by 80 m.
- Map area: 6,400 m2.
- Grid: 1025 by 1025; cell spacing 0.078125 m.
- Start: (-34 m, -34 m).
- Goal: (+34 m, +34 m).
- Straight-line distance: 96.1665222414 m.
- Peak-to-valley range: 6.0 m.
- Fixed floor friction: [1.0, 0.5, 0.5].
- Floor contact dimension: condim=3.
- Artificial safe diagonal corridor: absent.

## Authoritative SHA-256 identities

| Object | SHA-256 |
|---|---|
| `map/scene/heights_m.npy` | `59e60ddd91d799f44f84aa74a2ecff122ac01b1c7c7ea13fe14b032bc176eb9c` |
| `map/scene/terrain.hfield` | `d1632c89688f459f5e7abe9d1cc9b5af7f22230b1b19290f737ae3fed849f3db` |
| `map/scene/ant_fixed_quad_terrain.xml` | `af2514abb2558789caaf0cd8ee3792909a3dd77946f373b5fd2c8222fbb5f182` |
| `build_fixed_quad_terrain_map.py` | `a7d2a05fd1492e51af6f42ccd34dfd76832a9f88f3388495b2f7e5c5f60c1f62` |
| `map/fixed_quad_terrain_review_sheet.png` | `a9acea002721d259c0db06fcb195fe47613f5ba36af6ce88fea2fbedd47fd55e` |

## Experimental boundary

Approval fixes the environment input. It does not imply that the V22 policy can traverse the map. Training and evaluation outputs must be stored outside this directory and must report completion, fall or inversion, four-foot airborne events, foot slip and termination category separately from PPO return.
