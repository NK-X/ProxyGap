# Render Asset Audit

`ant_render_large_floor.xml` is a copy of the Gymnasium Ant-v5 `ant.xml` used
by the local ProxyGap environment. The only content change is the plane size:

```text
size="40 40 40" -> size="400 400 40"
```

This prevents the checkerboard texture ending when a policy travels beyond
approximately 40 m. It is used only for qualitative video rendering, never for
training or numerical evaluation.

| Asset | SHA-256 |
|---|---|
| Local Gymnasium source `ant.xml` | `CD5F83EF0EA35B0969E65D360C5BACD5B74CCAEF6B27E4433B5168C605E3E2BE` |
| `ant_render_large_floor.xml` | `E7869C73699DD7DFA58424C61186FEFE3133605C7A1F6A5E0BF286026287F7F0` |

An automated dynamics-equivalence test resets the default and render XMLs with
the same seed, supplies the same 25 pseudorandom actions, and requires all
observations and rewards to match within (10^{-12}). The test passed on
16 August 2026.
