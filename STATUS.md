# ProxyGap Status

Last updated: 20 August 2026

## Final integrated result

The current delivery is a hierarchical **known-map** result. An archived V4
bidirectional expert, PAIR0 contact contract and waypoint planner completed
two route contracts across three new reset seeds each: **6/6 arrivals, 0
falls, 0 torso-ground contacts and 0 duration-corrected sustained-slip
events**. Three preference-labelled videos were reproduced exactly from the
formal traces and decoded frame by frame.

- Final result: [`docs/V4_PAIR0_MULTIOBJECTIVE_FULL_MAP_FINAL_REPORT_20260820_CN.md`](docs/V4_PAIR0_MULTIOBJECTIVE_FULL_MAP_FINAL_REPORT_20260820_CN.md)
- Delivery summary: [`docs/OVERNIGHT_OPTIMISATION_AND_DELIVERY_REPORT_20260820_CN.md`](docs/OVERNIGHT_OPTIMISATION_AND_DELIVERY_REPORT_20260820_CN.md)
- Presentation draft: [`deliverables/ProxyGap_Final_Presentation_Draft_20260820.pptx`](deliverables/ProxyGap_Final_Presentation_Draft_20260820.pptx)
- Report draft: [`deliverables/ProxyGap_Final_Report_Draft_20260820.docx`](deliverables/ProxyGap_Final_Report_Draft_20260820.docx)

The result is not unseen-map generalisation, global route optimality,
biological natural gait or electrical-energy optimisation. Representative
successful runs still contain 9.25–10.02% complete control intervals without
foot contact.

## Canonical version

**V2 remains the canonical reward-research direction.** The later Project V3
terrain integration is reported separately so its planner and contact changes
are not confused with one frozen reward experiment.

| Workstream | Public status | Next gate |
|---|---|---|
| Intended task and safety requirements | Frozen for the final known-map mission | Retain arrival/dwell and five-substep safety definitions |
| Specified gait/contact pattern | Support improved; biological gait unresolved | Validate duty factor, phase and airborne exposure across speeds |
| Default-reward construct audit | Stage-1 development evidence | Replicate with independent training seeds |
| Reward-shaping mechanisms | Bounded interventions and rejected failures retained | Avoid further unconstrained reward scanning |
| External constraints | Separated from preference ranking | Retain explicit controller-versus-policy attribution |
| Body-dynamics replication protocol | Public protocol and code available; local outputs are outside this release | Do not infer a public result from the protocol alone |
| Final mitigation candidate | Frozen for the reported integration | Do not overwrite checkpoint, routes or final manifests |
| Formal known-map comparison | 6/6 reset-seed episodes completed | Add unseen maps and independent training seeds |
| Real hardware and disturbances | Out of scope | Future work only |

## Latest reward iterations

Two user-directed development versions are now recorded:

- `reward-v1-foot-landing`: four-foot grounded Vy/Vz shaping;
- `reward-v2-pitch-balance`: the same package plus event-level signed-pitch
  time balance.

The pitch version improved its declared balance score but increased landing
frequency and action roughness, so it remains development evidence rather than
a frozen final mitigation candidate. See
[`docs/REWARD_ITERATION_HISTORY_20260817.md`](docs/REWARD_ITERATION_HISTORY_20260817.md).

## Latest planar-transition development

The selected pre-pitch foot-landing policy has also been used as the starting
point for a command-conditioned flat-ground transition from positive-x motion
to braking and then positive-y motion. This is development evidence for one
fixed 90-degree translation change, not arbitrary planar navigation or yaw
control. See
[`docs/PLANAR_TRANSLATION_TRANSITION_20260818.md`](docs/PLANAR_TRANSLATION_TRANSITION_20260818.md).

## V1 status

The earlier control-cost coefficient sweep is preserved as **legacy
retrospective exploratory evidence**. It may motivate V2 but cannot be merged
with V2 development runs as if both came from one preregistered design.

## Public-release boundary

The final release adds only the hash-audited checkpoint, formal traces, videos,
reports and editable presentation materials supporting the reported result.
Smoke, failed, superseded, recovery, cache and unrelated local files remain
outside the release.
