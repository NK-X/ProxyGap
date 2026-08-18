# Preview visual QA record

- Reviewer: Codex self-review (not independent or blinded review)
- Review date: 18 August 2026
- Viewed at: native image size in the Codex desktop image viewer
- Files reviewed: `single_slope.png`, `single_hill.png`, `random_mixed.png`
- Dimensions: 3647–3666 × 1594 pixels; PNG RGBA; approximately 300 dpi
- Data/geometry check: the three images were regenerated from their saved terrain arrays; no manual values were retyped
- Semantic check: x, y and height units are metres; slope is labelled rise/run; both colour bars and the start/goal core outlines are decoded
- Perceptual check: no clipping, overlap that prevents decoding, blank panels, visible spikes, discontinuities or obvious grid faceting were observed
- Accessibility check: height uses viridis and slope uses cividis; start/goal cores use colour plus distinct labels
- Defects found: none requiring revision after the final fixed-physical-guard regeneration
- Unresolved limitation: a rendered surface cannot establish contact-dynamics validity; the resolution report records increased heightfield contact multiplicity
