# Deferred improvements

## 2026-09-28: larger extraction test

Deferred at the user's request while work focuses on the appointment-results UI.

Reference run: document `fc3c3bf640314ec6b8d436ad8e02b25c`, doctor profile Poirier version 3, 14 original pages and 98 appointment rows. Both model stages completed for 95 rows without recorded request failures or retries. Of those, 77 had no interpretation flags and 18 had unresolved annotations or conflicts. These are workflow counts, not an accuracy evaluation. Three rows were not processed because their patient assignment was missing.

### Doctor conventions to clarify

- Rows #31 and #45: define the intended billing output for VPV with CV. Do not assume the VS or VPEC CV mapping applies to VPV.
- Row #66: confirm whether this doctor's handwritten `09927` is an alias for the profile's `9927`, or must be preserved differently. Do not globally remove leading zeros from billing codes.
- Rows #54, #55, #57, #58, #59 and #60: clarify how to handle lesion-removal annotations paired with a visit code. Add only conventions explicitly supplied by the user; do not invent procedure mappings.

### Evidence and transcription

- Review redaction placement where it obscures useful handwriting. The saved crop for #64 covers the beginning of the procedure; #89 obscures the base visit annotation. Check the original locally before changing a redaction, preserving patient-identifier coverage.
- Review base-code ambiguities such as US/VS (#47), NPC/VPC (#59), and VPU/VPV (#74). Preserve raw model results and record any manual correction separately.
- Clarify when crossed-out text or unrelated clinical notes should create an issue affecting billing. Avoid treating harmless reading-order differences as billing conflicts.

### Patient assignment and CSV readiness

- Rows #68, #77 and #88 need patient assignment before processing; do not infer the patient from a neighbouring appointment.
- At inspection, the CSV builder reported blockers for #31, #45, #59, #64, #66, #68, #74, #77, #88 and #89.
- Keep the distinction between an informational warning and a missing procedure code or identity visible in the UI. Preserve the user's workflow: final validation happens later; no mandatory transcription-acceptance step.

### Confirmed working behavior

- Earlier CV cases in rows #9–#11 now produce `15824`, with `TT` for the telephone visits, without interpretation flags.
- CV suffix numbers, lists, or uncertainty about the suffix are not needed for billing when the base visit and presence of CV are clear. Preserve the original transcription; do not add a separate vulnerability-code output field.
