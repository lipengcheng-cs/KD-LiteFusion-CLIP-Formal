# Teacher–Student Data Contract Audit

- Audit status: **FAIL_CURRENT_TEACHER_PROTOCOL**
- Existing teacher eligibility: **EXPLORATORY_ONLY**
- Formal fixed-split teacher required: **True**
- Teacher train ∩ student val: **708**
- Teacher train ∩ student test: **639**
- Canonical label conflicts: **0**

## 3×3 overlap matrix

| teacher \ student | train | val | test |
|---|---:|---:|---:|
| train | 3657 | 708 | 639 |
| val | 776 | 135 | 158 |
| test | 773 | 152 | 153 |

## Decision

The existing reproduction teacher and its current Logits KD result are exploratory only. Teacher training includes samples assigned to the student's validation and/or test split. Do not use those results as formal paper evidence. Train a new supplied-source reproduction teacher strictly on the fixed student train split before formal KD.

## Contract details

- Teacher counts: `{'train': 5119, 'val': 1097, 'test': 1098}`
- Student counts: `{'train': 6090, 'val': 995, 'test': 950}`
- Teacher native class order: `['affected_individuals', 'infrastructure_and_utility_damage', 'not_humanitarian', 'other_relevant_information', 'rescue_volunteering_or_donation_effort']`
- Student fixed class order: `['affected_individuals', 'infrastructure_and_utility_damage', 'rescue_volunteering_or_donation_effort', 'other_relevant_information', 'not_humanitarian']`
- Off-diagonal split overlaps: `3206`
- Duplicate sample IDs: `{'teacher': {'train': 0, 'val': 0, 'test': 0}, 'student': {'train': 0, 'val': 0, 'test': 0}}`
- Legacy-label merge counts: `{'injured_or_dead_people': 2, 'missing_or_found_people': 2, 'vehicle_damage': 2}`

All overlapping samples are written to `data_overlap_samples.csv`; canonical
label conflicts are written to `label_conflicts.csv`.
