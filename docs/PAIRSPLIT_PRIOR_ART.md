# PairSplit — prior-art audit (what is borrowed, what is new)

Written 2026-11-11 after a targeted literature search. Purpose: make sure PairSplit is
not a re-named copy of an existing mechanism, and that every borrowed idea is cited with
the exact thing that was borrowed.

**Honesty rule for this document:** every arXiv ID below was checked against the paper
title in a live search. Where a claim is "I did not find X", it means *not found with the
queries listed in §5*, not "X does not exist".

---

## 1. The mechanism, restated as three separable claims

PairSplit is only defensible if we separate what is a **primitive** (someone else's) from
what is a **combination/regime** (ours):

| # | Claim | Status after the audit |
|---|---|---|
| C1 | *One spatial location may emit more than one box* | **NOT new.** CrowdDet (2003.09163), PBM (2003.12729), and even YOLOv3's 3-anchors-per-cell do this. |
| C2 | *A small head reading a local feature patch around a box can predict another box relative to it* | **NOT new.** This is the RoI head of Faster R-CNN (1506.01497) in miniature; PBM's PPFE does the pair version. |
| C3 | *Emit the second instance **post-hoc**, from a frozen detector, on the **tiny-object** regime where the response width exceeds object spacing, with targets derived from existing boxes and no EMD/Hungarian matching* | **Not found in this combination.** This is the contribution. |

So the paper must claim C3, never C1 or C2. Claiming C1 would be immediately rejected by
any reviewer who knows CrowdDet.

---

## 2. Prior mechanisms that share something with PairSplit

Ordered by how close they are.

### 2.1 CrowdDet — the closest prior, and the one that must be cited first

- Chu, Zheng, Zhang, Sun. *Detection in Crowded Scenes: One Proposal, Multiple
  Predictions.* CVPR 2020 (oral). **arXiv:2003.09163**.
- **Shares:** the core idea that one proposal should output a *set* of instances instead
  of one, because in crowds several objects fall inside the same receptive field.
- **Borrowed by us:** C1, and the framing "the failure is that the head is forced to pick
  one object per location".
- **Differs:**
  1. **Regime.** CrowdDet targets *large* pedestrians with high IoU overlap (CrowdHuman,
     CityPersons, people ~50–400 px). PairSplit targets *tiny* persons (2–16 px) where the
     overlap is not geometric but **representational**: the model's response blob is wider
     than the gap between two people, so the two people never get separate peaks. Measured
     in this repo: nearest detection sits 0.37 GT-sizes from the missed person, median IoU
     with the missed GT 0.319, 53.5% of misses have a detection at IoU ≥ 0.3 (R22).
  2. **Where it lives.** CrowdDet replaces the detection head and requires full
     retraining. PairSplit is attached **after** a frozen checkpoint.
  3. **Supervision.** CrowdDet needs EMD loss + dummy boxes + set NMS. PairSplit derives
     its target from existing boxes (nearest neighbour within 1.5·max(w,h)) and uses plain
     BCE + masked L1. No matching problem, no set NMS (the base model is `end2end: True`,
     so it emits top-k directly).
  4. **Output cardinality.** CrowdDet emits K boxes per proposal always (K fixed). PairSplit
     emits *at most one extra* box per existing detection, gated by a learned probability,
     so the head is a no-op where there is no pair.

### 2.2 PBM / R²-NMS — the closest *structural* sibling (one location → two boxes)

- Huang, Ge, Jie, Yoshie. *NMS by Representative Region: Towards Crowded Pedestrian
  Detection by Proposal Pairing.* CVPR 2020. **arXiv:2003.12729**.
- **Shares:** a head that predicts **two boxes from one location** (full-body + visible-body
  pair), plus a paired-proposal feature extractor that integrates features of both.
- **Borrowed by us:** the architectural pattern "one location, two box outputs, features
  shared between them" — this is the honest ancestor of PairSplit's output shape.
- **Differs:** the two boxes in PBM describe **the same person** (full body vs visible
  part). The two boxes in PairSplit describe **two different people**. PBM's second box is
  a *part* of the first; PairSplit's second box is a *neighbour* of the first, displaced by
  a learned relative offset. PBM also needs a visible-region annotation; PairSplit needs
  none.

### 2.3 Faster R-CNN / Cascade R-CNN / RefineDet — the RoI-head pattern

- Ren et al., **arXiv:1506.01497**; Cai & Vasconcelos, **arXiv:1712.00726**; Zhang et al.,
  **arXiv:1711.06897**.
- **Shares:** read a feature patch around a candidate box, predict a refined box.
- **Borrowed by us:** C2 (`grid_sample` of a 3×3 neighbourhood = a 1-cell RoIAlign; the MLP
  is a 2-layer RoI head).
- **Differs:** these heads **refine the same object**. PairSplit uses the same machinery to
  predict a **different object**. A reviewer may say "this is just a second stage" — the
  answer is that a second stage that re-predicts the same object cannot raise recall by
  construction, while PairSplit's output is a new instance, which is why it is scored as a
  new box against a *different* GT (guard G2 in the pre-registration).

### 2.4 RefineBox — the closest "post-hoc on a frozen detector" prior

- Chen et al. *Enhancing Your Trained DETRs with Box Refinement.* **arXiv:2307.11828**.
- **Shares:** a lightweight, plug-and-play refinement module trained on top of a **frozen**
  detector, using its features and coarse boxes.
- **Borrowed by us:** the *deployment* pattern (frozen backbone, small add-on head, no
  retraining of the detector) — this is what makes PairSplit comparable across papers.
- **Differs:** RefineBox improves **localisation of already-detected** objects. It cannot
  increase recall. PairSplit's whole claim is a recall gain on a stratum where the object
  was never emitted. Different target, different metric (AP50_tiny vs box IoU).

### 2.5 IterDet — iterative detection with history

- Rukhovich et al. *IterDet: Iterative Scheme for Object Detection in Crowded
  Environments.* ICPR 2020. **arXiv:2005.05708**.
- **Shares:** the diagnosis that NMS/one-pass decoding loses people in crowds, and the
  remedy of running detection more than once with knowledge of what was already found.
- **Borrowed by us:** nothing structurally; it is the alternative we did **not** take.
- **Differs:** IterDet re-runs the whole detector N times (expensive, and it needs the
  detector to be trainable). PairSplit runs one extra 23 k-parameter MLP pass over the
  existing detections. IterDet also has no notion of "second instance relative to this
  box"; it re-detects globally.
- **Why we did not use it:** cost. IterDet multiplies inference cost by the number of
  iterations; our cost census already shows tiling at 4.68× the full-frame pass, so a
  multiplicative scheme is not affordable in this regime.

### 2.6 SKU-110K / EM-Merger — post-processing that splits box clusters

- Goldman et al. *Precise Detection in Densely Packed Scenes.* CVPR 2019.
  **arXiv:1904.00853**.
- **Shares:** the observation that NMS collapses densely packed objects, and that the fix
  must produce *more* boxes than the raw detector emits.
- **Borrowed by us:** nothing. It is the post-processing-only alternative.
- **Differs:** EM-Merger is **unsupervised geometry** on the detection set (fit a mixture of
  Gaussians, split clusters). It cannot invent a second box where the detector produced
  only one — it only re-partitions boxes that already exist. PairSplit adds information
  from the image features. This is the key distinction: EM-Merger is a re-arrangement,
  PairSplit is a prediction.

### 2.7 Subitizing / density-guided detection — "how many objects are here?"

- Zhang et al. *Salient Object Subitizing.* CVPR 2015 (count 0–4 per image/region).
- Li et al. *Density Map Guided Object Detection in Aerial Images (DMNet).* CVPRW 2020.
  **arXiv:2004.05520**.
- **Shares:** the idea that a local region can carry a *count* that the detector's
  one-box-per-location output cannot express.
- **Borrowed by us:** the motivation only ("the head knows there are two, but can only say
  one"). We deliberately did **not** implement a count head.
- **Differs:** a count is not a box. DMNet uses density to decide *where to crop*; a
  subitizing head outputs a number. Neither produces the second instance's coordinates,
  which is what AP needs. PairSplit predicts the count implicitly (a Bernoulli) **and** the
  geometry.

### 2.8 YOLOv10 dual assignment / one-to-many auxiliary heads (Co-DETR)

- Wang et al. *YOLOv10.* **arXiv:2405.14458**; Zong et al. *Co-DETR.* **arXiv:2211.12860**.
- **Shares:** "one-to-many" supervision exists in modern YOLO/DETR training.
- **Borrowed by us:** nothing.
- **Differs:** those one-to-many branches exist **only during training** to enrich
  gradients, and are **dropped at inference**; at inference they are strictly one-to-one.
  PairSplit operates **at inference** and its one-to-many output is the product, not a
  training aid. Also note the base model here is `end2end: True` (no NMS), so the
  "one-to-many gives redundant boxes" argument does not apply — the redundancy is already
  removed upstream.

### 2.9 YOLOv3-style multi-anchor cells

- Redmon & Farhadi, **arXiv:1804.02767**.
- **Shares:** several boxes per grid cell.
- **Differs:** the anchors are **fixed priors**, identical for every cell, and they compete
  to explain the *same* object (assignment picks one). PairSplit's second box is
  **conditioned on the first detection's content** and describes a *different* object.
  This is the difference between "3 guesses at one object" and "1 guess at a second
  object".

### 2.10 Relation Networks / duplicate removal

- Hu et al. *Relation Networks for Object Detection.* **arXiv:1711.11575**.
- **Shares:** a post-hoc module that looks at object-object relations after detection.
- **Differs:** it **removes** duplicates; PairSplit **adds** a missing instance. Opposite
  sign on the detection count. Worth citing to show we are aware of the post-hoc-relation
  family.

### 2.11 SAHI / tiling, Scale Match, NWD — the tiny-object baseline family

- Akyon et al. *SAHI.* **arXiv:2202.06934**; Yu et al. *Scale Match for Tiny Person
  Detection.* **arXiv:1912.10664**; Wang et al. *NWD.* **arXiv:2110.13389**; TinyPerson
  benchmark, **arXiv:2009.07506**.
- **Role in our story:** these are the **resolution/scale** fixes. Our own measurement
  shows tiling recovers +16.5 pp AP50_tiny (0.1886 → 0.3540) but leaves a **+17 pp**
  close-pair gap. PairSplit is the residual fix *after* the resolution fix, which is why it
  must be evaluated on top of tiling, not instead of it.

---

## 3. What is genuinely ours (defensible wording for the paper)

1. **The regime + the diagnosis.** The claim that in tiny-object detection the residual
   failure after 1:1 inference is *representational multiplicity*, evidenced by: nearest
   detection 0.37 GT-sizes away, median IoU 0.319 with the missed GT, 53.5% of misses
   having a detection at IoU ≥ 0.3, and only 3.7% of misses being NMS-credited to the
   neighbour. This diagnosis is measured in this repo and is not, to our knowledge, stated
   for TinyPerson.
2. **A post-hoc instance-splitting head.** Attaching multiplicity to a **frozen** detector
   as a 23 k-parameter add-on, so the gain is attributable and comparable across papers.
   The closest post-hoc prior (RefineBox) refines, it does not add instances.
3. **Annotation-free targets.** Deriving the second-instance target from the nearest
   neighbouring GT (radius 1.5·max(w,h), IoU ≥ 0.5 match) instead of EMD/Hungarian
   matching. This removes CrowdDet's most awkward machinery.
4. **Scale-invariant decode by construction.** Offsets in units of the source box, so one
   head serves 4 px and 40 px pairs; validated by a unit test.

## 4. What we must NOT claim

- ❌ "First to predict multiple instances per location" — CrowdDet, 2020.
- ❌ "First to use a paired box head" — PBM, 2020.
- ❌ "First post-hoc head on a frozen detector" — RefineBox, 2023.
- ❌ "Solves crowded detection" — we solve a *tiny-object* residual, and only measured on
  one dataset.
- ❌ "Replaces tiling" — tiling gives +16.5 pp; our ceiling is +3–4 pp on top.

## 5. Search log (what was actually queried)

Queries run (web search, 2026-11-11): "one proposal multiple predictions CrowdDet
multiplicity head EMD"; "tiny object detection second instance head split merged detection
crowded small objects 2024 2025"; "post-hoc second instance head frozen detector split
overlapping boxes"; "one-to-many label assignment tiny objects merged boxes NMS-free YOLO";
"subitizing counting objects per cell"; "counting-guided / density-map-guided detection";
"post-hoc refinement head frozen detector box refinement second stage"; "IterDet iterative
detection crowded"; "R2NMS PBM set NMS crowded pedestrian"; "EM-merger SKU110K splitting
overlapping boxes"; "tiny object detection merged predictions offset branch neighbouring
center 2023 2024"; "QueryDet cascaded sparse query"; "relative to the detected box predict
additional bounding box neighbour auxiliary head"; "TinyPerson benchmark SOTA 2020 2021".

Not found: any paper that (a) attaches a multiplicity head to a **frozen** detector, or
(b) formulates the tiny-object residual as "response width > object spacing" and fixes it
with a relative-offset second-instance head. Absence of evidence, stated as such.

## 6. Correction to earlier notes

Earlier drafts of `CRITERIA_PAIRSPLIT_PREREG.md`, `TINY_RESEARCH_LOG.md` (R23) and
`TINY_RESEARCH_FINAL.md` §14 cited CrowdDet as **arXiv:1906.04423**. That ID is wrong.
The correct ID is **arXiv:2003.09163** (verified against the title). All three files are
corrected. See `docs/LESSONS.md` L-016.
