# SAFEPASS P3: Roadmap from Real CCTV Footage (e.g. a Church Venue) to a Working System

**Purpose:** a step-by-step method for acquiring church/venue CCTV footage and wiring it into
the already-built P1-P3 pipeline. Everything below maps onto existing code: no new schemas
are needed. The plan is ordered as a sequence of gates - do not start a later gate before
the earlier one is signed off.

```
GATE 0            GATE 1              GATE 2              GATE 3               GATE 4              GATE 5
Consent &       Site survey     Footage         Pixels ->         Learned topology     Operator
governance      & venue graph   acquisition     Candidates        from real logs       workflow (P4)
(no footage)    (no footage)    (1-2 week pilot) (P1/P2 run)      (P3 learner)         (drills)
```

---

## GATE 0 - Consent and governance (before touching any footage)

A church is a private venue with a duty-of-care population (children, elderly). Get this
wrong and the whole dataset becomes unusable.

1. **Written agreement with the church authorities** covering: purpose (locating reported
   missing persons only), what is recorded, who accesses it, retention period, deletion.
2. **Consent framework (DPDP Act 2023)**: the intake schema already models guardian
   consent (`IntakeRequest.consent_confirmed`, guardian name/contact). Enforce it:
   - No footage of anyone is *searched* without an active, consented case.
   - No minors' biometric data is enrolled or retained, ever (project rule).
3. **Signage** at every entrance: "CCTV in operation for safety purposes, contact <office>".
4. **Data minimisation**:
   - Retain raw footage only for the agreed window (suggest 30 days), then purge.
   - Store extracted face crops/embeddings, not whole videos, in the searchable layer.
   - Access control: encrypted disk, named operator accounts, audit log of every search.
5. **Prefer a pilot with a consenting congregation event** (e.g. a volunteer-run service)
   rather than a regular service, for the first acquisition round.

**Deliverable:** signed agreement + a one-page data-handling SOP.

---

## GATE 1 - Site survey and the real venue graph (no footage needed)

This replaces the synthetic `default_venue()` with the actual building. It is cheap,
fast, and everything downstream depends on it.

### 1.1 Camera inventory
For every camera, fill one row (template table below):

| Field | Example | Why P3 needs it |
|---|---|---|
| camera_id | CAM-01 | node id |
| camera_name | Main West Door | display |
| zone_name | West Porch | search zone |
| mounting point (floor plan) | porch, 3.2 m | `coordinates` |
| height / tilt | 3.2 m / 15 deg | capture quality check |
| observed area | door + 4 m inside | adjacency decision |
| resolution / fps / codec | 1080p / 25 / H.264 | Gate 2 |
| clock source | NVR, NTP? offset? | Gate 2, pruner |
| sees faces frontally? | yes (choke point) | P1 viability |

### 1.2 Floor plan and distances
1. Get the church floor plan (or sketch one). Mark camera positions on it - these become
   the graph `coordinates` in metres.
2. Measure **walking-path distances** between adjacent camera fields of view with a
   laser distance meter (or pace + scale from the plan). Measure along the actual
   walkable route (around pews, not through them), not straight-line - this feeds
   `distance_meters` and therefore every transit window.
3. Walk each route yourself and time it at a slow, normal and brisk pace. Record the
   observed speeds - for a church population (elderly, families) expect slower speeds
   than the default `v_min=0.9, v_max=4.5 m/s`. Consider re-parameterising, e.g.
   `v_min=0.5, v_max=3.0` - it is a constructor argument on `VenueTopologyGraph`.

### 1.3 Adjacency and blind spots
- Two cameras are **adjacent** iff a person can walk from one's field of view to the
  other's without necessarily passing a third camera. Verify by walking each route.
- Deliberately identify the **blind-spot gaps** (storage rooms, corridors, the area
  behind the altar, parking walkways) - these are exactly where the forecaster earns
  its keep.
- Sanity-check the graph: `get_shortest_path()` between random camera pairs should
  match the routes you actually walked.

### 1.4 Cold-start priors
Ask the verger/wardens: "if someone is at camera X, where are they most likely to go
next?" Encode as `set_priors()` weights. They only need to be roughly right - Gate 4
replaces them with learned values - but `alpha=0` makes the learner start *exactly*
from them, so they are the fallback model until enough data exists.

### 1.5 Export the graph
```python
from topology.graph import VenueTopologyGraph
g = VenueTopologyGraph(v_min=0.5, v_max=3.0)
g.add_camera("CAM-01", "Main West Door", "West Porch", (x, y)) ...
g.add_corridor("CAM-01", "CAM-02", distance_meters=14.0) ...
g.set_priors("CAM-01", {"CAM-02": 0.5, ...})
g.validate()
g.to_json("venue_church.json")   # feeds everything, including P4
```
The file format is already proven by `topology/stubs/venue_graph.json`, and
`VenueTopologyGraph.from_json()` loads it anywhere in the system.

**Deliverable:** `venue_church.json` + camera inventory sheet + floor plan with
coordinates and measured distances.

---

## GATE 2 - Footage acquisition (the "how to capture" question)

### 2.1 Two acquisition modes
| Mode | How | When |
|---|---|---|
| **A. NVR export** (simplest) | Export per-camera MP4/H.264 from the church recorder for chosen event windows | pilot, one-off drills |
| **B. Live tap** (best long-term) | RTSP pull per camera with ffmpeg to fragmented MP4 | continuous operation |

Mode A is enough to validate everything. Move to B only when the system is trusted.

```bash
# Mode B example: 1080p direct-from-camera copy, no re-encode, 1 file per camera per hour
ffmpeg -rtsp_transport tcp -i rtsp://<nvr>/cam01 -c copy -f mp4 -strftime 1 cam01_%Y%m%d_%H.mp4
```

### 2.2 Capture specifications (what to ask for / configure)
1. **Resolution:** at least 1080p. P1 needs a face height of roughly **>= 40-60 px** to
   match reliably - so at entrance/choke-point cameras a face at 3-4 m must be >= 50 px.
   Check this in Gate 1; if the church can reposition even one camera, put it where
   faces are captured **frontally at a choke point** (doors) - that single camera
   becomes the enrollment-quality anchor for the whole system.
2. **Frame rate:** 15-25 fps. Higher is wasted (P1 samples 1 frame / 0.5 s anyway);
   lower hurts tracking.
3. **Codec/container:** H.264 MP4, and **never re-encode** - transcode once, at the end
   of ingest, only if necessary. Double-compressed CCTV is the classic face-killer.
4. **Continuous recording, not motion-triggered**, during capture windows. Motion gaps
   create phantom "teleportations" and fake blind spots in the pruner. If only
   motion-triggered recording exists, note the config - the pruner can be run in
   manual-fallback mode (`use_learned=False`) until the effect is measured.
5. **Clocks:** NTP-sync the NVR (or record its drift). All P3 math is delta-t based;
   unsynchronised camera clocks look like teleportation or impossible speed. The pruner
   has `clock_skew_tolerance_seconds` for residual skew - measure the actual skew once
   (wave at two adjacent cameras, compare timestamps) and set it explicitly.
6. **Timestamps on file:** keep the NVR's on-screen timestamp or embed it in the
   manifest - never rely on file modification times.

### 2.3 How much and when
- **For a search case (operational):** the continuous window around the case, all cameras.
- **For learning the topology (Gate 4):** you need *movement* - people entering, walking,
  leaving. Capture **4-8 busy events** (Sunday services, a wedding, a funeral - ask
  consent) at all cameras simultaneously. Our learning curve says: ~10-25 distinct
  movers per camera route already halve the prior error, and a few hundred consolidate
  it - one busy Sunday is roughly a few hundred movers.
- **Storage math:** 8 cameras x 1080p/25fps H.264 ~ 4 Mbps each ~ 1.7 GB/hour/camera
  ~ 14 GB/hour total; a 3-hour service ~ 42 GB. Plan disk accordingly; purge after
  the retention window.

### 2.4 File layout and manifest (this is what "incorporating" looks like on disk)
```
data/
  raw/CAM-01/2026-11-01/service_0900_1200.mp4
  raw/CAM-02/...
  manifest/
    2026-11-01.json     # [{camera_id, camera_name, file, start_time (ISO, with
                        #  timezone), fps, resolution, clock_offset_seconds}]
```
`face/video_search.py` already takes exactly this config per stream
(`camera_id`, `camera_name`, `video_path`, `start_time`).

**Deliverable:** 4-8 event windows of synchronised multi-camera footage + manifest.

---

## GATE 3 - Pixels to Candidates (run the existing P1/P2 on real footage)

1. **Frame harvesting:** sample 2 fps per camera into JPEG (P1's `sample_interval_sec`).
2. **Run the P1 pipeline** (`VideoSearchEngine`) per camera file. Output: `Candidate`
   objects with `camera_id`, `camera_name`, `timestamp` (ISO), `face_score`,
   `fused_score`, quality fields - already the contract P3 consumes.
3. **Run P2** (OSNet appearance + colour) so `fused_score` is the quality-weighted
   multi-modal score, not face-only.
4. **Immediate per-camera QC you must do before anything else:**
   - Detection rate: how many faces/hour found per camera at 2 fps?
   - Face size distribution: fraction of detections with usable face height.
   - Duplicate suppression: consecutive same-person sightings should collapse - the
     learner's visit logic handles gaps <= 20 s; check it matches the camera reality.
5. **Do not lower thresholds to get more matches.** Low-precision matches are exactly the
   look-alikes our evaluation stress-tested - the topology handles the *rate* we tested
   (up to 50%), not unbounded noise.

**Deliverable:** per-event candidate logs (JSON) that look exactly like
`topology/stubs/candidates_pruned.json` (minus the P3 fields, which get filled next).

---

## GATE 4 - Bootstrap the learned topology on real data (the delicate part)

The learner (`TransitionLearner`) needs sightings labelled by person. On real footage
nobody hands you labels - this is the main new engineering step, and it is a
chicken-and-egg: you need re-ID to label people, and topology to trust re-ID.

### 4.1 The curriculum (three passes, increasingly automatic)
**Pass 1 - manual priors only (day 1).** Run the full pipeline with the Gate 1 graph,
`use_learned=False`. The physical transit bounds already reject teleportation; the
manual priors already rank next cameras. This is a working system from the first
hour of footage.

**Pass 2 - strict-threshold pairs (week 1).** Build pseudo-person logs:
1. Within a camera, tracklets are cheap (detection overlap). Give each tracklet an ID.
2. Across cameras, link two tracklets only if face/appearance similarity is **very
   high** (e.g. fused >= 0.85 - err on the side of too few links) AND the pair passes
   `SpatiotemporalPruner.score_pair()`. Each accepted link is a `(person_id,
   camera_id, timestamp)` sighting pair - exactly what `TransitionLearner.update()`
   accepts (it also takes contract `Candidate`s keyed by `case_id`, dicts, or
   `(person, camera, t)` tuples).
3. Feed the linked sightings to the learner with `alpha=0` (start exactly from priors)
   and `track_cleaning="chain"`. Only a few hundred linked transitions are needed
   (our curve: KL error ~0.01-0.05 with ~100-400 movers).

**Pass 3 - iterate (week 2+).** Re-run the linking with the *learned* model in the
pruner (it is a better filter than physical bounds alone - our ablation: look-alike
discrimination AUC 0.69 -> 0.93). Refit the learner. Each iteration makes both the
linking and the model better.

### 4.2 Site-specific parameters to re-estimate (do not inherit the defaults blindly)
| Parameter | Default | How to set it for the church |
|---|---|---|
| `v_min`, `v_max` | 0.9 / 4.5 m/s | from Gate 1.2 walking timings (elderly congregation -> lower both) |
| `dwell` expectation | simulator's 2-8 s | people loiter at the door talking - the learner picks this up from data; only the *blind-spot flag* threshold (`blind_spot_coverage=0.98`) may need tuning |
| `clock_skew_tolerance_seconds` | 0 | measured in Gate 2.2.5 |
| `same_visit_gap_seconds` | 20 s | set from observed detection spacing |
| event-conditioned behaviour | - | church movement is event-driven (procession in, procession out); if priors differ hugely between events, learn per-event models later - out of scope for now |

### 4.3 Validation drill (the ground truth you control)
Before trusting the learned model on real cases, run 3-5 consenting adult volunteers
through scripted routes:
1. Volunteer walks known routes (include at least one blind-spot gap and one doubling-back
   route), a helper logs entry/exit times per camera with a phone (synced to NTP).
2. Enroll the volunteer's face (P1), run the full pipeline on that window.
3. Score with the existing eval harness (E1-E6 have portable definitions):
   - trajectory precision/recall of true sightings (expect > 0.95),
   - blind-spot interval hit,
   - next-camera top-1/top-3 and arrival-window coverage.
4. If the numbers are far off the synthetic benchmarks, the usual culprits in order:
   clock skew, mis-measured distances, motion-triggered gaps, low-resolution faces.

**Deliverable:** learned model JSON (`TransitionLearner.to_json()`), validation report
per drill, updated venue graph loaded via `graph.load_learned_model()`.

---

## GATE 5 - Operator workflow (P4 integration, already stubbed)

The contract round-trip is already proven - `topology/stubs/search_case_result.json`
is exactly what P4's dashboard consumes:

```
intake (guardian consent, last seen camera/time)
   -> P1/P2 search across the event window -> Candidates
   -> SpatiotemporalPruner.prune(candidates, anchor_camera_id, anchor_timestamp)
   -> TrajectoryReconstructor.reconstruct(pruned)   -> reconstructed_trajectory
   -> BlindSpotForecaster.forecast(last camera, last time, history) -> predicted_next_zone
   -> operator follows the predicted cameras within the arrival window
```

Drill the full loop with a volunteer "missing person" at least twice before any real
case. Time-to-locate is the metric that matters to the church, not model metrics.

---

## What can go wrong - the honest list

1. **Cameras only see the back of heads.** Churches mount cameras high for coverage,
   not faces. Fix in Gate 1 (one choke-point camera at face height) - no software
   fixes this.
2. **Motion-triggered recorders** create fake blind spots. Ask for continuous, or at
   least know the config.
3. **Clock drift** silently destroys delta-t reasoning. Measure, set the tolerance.
4. **Event-driven crowds** (everyone leaves at once through one door) - fine for
   learning, but a real search case during a crush is the hardest condition; validate
   with a drill at a busy exit.
5. **Pseudo-tracklet linking noise** is the main new risk - that is why Pass 2 uses
   a deliberately strict threshold and why the learner's chain cleaning exists (it
   was validated to keep learning accurate with up to 50% false matches).
6. **Consent drift**: the moment the system is used for anything beyond consented
   missing-person cases (curiosity searches), stop - it is both an ethical and a
   legal line.

---

## Timeline suggestion (assuming weekends are the busy days)

| Week | Gate | Effort |
|---|---|---|
| 1 | Gate 0 + Gate 1 (agreement, survey, venue JSON) | 1-2 days on site |
| 2 | Gate 2 pilot: capture 1-2 events; manifest; QC | half day on site + processing |
| 2-3 | Gate 3: P1/P2 runs, per-camera QC | automated, mostly compute |
| 3-4 | Gate 4 Pass 1-2: manual priors live, strict linking, first learned model | 2-3 days eng |
| 5 | Gate 4 Pass 3 + validation drills (volunteers) | 1 day on site per drill |
| 6+ | Gate 5: P4 drills, then limited real use | with P4 track |

Nothing here blocks on the P1/P2 deadline (29 Oct) - Gates 0-1 need no footage and can
be done in parallel any time.
