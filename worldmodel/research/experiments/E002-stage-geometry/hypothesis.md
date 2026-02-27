# E002: Explicit Stage Geometry

**Category:** encoding
**Target wall:** on_ground recall crisis
**Target metric:** on_ground recall improvement, position error on platform stages (Battlefield, Yoshi's)

## Hypothesis

The model learns stage geometry from a 4-dimensional stage embedding. It has to figure out from data alone that Battlefield's ground is at y=0, that it has platforms at y=27.2, that Yoshi's platforms are at y=23.45, etc. This is asking the model to memorize collision geometry from a 4-float vector.

Replace the learned stage embedding with ~20 floats of explicit geometry: main platform edges, platform positions (padded to 3), blast zone boundaries. The model no longer has to guess where the ground is — it's told directly.

## Intervention

Add to `encoding.py`:
- `STAGE_GEOMETRY` lookup table (already exists in `generate_demo.py`)
- Replace stage embedding with concatenated geometry floats
- Per frame: `[ground_left, ground_right, plat1_x, plat1_y, plat1_w, plat2_x, plat2_y, plat2_w, plat3_x, plat3_y, plat3_w, blast_left, blast_right, blast_top, blast_bottom]` = 15 floats (padded, zero for missing platforms)

## What to measure

- on_ground recall (primary — does knowing platform positions help the model predict landings?)
- Position error on platform stages vs FD (FD has no platforms — should be easier baseline)
- Pos error near platform heights (y within 5 units of a platform)

## Run config

Baseline config with stage geometry replacing stage embedding. 2K games, 2 epochs, A100.

## Risks

- Adds 15 floats to the input but removes 4 embedding dims. Net input size change is small.
- The model might already have learned stage geometry well enough from data. If so, this is a null result — still worth knowing.
- Platform stages are minority in the data if random sampling. May need stage-stratified eval.
