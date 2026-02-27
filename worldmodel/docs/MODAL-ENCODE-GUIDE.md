# Modal Encode Guide — v3 Data Pipeline

How to get large datasets encoded and onto Modal for training. Adapted from the 22K pipeline that worked (Feb 25-26, 2026).

## The Pattern

```
Local: parsed data (games/ + meta.json)
  → tar + upload to Modal volume
  → pre_encode_parallel: N workers encode chunks in parallel
  → _concat_chunks: single high-RAM container assembles final .pt
  → Train from the .pt on GPU
```

## What Worked for 22K

- `parsed-v2.tar` uploaded to Modal volume (3.4 GB compressed)
- 11 parallel workers (2K games each), 32 GB RAM per worker
- 1 concat worker at 128 GB RAM (22K = ~88 GB tensor data)
- Total time: ~20-40 min encode + concat
- Output: `encoded-22k.pt` (74.4 GB) on volume

## v3 Differences from v2

| | v2 | v3 |
|---|---|---|
| Float dims per frame | 58 (29/player) | 144 (72/player) |
| Int dims per frame | 15 (7/player + 1) | 17 (8/player + 1) |
| Bytes per frame | ~352 | ~712 |
| Tar source | `parsed-v2.tar` → `/tmp/parsed-v2` | Need new tar path |
| meta.json filter | `is_training=True` | `game_end_method==2` (GAME-only) |
| encoding_config | YAML subset only | Full resolved config (fixed in pre_encode.py) |
| NaN risk | None observed | hitstun_remaining can be NaN — must sanitize |

## Size Estimates

| Dataset | Games | Est. frames | Float tensor | Int tensor | Total .pt |
|---------|-------|-------------|-------------|-----------|----------|
| 2K GAME-only (done) | 2,000 | 19.4M | 10.5 GB | 2.5 GB | 13.8 GB |
| 12K GAME-only | 11,896 | ~112M | ~61 GB | ~14 GB | ~80 GB |
| 100K GAME-only | ~65K est. | ~611M | ~330 GB | ~78 GB | ~430 GB |

## Plan: 12K GAME-only

Local data at `~/claude-projects/nojohns-training/data/game-only-v3-12k/` (meta.json + symlinked games/).

### Step 1: Tar the parsed data

```bash
# The games/ dir is a symlink to parsed-v3-24k/games/ which has ALL 22K games.
# Only tar the ones we need (from GAME-only meta.json).
cd ~/claude-projects/nojohns-training/data
.venv/bin/python -c "
import json, subprocess
with open('game-only-v3-12k/meta.json') as f:
    meta = json.load(f)
md5s = [g['slp_md5'] for g in meta]
# Write file list
with open('/tmp/game-only-12k-files.txt', 'w') as f:
    for md5 in md5s:
        f.write(f'games/{md5}\n')
print(f'{len(md5s)} files to tar')
"

# Create tar with only GAME-only files
cd ~/claude-projects/nojohns-training/data/parsed-v3-24k
tar cf ~/claude-projects/nojohns-training/data/parsed-v3-12k-gameonly.tar \
  -T /tmp/game-only-12k-files.txt meta.json
```

### Step 2: Upload tar to Modal

```bash
.venv/bin/modal volume put melee-training-data \
  ~/claude-projects/nojohns-training/data/parsed-v3-12k-gameonly.tar \
  /parsed-v3-12k-gameonly.tar
```

### Step 3: Update _encode_chunk for v3

Two changes needed in `modal_train.py::_encode_chunk`:

1. **Tar path**: Currently hardcoded to `parsed-v2.tar` → `/tmp/parsed-v2`. Need to accept tar path as parameter or use the new tar.
2. **NaN sanitization**: After encoding, run `torch.nan_to_num(ds.floats, nan=0.0)` before saving chunk.

### Step 4: Run parallel encode

```bash
.venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode_parallel \
  --config worldmodel/experiments/mamba2-v3-2k-test.yaml \
  --dataset ~/claude-projects/nojohns-training/data/game-only-v3-12k \
  --max-games 12000 \
  --output /encoded-game-v3-12k.pt \
  --chunk-size 2000
```

### Step 5: Train

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/mamba2-v3-2k-test.yaml \
  --encoded-file /encoded-game-v3-12k.pt \
  --epochs 3 \
  --run-name mamba2-v3-12k
```

### Resource estimates

- 6 parallel workers (2K games each), 32 GB RAM per worker
- 1 concat worker: **128 GB** should fit (~80 GB tensor + overhead)
- Encode time: ~20-30 min
- Concat time: ~10-15 min
- Upload time: tar should be ~2-3 GB, ~5 min upload

## Plan: 100K GAME-only (from ranked)

ScavieFae parsing 116K ranked replays. After GAME-only filtering, expect ~65K games.

### Problem: 100K doesn't fit in 128 GB for concat

100K games ≈ 430 GB tensor data. Can't pre-allocate in RAM.

### Options

**A. Keep chunks separate (recommended for time pressure)**
- Encode in chunks as usual (33 workers × 2K games)
- Don't concat — modify `train()` to accept multiple chunk files
- Pro: No memory limit, works immediately
- Con: Requires train() changes, slightly slower data loading

**B. Streaming concat with memmap**
- Use numpy memmap (as in `pre_encode_chunked.py`) for the final tensor
- Write chunks to memmap file on disk, then `torch.save` from memmap
- Pro: Works within memory limits
- Con: Needs large NVMe (~500 GB), slower

**C. Two-tier chunking**
- Split 100K into 5 × 20K "super-chunks"
- Each super-chunk: parallel encode + concat at 128 GB (fits)
- Train from 5 .pt files (modify train() to load multiple)
- Pro: Reuses existing concat code
- Con: Still needs train() changes

**D. Bump concat RAM to 512 GB**
- Modal supports up to 1 TB RAM instances
- Cost: higher per-hour, but concat only runs ~15 min
- Pro: Zero code changes
- Con: Might not be available / expensive

### Recommended path: Option D first, fall back to C

512 GB Modal instance for 15 min of concat is cheap. If unavailable, option C with 5 × 20K super-chunks reuses all existing code.

### Data transfer from ScavieFae

```bash
# On ScavieFae, after parsing completes:
# 1. Filter to GAME-only
cd ~/data/parsed-v3-ranked
python3 -c "
import json
with open('meta.json') as f:
    meta = json.load(f)
game_only = [g for g in meta if g.get('game_end_method') == 2]
print(f'GAME-only: {len(game_only)} of {len(meta)}')
with open('meta-gameonly.json', 'w') as f:
    json.dump(game_only, f)
"

# 2. Tar GAME-only files
python3 -c "
import json
with open('meta-gameonly.json') as f:
    meta = json.load(f)
with open('/tmp/ranked-gameonly-files.txt', 'w') as f:
    for g in meta:
        f.write(f'games/{g[\"slp_md5\"]}\n')
print(f'{len(meta)} files')
"
tar cf ~/data/ranked-v3-gameonly.tar -T /tmp/ranked-gameonly-files.txt meta-gameonly.json

# 3. Upload to Modal (from ScavieFae or transfer to Scav first)
# If ScavieFae has modal CLI:
modal volume put melee-training-data ~/data/ranked-v3-gameonly.tar /parsed-v3-ranked-gameonly.tar
# Otherwise: scp to Scav, upload from there
```

## Code Changes Needed

### Must fix before 12K run

1. **`_encode_chunk`**: Parameterize tar path (currently hardcoded `parsed-v2.tar`)
2. **`_encode_chunk`**: Add `torch.nan_to_num()` after encoding
3. **`_concat_chunks`**: Save full resolved encoding_config (use `dataclasses.asdict()`)
4. **`pre_encode_parallel`**: Remove `is_training` filter (v3 meta.json doesn't use this)

### Nice to have for 100K

5. **`_concat_chunks`**: Option to bump RAM to 512 GB via config
6. **`train()`**: Accept comma-separated encoded files for multi-chunk training
7. **NaN audit step**: Log NaN/Inf counts per chunk before saving

## Lessons Learned

- **NaN in hitstun_remaining** caused training to diverge (loss=NaN at batch 1000). Root cause: subnormal float recovery in parser produces NaN for some replays. Fix: sanitize with `nan_to_num` at encode time.
- **Hardcoded shape check** (`float_tgt != (30,)`) broke on wider v3 tensors. Fixed to compute from EncodingConfig.
- **Encoding config saving** only saved YAML subset, not resolved defaults. Fixed to save full `dataclasses.asdict(enc_cfg)`.
- **Local encoding of 12K+ games OOMs** on MacBook (36 GB RAM, ~80 GB tensor). Must use Modal.
- **Batch rate ~2x slower** with v3 encoding (wider tensors): ~0.57s/batch vs ~0.06s/batch on A100 at bs=4096. (Update: need to verify — the 2K run's batch timing was 9.5 min / 1000 batches = 0.57s, which is 10x slower than expected. May be a DataLoader bottleneck from wider tensors.)
