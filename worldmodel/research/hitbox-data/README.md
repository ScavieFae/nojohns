# Hitbox Data Lookup Table

Static lookup table mapping `(character_id, action_state, state_age)` → hitbox properties for every attack frame in Melee. Gives the world model explicit physics information it currently has to infer from patterns.

## What's here

| File | What |
|------|------|
| `build_hitbox_table.py` | Builder script — reads meleedb + ssbm-data, outputs hitbox_table.json |
| `hitbox_table.json` | The lookup table (~7 MB, 33,749 entries) |
| `melee_hitboxes.csv` | Raw hitbox data from [BroccoliRaab/meleedb](https://github.com/BroccoliRaab/meleedb) (3,904 rows) |
| `action_state.json` | Action state ID mapping from [hohav/ssbm-data](https://github.com/hohav/ssbm-data) |

## Coverage

**100%** of attack frames in training data have a table entry. Verified on 20 games from parsed-v3-2k (125,018 attack frames).

The entries break down by source:

| Source | Entries | What it provides |
|--------|---------|-----------------|
| meleedb matches | 26,030 | Real hitbox properties (damage, angle, BKB, KBG, size, element) |
| Throw data | 256 | Hardcoded throw properties for top 7 characters |
| Gap fills | 7,398 | `is_active: False` for special move phases without hitboxes (laser loop, shine hold, etc.) |
| Overflow fills | 602 | Extended state_age entries beyond meleedb's IASA |

## Table format

Key: `"char_id:action_id:state_age"` (all 0-indexed, state_age matches Melee's `state_age` directly)

```json
"1:63:6": {
  "damage": 18.0,
  "angle": 80,
  "bkb": 30,
  "kbg": 112,
  "size": 5.0,
  "element": "normal",
  "is_active": true,
  "total_frames": 41,
  "iasa": 41
}
```

That's Fox upsmash at state_age=6 — first active hitbox frame, 18% damage, 80° launch angle.

## How to use in encoding

The table is a static lookup. At pre-encode time:

```python
DEFAULT = {"damage": 0, "angle": 0, "bkb": 0, "kbg": 0, "size": 0, "is_active": False}

key = f"{char_id}:{action}:{state_age}"
entry = hitbox_table.get(key, DEFAULT)

# Normalize and append to player frame encoding
features = [
    float(entry["is_active"]),
    entry["damage"] / 30.0,    # max ~30% in Melee
    entry["angle"] / 361.0,    # 361 = Sakurai angle
    entry["bkb"] / 200.0,      # base knockback
    entry["kbg"] / 200.0,      # knockback growth
]
```

Gated behind `EncodingConfig(hitbox_data=True)`. Fallback to zeros for any missing key — no need to pre-populate every possible triple.

## Character coverage

Top 5 competitive characters (our current training filter) have the best meleedb coverage:

| Character | Real hitbox entries | Notes |
|-----------|-------------------|-------|
| Fox | Normals + shine + up-B | Laser is projectile (no body hitbox, gap-filled) |
| Falco | Normals + shine + up-B | Same laser situation |
| Marth | Normals + all 4 dancing blade hits (ground + air) + dolphin slash + counter | Most complete |
| Sheik | Normals | Needles/vanish gap-filled |
| Captain Falcon | Normals + falcon punch + raptor boost + falcon dive + falcon kick | Very complete |

## What's NOT done yet

1. **Not wired into encoding.py** — table exists, encoder doesn't read it yet
2. **Not tested as a training feature** — need A/B: baseline vs baseline + hitbox_data
3. **Throw data is approximate** — active frames estimated (frames 4-6), should verify against frame data sources

## Spot checks

Fox fair (action 66): 7% damage at state_age 5-7, second hit 5% at 15-17 ✓
Fox upsmash (action 63): 18% at 80° (clean) state_age 6-8, 13% (late) 9-16 ✓
Fox shine (action 360): electric element, 72° angle, state_age 0 ✓

## Regenerating

```bash
.venv/bin/python worldmodel/research/hitbox-data/build_hitbox_table.py
```

Requires parsed training data at `~/claude-projects/nojohns-training/data/parsed-v3-2k` for gap-fill. Without it, still produces the meleedb-sourced entries (26K) but won't hit 100% coverage.
