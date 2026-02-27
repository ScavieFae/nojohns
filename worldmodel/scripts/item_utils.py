"""Shared item/projectile extraction utilities for Slippi replay parsers.

Used by both build_dataset.py and parse_archive.py to extract real item data
from peppi_py's raw arrow struct (bypassing the Python wrapper which drops items).

ItemAssigner ported from slippi-ai/slippi_db/parsing_utils.py.
"""

import numpy as np
import pyarrow as pa

NUM_ITEM_SLOTS = 15


class ItemAssigner:
    """Map item spawn IDs to stable slot indices 0..NUM_ITEM_SLOTS-1.

    Items keep their slot across frames so the model can track individual
    projectiles over time. Ported from slippi-ai/slippi_db/parsing_utils.py.
    """

    def __init__(self):
        self.assignments: dict[int, int] = {}
        self.free_slots = list(range(NUM_ITEM_SLOTS - 1, -1, -1))  # stack, high→low

    def assign(self, item_ids) -> list[int]:
        # Free slots for items that disappeared
        ids_set = set(item_ids)
        for item_id in list(self.assignments):
            if item_id not in ids_set:
                self.free_slots.append(self.assignments.pop(item_id))

        slots = []
        for item_id in item_ids:
            if item_id in self.assignments:
                slots.append(self.assignments[item_id])
            elif self.free_slots:
                slot = self.free_slots.pop()
                self.assignments[item_id] = slot
                slots.append(slot)
            else:
                slots.append(-1)  # overflow — will be skipped
        return slots


def extract_items(slp_path: str, num_frames: int) -> pa.StructArray | None:
    """Extract items from raw peppi_py arrow struct.

    Returns a PyArrow StructArray with 15 item slots (item_0..item_14),
    each containing {exists, type, state, x, y} arrays of length num_frames.
    Returns None on failure (graceful fallback to empty items).
    """
    try:
        import peppi_py._peppi as raw_peppi
        raw_game = raw_peppi.read_slippi(slp_path)
        items_list = raw_game.frames.field('item')  # ListArray
    except Exception:
        return None

    if items_list is None or len(items_list) == 0:
        return None

    # Extract flat numpy arrays from arrow (fast — no per-frame .as_py())
    offsets = items_list.offsets.to_numpy()
    values = items_list.values
    if len(values) == 0:
        return None

    all_ids = values.field('id').to_numpy()
    all_types = values.field('type').to_numpy()
    all_states = values.field('state').to_numpy()
    all_x = values.field('position').field('x').to_numpy()
    all_y = values.field('position').field('y').to_numpy()

    # Allocate output arrays
    exists = np.zeros((num_frames, NUM_ITEM_SLOTS), dtype=bool)
    type_id = np.zeros((num_frames, NUM_ITEM_SLOTS), dtype=np.uint16)
    state = np.zeros((num_frames, NUM_ITEM_SLOTS), dtype=np.uint8)
    x = np.zeros((num_frames, NUM_ITEM_SLOTS), dtype=np.float32)
    y = np.zeros((num_frames, NUM_ITEM_SLOTS), dtype=np.float32)

    assigner = ItemAssigner()
    n_arrow_frames = min(len(items_list), num_frames)
    for frame_idx in range(n_arrow_frames):
        start, end = offsets[frame_idx], offsets[frame_idx + 1]
        if start == end:
            # No items this frame — still call assign with empty to free slots
            assigner.assign([])
            continue
        frame_ids = all_ids[start:end]
        slots = assigner.assign(frame_ids)
        for i, slot in enumerate(slots):
            if slot < 0:
                continue  # overflow — more than 15 items
            idx = start + i
            exists[frame_idx, slot] = True
            type_id[frame_idx, slot] = all_types[idx]
            state[frame_idx, slot] = all_states[idx]
            x[frame_idx, slot] = all_x[idx]
            y[frame_idx, slot] = all_y[idx]

    # Build PyArrow struct matching the schema parse.py expects
    item_slots = []
    for i in range(NUM_ITEM_SLOTS):
        slot_struct = pa.StructArray.from_arrays(
            [
                pa.array(exists[:, i]),
                pa.array(type_id[:, i]),
                pa.array(state[:, i]),
                pa.array(x[:, i]),
                pa.array(y[:, i]),
            ],
            names=["exists", "type", "state", "x", "y"],
        )
        item_slots.append(slot_struct)

    return pa.StructArray.from_arrays(
        item_slots, names=[f"item_{i}" for i in range(NUM_ITEM_SLOTS)]
    )


def extract_combat_context(slp_path: str, num_frames: int) -> dict | None:
    """Extract state_flags and hitstun from raw peppi_py arrow struct.

    Returns dict with per-port keys:
        p{0,1}_state_flags_{0..4}: (num_frames,) uint8 arrays
        p{0,1}_hitstun_remaining: (num_frames,) float32 array
    Returns None on failure.
    """
    try:
        import peppi_py._peppi as raw_peppi
        raw_game = raw_peppi.read_slippi(slp_path)
    except Exception:
        return None

    result = {}
    port_names = ['P1', 'P2']
    for port_idx in range(2):
        try:
            post = raw_game.frames.field('ports').field(port_names[port_idx]) \
                       .field('leader').field('post')
        except Exception:
            return None
        prefix = f"p{port_idx}_"

        # state_flags: 5 uint8 fields
        try:
            sf = post.field('state_flags')
            for byte_idx in range(5):
                arr = sf.field(str(byte_idx)).to_numpy().astype(np.uint8)
                result[f"{prefix}state_flags_{byte_idx}"] = arr[:num_frames]
        except Exception:
            # state_flags not available in this replay version
            for byte_idx in range(5):
                result[f"{prefix}state_flags_{byte_idx}"] = np.zeros(num_frames, dtype=np.uint8)

        # misc_as: float32 field, needs bit-recovery for subnormals
        try:
            raw_misc = post.field('misc_as').to_numpy().astype(np.float32)
            subnormal = (np.abs(raw_misc) < 1e-10) & (raw_misc != 0)
            recovered = np.where(
                subnormal,
                np.frombuffer(raw_misc.tobytes(), dtype=np.uint32).astype(np.float32),
                raw_misc,
            )
            result[f"{prefix}hitstun_remaining"] = np.clip(recovered[:num_frames], 0, 200)
        except Exception:
            result[f"{prefix}hitstun_remaining"] = np.zeros(num_frames, dtype=np.float32)

    return result


def empty_items_pa(num_frames: int) -> pa.StructArray:
    """Build 15 empty item slots as a PyArrow StructArray."""
    empty_item = pa.StructArray.from_arrays(
        [
            pa.array(np.zeros(num_frames, dtype=bool)),
            pa.array(np.zeros(num_frames, dtype=np.uint16)),
            pa.array(np.zeros(num_frames, dtype=np.uint8)),
            pa.array(np.zeros(num_frames, dtype=np.float32)),
            pa.array(np.zeros(num_frames, dtype=np.float32)),
        ],
        names=["exists", "type", "state", "x", "y"],
    )
    return pa.StructArray.from_arrays(
        [empty_item] * NUM_ITEM_SLOTS,
        names=[f"item_{i}" for i in range(NUM_ITEM_SLOTS)],
    )
