import argparse
import json
import os
from typing import Dict, List, Tuple

import h5py
import numpy as np


def _derive_pair_paths(input_path: str) -> Tuple[str, str]:
    """
    Given a path to either the .h5 or .json file, derive and validate both paths.
    Returns (h5_path, json_path).
    """
    if input_path.endswith('.h5'):
        h5_path = input_path
        json_path = input_path[:-3] + '.json'
    elif input_path.endswith('.json'):
        json_path = input_path
        h5_path = input_path[:-5] + '.h5'
    else:
        raise ValueError("Input must be a .h5 or .json file path")

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON index file not found: {json_path}")
    return h5_path, json_path


def _load_index(json_path: str) -> Dict:
    with open(json_path, 'r') as f:
        return json.load(f)


def _compute_returns(h5_path: str, episodes: List[Dict]) -> Dict[int, float]:
    """
    Compute per-episode undiscounted return as sum of rewards stored under each trajectory group.
    Returns a mapping from episode_id to return.
    """
    returns: Dict[int, float] = {}
    with h5py.File(h5_path, 'r') as f:
        for ep in episodes:
            ep_id = int(ep["episode_id"]) if "episode_id" in ep else None
            if ep_id is None:
                raise KeyError("Episode entry missing 'episode_id'")
            grp_name = f"traj_{ep_id}"
            if grp_name not in f:
                # Skip if group missing; keep behavior explicit
                continue
            grp = f[grp_name]
            if 'rewards' not in grp:
                raise KeyError(f"Rewards dataset missing for {grp_name} in {h5_path}")
            rew = np.asarray(grp['rewards'][:], dtype=np.float64)
            returns[ep_id] = float(np.nansum(rew))
    return returns


def _make_output_paths(h5_in: str, percent: float, out_dir: str = None) -> Tuple[str, str]:
    base_dir = os.path.dirname(h5_in) if out_dir is None else out_dir
    os.makedirs(base_dir, exist_ok=True)
    base_name = os.path.basename(h5_in)
    stem, ext = os.path.splitext(base_name)
    suffix = f".top{(1 - percent) * 100 :.0f}"
    h5_out = os.path.join(base_dir, f"{stem}{suffix}{ext}")
    json_out = h5_out[:-3] + '.json'
    return h5_out, json_out


def _filter_and_copy(h5_in: str, json_in: str, h5_out: str, json_out: str, keep_ids: List[int]):
    index = _load_index(json_in)

    # Copy env_info, filter episodes list to only kept ids (preserve all metadata fields)
    in_episodes = index.get('episodes', [])
    kept_set = set(int(x) for x in keep_ids)
    out_episodes = [ep for ep in in_episodes if int(ep.get('episode_id', -1)) in kept_set]

    out_index = {
        'env_info': index.get('env_info', {}),
        'episodes': out_episodes,
    }

    # Save JSON first
    with open(json_out, 'w') as jf:
        json.dump(out_index, jf, indent=2)

    # Copy the selected trajectory groups
    with h5py.File(h5_in, 'r') as fin, h5py.File(h5_out, 'w') as fout:
        # Copy file-level attributes if any
        for k, v in fin.attrs.items():
            try:
                fout.attrs[k] = v
            except Exception:
                pass

        for ep in out_episodes:
            ep_id = int(ep['episode_id'])
            gname = f"traj_{ep_id}"
            if gname not in fin:
                continue
            # h5py copy preserves datasets and their properties
            fin.copy(fin[gname], fout, name=gname)


def main():
    parser = argparse.ArgumentParser(description="Filter ManiSkill dataset to top-X% returns and save as new dataset.")
    parser.add_argument('-i', '--input', required=True, help='Path to input .h5 or .json file')
    parser.add_argument('-p', '--percent', type=float, default=0.9, help='Fraction of max return to keep (e.g., 0.9)')
    parser.add_argument('-o', '--output_dir', default=None, help='Optional output directory; defaults to input file directory')
    args = parser.parse_args()

    h5_in, json_in = _derive_pair_paths(os.path.abspath(args.input))
    index = _load_index(json_in)
    episodes = index.get('episodes', [])
    if len(episodes) == 0:
        raise RuntimeError(f"No episodes found in {json_in}")

    ep_returns = _compute_returns(h5_in, episodes)
    if len(ep_returns) == 0:
        raise RuntimeError("No returns computed; check that trajectory groups exist and contain rewards")

    all_returns = np.array(list(ep_returns.values()), dtype=np.float64)
    r_min = float(np.nanmin(all_returns))
    r_max = float(np.nanmax(all_returns))
    threshold = args.percent * r_max

    keep_ids = [ep_id for ep_id, ret in ep_returns.items() if ret >= threshold]
    kept_returns = np.array([ep_returns[eid] for eid in keep_ids], dtype=np.float64)

    print(f"Input: {h5_in}")
    print(f"Episodes in index (JSON): {len(episodes)}")
    print(f"Trajectories found in H5: {len(ep_returns)}")
    print(f"Original return range: min={r_min:.3f}, max={r_max:.3f}")
    print(f"Threshold ({args.percent*100:.1f}% of max): {threshold:.3f}")
    print(f"Keeping {len(keep_ids)} trajectories (>= threshold)")
    if kept_returns.size > 0:
        print(f"Filtered return range: min={float(np.nanmin(kept_returns)):.3f}, max={float(np.nanmax(kept_returns)):.3f}")
    else:
        print("Filtered return range: N/A (no episodes kept)")

    h5_out, json_out = _make_output_paths(h5_in, args.percent, args.output_dir)
    _filter_and_copy(h5_in, json_in, h5_out, json_out, keep_ids)

    print(f"Saved filtered dataset:\n  H5:   {h5_out}\n  JSON: {json_out}")


if __name__ == '__main__':
    main()


