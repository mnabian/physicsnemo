import json
import os


def load_global_features(json_path: str) -> dict[str, dict[str, float]]:
    """
    Load global features JSON once.

    Returns:
        dict[str, dict[str, float]]:
            Mapping run_id -> global feature dict
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Global features file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(
            "Global features JSON must be a dict keyed by run_id"
        )

    # Optional: sanity check values
    for run_id, features in data.items():
        if not isinstance(features, dict):
            raise TypeError(
                f"Global features for run '{run_id}' must be a dict"
            )

    return data


def get_global_features_for_run(
    all_global_features: dict[str, dict[str, float]],
    run_id: str,
) -> dict[str, float]:
    """
    Fetch global features for a single run.

    Args:
        all_global_features: output of load_global_features
        run_id: key identifying the run (e.g. derived from filename)

    Returns:
        dict[str, float]: global scalar features for this run
    """
    try:
        return all_global_features[run_id]
    except KeyError:
        raise KeyError(
            f"run_id '{run_id}' not found in global features file"
        )
