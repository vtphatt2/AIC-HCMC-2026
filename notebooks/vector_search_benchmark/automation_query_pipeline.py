"""
Select unique scene candidates for text-query benchmark generation.

The output is meant for Search-by-Text vector benchmark data, not temporal
search. Each candidate has one representative keyframe plus cluster-level
ground truth, so a generated query can match any keyframe in the same unique
scene interval.
"""
import json
import logging
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
REMOTE_ROOT = REPO_ROOT / "remote-server"
sys.path.insert(0, str(REMOTE_ROOT))

from scripts.sample_paths import default_sample_root, sample_subdir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("automation_query_pipeline")


class Config:
    sample_root = default_sample_root(REPO_ROOT)
    features_dir = None
    excluded_frame_ids = {
        "L01_V005_041311",  # black frame
        "L03_V001_003376",  # cropped sleeve close-up, too low-information
        "L01_V003_017907",  # blurry vertical phone frame
        "L01_V003_017989",  # blurry vertical phone frame
    }
    similarity_threshold = 0.8
    max_frame_gap = 250
    max_other_video_sim = 0.78
    max_outside_scene_sim = 0.82
    scene_gap = 500
    min_uniqueness_margin = 0.02
    min_cluster_size = 1
    ground_truth_neighbor_count = 2
    ground_truth_max_gap = 75
    ground_truth_min_similarity = 0.72
    max_candidates = 0
    output_path = REMOTE_ROOT / "candidate_keyframes.json"


def load_all_features(features_dir: Path) -> list[dict]:
    """Loads all feature embeddings from .npy files along with metadata."""
    records = []
    if not features_dir.is_dir():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    # Find all subdirectories corresponding to video IDs
    video_dirs = sorted([d for d in features_dir.iterdir() if d.is_dir() and d.name != "selected_keyframes"])

    logger.info("Loading features from %d video directories...", len(video_dirs))
    for video_dir in video_dirs:
        video_id = video_dir.name
        npy_paths = sorted(video_dir.glob("*.npy"))
        for path in npy_paths:
            frame_stem = path.stem
            try:
                frame_num = int(frame_stem)
            except ValueError:
                continue

            records.append({
                "video_id": video_id,
                "frame_id": f"{video_id}_{frame_stem}",
                "frame_number": frame_num,
                "feature_path": path,
            })

    logger.info("Loaded metadata for %d keyframes.", len(records))
    return records


def remove_excluded_frames(records: list[dict], excluded_frame_ids: set[str]) -> list[dict]:
    if not excluded_frame_ids:
        return records
    filtered = [record for record in records if record["frame_id"] not in excluded_frame_ids]
    logger.info("Excluded %d manually rejected keyframes.", len(records) - len(filtered))
    return filtered


def load_normalized_embeddings(records: list[dict]) -> np.ndarray:
    logger.info("Loading numpy embeddings into memory...")
    embeddings = []
    for record in tqdm(records, desc="Reading feature files"):
        vector = np.load(record["feature_path"]).astype("float32").reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError(f"Zero-norm feature vector: {record['feature_path']}")
        embeddings.append(vector / norm)
    matrix = np.stack(embeddings).astype("float32")
    logger.info("Embeddings matrix shape: %s", matrix.shape)
    return matrix


def connected_components(sims: np.ndarray, threshold: float) -> list[list[int]]:
    logger.info("Generating adjacency matrix with threshold >= %s...", threshold)
    adj = sims >= threshold

    logger.info("Clustering keyframes via connected components...")
    visited = np.zeros(adj.shape[0], dtype=bool)
    clusters = []
    for index in range(adj.shape[0]):
        if visited[index]:
            continue
        queue = [index]
        component = []
        visited[index] = True
        while queue:
            current = queue.pop()
            component.append(current)
            neighbors = np.where(adj[current])[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(int(neighbor))
        clusters.append(component)
    logger.info("Found %d raw similarity clusters.", len(clusters))
    return clusters


def nearest_outside_scene(
    records: list[dict],
    sims_to_centroid: np.ndarray,
    component_set: set[int],
    video_id: str,
    min_frame: int,
    max_frame: int,
    scene_gap: int,
) -> dict:
    outside_video = []
    outside_same_video_scene = []
    for index, record in enumerate(records):
        if index in component_set:
            continue
        similarity = float(sims_to_centroid[index])
        if record["video_id"] != video_id:
            outside_video.append((similarity, index))
            continue
        if record["frame_number"] < min_frame - scene_gap or record["frame_number"] > max_frame + scene_gap:
            outside_same_video_scene.append((similarity, index))

    def best(items: list[tuple[float, int]]) -> dict | None:
        if not items:
            return None
        similarity, index = max(items, key=lambda item: item[0])
        record = records[index]
        return {
            "frame_id": record["frame_id"],
            "video_id": record["video_id"],
            "frame_number": record["frame_number"],
            "similarity": round(similarity, 6),
        }

    return {
        "other_video": best(outside_video),
        "same_video_outside_scene": best(outside_same_video_scene),
    }


def video_index_map(records: list[dict]) -> dict[str, list[int]]:
    by_video: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_video.setdefault(record["video_id"], []).append(index)
    for indexes in by_video.values():
        indexes.sort(key=lambda index: records[index]["frame_number"])
    return by_video


def expand_ground_truth(
    candidate: dict,
    records: list[dict],
    x_norm: np.ndarray,
    by_video: dict[str, list[int]],
    protected_frame_ids: set[str],
    neighbor_count: int,
    max_gap: int,
    min_similarity: float,
) -> list[str]:
    core_frame_ids = set(candidate["ground_truth_frame_ids"])
    component = candidate["_component_indices"]
    centroid = np.mean(x_norm[component], axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    min_frame, max_frame = candidate["frame_range"]

    before = []
    after = []
    between = []
    for index in by_video[candidate["video_id"]]:
        record = records[index]
        frame_id = record["frame_id"]
        if frame_id in core_frame_ids:
            continue
        if frame_id in protected_frame_ids:
            continue

        frame_number = record["frame_number"]
        if min_frame <= frame_number <= max_frame:
            between.append(index)
        elif 0 < min_frame - frame_number <= max_gap:
            before.append(index)
        elif 0 < frame_number - max_frame <= max_gap:
            after.append(index)

    before = sorted(before, key=lambda index: min_frame - records[index]["frame_number"])[:neighbor_count]
    after = sorted(after, key=lambda index: records[index]["frame_number"] - max_frame)[:neighbor_count]

    expanded = set(core_frame_ids)
    for index in between + before + after:
        similarity = float(np.dot(x_norm[index], centroid))
        if similarity >= min_similarity:
            expanded.add(records[index]["frame_id"])

    return sorted(expanded, key=lambda frame_id: int(frame_id.rsplit("_", 1)[1]))


def cluster_and_filter(
    records: list[dict],
    threshold: float,
    max_frame_gap: int,
    max_other_video_sim: float,
    max_outside_scene_sim: float,
    scene_gap: int,
    min_uniqueness_margin: float,
    min_cluster_size: int,
    ground_truth_neighbor_count: int,
    ground_truth_max_gap: int,
    ground_truth_min_similarity: float,
    max_candidates: int,
) -> list[dict]:
    """Clusters keyframe embeddings and filters out non-unique clusters."""
    N = len(records)
    if N == 0:
        return []

    x_norm = load_normalized_embeddings(records)
    logger.info("Computing cosine similarity matrix...")
    sims = np.dot(x_norm, x_norm.T)
    clusters = connected_components(sims, threshold)
    by_video = video_index_map(records)

    unique_candidates = []
    discarded_small = 0
    discarded_multi_video = 0
    discarded_multi_scene = 0
    discarded_other_video = 0
    discarded_same_video_repeat = 0
    discarded_low_margin = 0

    for component in clusters:
        cluster_records = [records[i] for i in component]
        video_ids = {r["video_id"] for r in cluster_records}
        if len(component) < min_cluster_size:
            discarded_small += 1
            continue

        if len(video_ids) > 1:
            discarded_multi_video += 1
            continue

        video_id = next(iter(video_ids))
        frame_numbers = [r["frame_number"] for r in cluster_records]
        min_frame = min(frame_numbers)
        max_frame = max(frame_numbers)
        if (max_frame - min_frame) > max_frame_gap:
            discarded_multi_scene += 1
            continue

        cluster_vectors = x_norm[component]
        centroid = np.mean(cluster_vectors, axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)

        similarities = np.dot(cluster_vectors, centroid)
        best_index_in_cluster = np.argmax(similarities)
        best_overall_index = component[best_index_in_cluster]
        sims_to_centroid = np.dot(x_norm, centroid)
        nearest = nearest_outside_scene(
            records,
            sims_to_centroid,
            set(component),
            video_id,
            min_frame,
            max_frame,
            scene_gap,
        )

        nearest_other_video_sim = (
            nearest["other_video"]["similarity"] if nearest["other_video"] else -1.0
        )
        nearest_same_video_scene_sim = (
            nearest["same_video_outside_scene"]["similarity"]
            if nearest["same_video_outside_scene"]
            else -1.0
        )
        outside_scene_sim = max(nearest_other_video_sim, nearest_same_video_scene_sim)
        internal_centroid_sim = float(np.mean(similarities))
        uniqueness_margin = internal_centroid_sim - outside_scene_sim

        if nearest_other_video_sim >= max_other_video_sim:
            discarded_other_video += 1
            continue
        if nearest_same_video_scene_sim >= max_outside_scene_sim:
            discarded_same_video_repeat += 1
            continue
        if uniqueness_margin < min_uniqueness_margin:
            discarded_low_margin += 1
            continue
        
        candidate_record = records[best_overall_index]
        cluster_frame_ids = [records[index]["frame_id"] for index in sorted(component, key=lambda i: records[i]["frame_number"])]
        unique_candidates.append({
            "_component_indices": component,
            "frame_id": candidate_record["frame_id"],
            "video_id": candidate_record["video_id"],
            "frame_number": candidate_record["frame_number"],
            "cluster_size": len(component),
            "frame_range": [min_frame, max_frame],
            "ground_truth_frame_ids": cluster_frame_ids,
            "uniqueness_margin": round(uniqueness_margin, 6),
        })

    unique_candidates.sort(
        key=lambda item: (
            item["uniqueness_margin"],
            item["cluster_size"],
        ),
        reverse=True,
    )
    if max_candidates > 0:
        unique_candidates = unique_candidates[:max_candidates]

    protected_frame_ids = {
        frame_id
        for candidate in unique_candidates
        for frame_id in candidate["ground_truth_frame_ids"]
    }
    for candidate in unique_candidates:
        candidate["ground_truth_frame_ids"] = expand_ground_truth(
            candidate,
            records,
            x_norm,
            by_video,
            protected_frame_ids,
            ground_truth_neighbor_count,
            ground_truth_max_gap,
            ground_truth_min_similarity,
        )
        del candidate["_component_indices"]

    unique_candidates = [
        {"id": f"c{index:03d}", **candidate}
        for index, candidate in enumerate(unique_candidates, start=1)
    ]

    logger.info("Filtering complete:")
    logger.info("  - Discarded (too small): %d", discarded_small)
    logger.info("  - Discarded (spanning multiple videos): %d", discarded_multi_video)
    logger.info("  - Discarded (exceeding frame gap / multiple scenes): %d", discarded_multi_scene)
    logger.info("  - Discarded (similar other-video scene): %d", discarded_other_video)
    logger.info("  - Discarded (similar repeated same-video scene): %d", discarded_same_video_repeat)
    logger.info("  - Discarded (low uniqueness margin): %d", discarded_low_margin)
    logger.info("  - Selected unique candidates: %d", len(unique_candidates))

    return unique_candidates


def main() -> None:
    config = Config()
    try:
        features_dir = config.features_dir or sample_subdir(config.sample_root.resolve(), "PECore-features")
        records = remove_excluded_frames(load_all_features(features_dir), config.excluded_frame_ids)
        candidates = cluster_and_filter(
            records,
            threshold=config.similarity_threshold,
            max_frame_gap=config.max_frame_gap,
            max_other_video_sim=config.max_other_video_sim,
            max_outside_scene_sim=config.max_outside_scene_sim,
            scene_gap=config.scene_gap,
            min_uniqueness_margin=config.min_uniqueness_margin,
            min_cluster_size=config.min_cluster_size,
            ground_truth_neighbor_count=config.ground_truth_neighbor_count,
            ground_truth_max_gap=config.ground_truth_max_gap,
            ground_truth_min_similarity=config.ground_truth_min_similarity,
            max_candidates=config.max_candidates,
        )

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "purpose": "unique-scene candidates for text retrieval benchmarking",
                        "feature_count": len(records),
                        "candidate_count": len(candidates),
                        "similarity_threshold": config.similarity_threshold,
                        "excluded_frame_ids": sorted(config.excluded_frame_ids),
                        "ground_truth_expansion": {
                            "neighbor_count_each_side": config.ground_truth_neighbor_count,
                            "max_frame_gap": config.ground_truth_max_gap,
                            "min_visual_similarity": config.ground_truth_min_similarity,
                            "protect_other_candidate_core_frames": True,
                        },
                    },
                    "candidates": candidates,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("Saved %d unique candidate keyframes to %s.", len(candidates), config.output_path)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)


if __name__ == "__main__":
    main()
