from dataclasses import asdict, dataclass
import io
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from problem.cvrp import CAPACITIES  # noqa: E402
from problem.cvrp_balance import generate_datasets as generate_cvrp_balance  # noqa: E402
from route_solver.utils import zoom  # noqa: E402

DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
SAVED_MODEL_DIR = os.path.join(ROOT_DIR, 'saved_model')
RESULT_DIR = os.path.join(ROOT_DIR, 'result')
LOG_DIR = os.path.join(ROOT_DIR, 'log')

VRPLIB_PATTERN = re.compile(r'p\d+ E\d+-\d+[ec]\.npy$')
VEHICLE_PATTERN = re.compile(r'-v(\d+)\.npy$')
BENCHMARK_VEHICLE_PATTERN = re.compile(r'-(\d+)[ec]\.npy$')

DEFAULT_VEHICLES: Dict[int, int] = {
    20: 4,
    50: 7,
    100: 11,
    200: 14,
    500: 24,
    1000: 32,
}

DEFAULT_REF_POINTS: Dict[int, Tuple[float, float]] = {
    20: (30.0, 3.0),
    50: (50.0, 3.0),
    75: (65.0, 3.0),
    100: (80.0, 3.0),
    120: (96.0, 3.0),
    150: (120.0, 3.0),
    199: (180.0, 3.0),
    200: (180.0, 3.0),
    500: (400.0, 3.0),
    1000: (800.0, 3.0),
}


@dataclass
class DatasetMeta:
    filename: str
    size_bytes: int
    shape: List[int]
    num_instances: int
    node_count: int
    feature_dim: int
    customer_size: int
    vehicle_num: Optional[int]
    kind: str
    auto_zoom: bool
    ref_point: List[float]


def _read_npy_shape(path: str) -> Tuple[int, ...]:
    """Read only the .npy header without loading the full array into memory."""
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(f)
        else:
            shape, _, _ = np.lib.format._read_array_header(f, version)
        return tuple(int(x) for x in shape)


def resolve_dataset_path(filename: str, datasets_dir: str = DATASETS_DIR) -> str:
    safe_name = os.path.basename(filename)
    if safe_name != filename or not safe_name.endswith('.npy'):
        raise ValueError('Tên file dataset không hợp lệ.')
    path = os.path.join(datasets_dir, safe_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Không tìm thấy dataset: {safe_name}')
    return path


def infer_vehicle_num(filename: str, customer_size: int) -> Optional[int]:
    m_synth = VEHICLE_PATTERN.search(filename)
    if m_synth:
        return int(m_synth.group(1))
    if VRPLIB_PATTERN.search(filename):
        m_bench = BENCHMARK_VEHICLE_PATTERN.search(filename)
        if m_bench:
            return int(m_bench.group(1))
    return DEFAULT_VEHICLES.get(customer_size)


def build_dataset_meta(filename: str, path: str) -> Optional[DatasetMeta]:
    shape = _read_npy_shape(path)
    if len(shape) != 3:
        return None
    num_instances, node_count, feature_dim = shape
    customer_size = max(0, node_count - 1) if feature_dim >= 3 else node_count
    auto_zoom = VRPLIB_PATTERN.search(filename) is not None
    kind = 'Benchmark VRPLIB' if auto_zoom else 'Tổng hợp'
    ref = DEFAULT_REF_POINTS.get(customer_size, (80.0, 3.0))
    return DatasetMeta(
        filename=filename,
        size_bytes=os.path.getsize(path),
        shape=list(shape),
        num_instances=num_instances,
        node_count=node_count,
        feature_dim=feature_dim,
        customer_size=customer_size,
        vehicle_num=infer_vehicle_num(filename, customer_size),
        kind=kind,
        auto_zoom=auto_zoom,
        ref_point=[ref[0], ref[1]],
    )


def scan_datasets(datasets_dir: str = DATASETS_DIR) -> List[Dict[str, Any]]:
    if not os.path.isdir(datasets_dir):
        return []
    result: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(datasets_dir)):
        if not name.endswith('.npy'):
            continue
        path = os.path.join(datasets_dir, name)
        if not os.path.isfile(path):
            continue
        meta = build_dataset_meta(name, path)
        if meta is not None:
            result.append(asdict(meta))
    return result


def get_dataset_summary(filename: str) -> Dict[str, Any]:
    path = resolve_dataset_path(filename)
    meta = build_dataset_meta(filename, path)
    if meta is None:
        raise ValueError(f'Dataset {filename} không đúng định dạng mảng 3 chiều.')

    data = np.load(path)
    xy = data[:, :, :2]
    x_min, x_max = float(xy[:, :, 0].min()), float(xy[:, :, 0].max())
    y_min, y_max = float(xy[:, :, 1].min()), float(xy[:, :, 1].max())
    out_of_unit_box = x_min < 0.0 or x_max > 1.0 or y_min < 0.0 or y_max > 1.0

    summary: Dict[str, Any] = {
        'meta': asdict(meta),
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'out_of_unit_box': out_of_unit_box,
        'has_demand': data.shape[2] >= 3,
    }

    if data.shape[2] >= 3:
        demands = data[:, :, 2]
        depot_demand_max = float(np.abs(demands[:, 0]).max())
        total_demands = demands[:, 1:].sum(axis=1)
        min_vehicles_needed = np.ceil(np.maximum(0.0, total_demands - 1e-5)).astype(int) + 1
        feasible_count = (
            int(np.sum(meta.vehicle_num >= min_vehicles_needed))
            if meta.vehicle_num is not None
            else None
        )

        hist_counts, hist_edges = np.histogram(total_demands, bins=min(16, max(4, data.shape[0] // 8)))
        summary.update(
            {
                'depot_demand_ok': depot_demand_max < 1e-6,
                'total_demand_mean': float(total_demands.mean()),
                'total_demand_min': float(total_demands.min()),
                'total_demand_max': float(total_demands.max()),
                'feasible_count': feasible_count,
                'histogram': {
                    'counts': [int(c) for c in hist_counts],
                    'edges': [float(e) for e in hist_edges],
                },
            }
        )

    return summary


def build_cli_commands(meta: DatasetMeta, problem_index: int) -> Dict[str, str]:
    rel_path = f'datasets/{meta.filename}'
    veh = meta.vehicle_num or 4
    val_batch = 1 if meta.num_instances == 1 else min(100, meta.num_instances)
    ref = meta.ref_point

    eam_eval = (
        f'.\\.venv\\Scripts\\python.exe .\\src\\main.py '
        f'--eval_only --no_log --problem mo_cvrp '
        f'--customer_size {meta.customer_size} --vehicle_num {veh} '
        f'--val_range 0 {meta.num_instances} --val_batch_size {val_batch} '
        f'--val_dataset "{rel_path}" --eval_type pareto_sample --sample_times 32 '
        f'--load_path "<duong_dan_model.pt>"'
    )

    cmma_run = (
        f'.\\.venv\\Scripts\\python.exe .\\src-CMMA\\main.py '
        f'--load_file "{rel_path}" --problem_index {problem_index} '
        f'--vehicle_num {veh} --population_size 50 --N_gen 200 --save_per_gen 50 '
        f'--ref_point {ref[0]:g} {ref[1]:g}'
    )

    return {'eam_eval': eam_eval, 'cmma_run': cmma_run}


def get_dataset_instance(
    filename: str, problem_index: int, apply_zoom: Optional[bool] = None
) -> Dict[str, Any]:
    path = resolve_dataset_path(filename)
    meta = build_dataset_meta(filename, path)
    if meta is None:
        raise ValueError(f'Dataset {filename} không đúng định dạng mảng 3 chiều.')

    data = np.load(path)
    if problem_index < 0 or problem_index >= data.shape[0]:
        raise IndexError(f'problem_index phải nằm trong [0, {data.shape[0] - 1}].')

    instance = data[problem_index].copy()
    if apply_zoom is None:
        xy = instance[:, :2]
        out_of_box = bool(xy.min() < 0.0 or xy.max() > 1.0)
        apply_zoom = meta.auto_zoom or out_of_box

    zoom_ratio: Optional[float] = None
    if apply_zoom:
        tensor = torch.tensor(instance, dtype=torch.float32).unsqueeze(0)
        zoomed, ratio = zoom(tensor)
        instance = zoomed.squeeze(0).numpy()
        zoom_ratio = float(ratio.item())

    has_demand = instance.shape[1] >= 3
    nodes = [
        {
            'id': idx,
            'role': 'Depot' if idx == 0 else 'Khách hàng',
            'x': float(instance[idx, 0]),
            'y': float(instance[idx, 1]),
            'demand': float(instance[idx, 2]) if has_demand else None,
        }
        for idx in range(instance.shape[0])
    ]

    total_demand = float(instance[1:, 2].sum()) if has_demand else None
    min_vehicles_needed = (
        int(math.ceil(max(0.0, total_demand - 1e-5)) + 1)
        if total_demand is not None
        else None
    )

    return {
        'filename': filename,
        'problem_index': problem_index,
        'zoom_applied': bool(apply_zoom),
        'zoom_ratio': zoom_ratio,
        'has_demand': has_demand,
        'total_demand': total_demand,
        'min_vehicles_needed': min_vehicles_needed,
        'nodes': nodes,
        'cli_commands': build_cli_commands(meta, problem_index),
    }


def save_uploaded_dataset(
    file_bytes: bytes, raw_name: str, datasets_dir: str = DATASETS_DIR
) -> Dict[str, Any]:
    safe_name = os.path.basename(raw_name.strip())
    if not safe_name.endswith('.npy'):
        raise ValueError('File tải lên phải có phần mở rộng .npy')

    arr = np.load(io.BytesIO(file_bytes), allow_pickle=False)
    if arr.ndim != 3 or arr.shape[2] not in (2, 3):
        raise ValueError(
            f'Shape không hợp lệ: {arr.shape}. Cần mảng 3 chiều (số_bài_toán, số_node, 2 hoặc 3).'
        )

    os.makedirs(datasets_dir, exist_ok=True)
    target_path = os.path.join(datasets_dir, safe_name)
    np.save(target_path, arr)
    meta = build_dataset_meta(safe_name, target_path)
    return asdict(meta) if meta is not None else {'filename': safe_name, 'shape': list(arr.shape)}


def create_random_dataset(
    num_instances: int,
    customer_size: int,
    vehicle_num: int,
    vehicle_capacity: int,
    seed: int,
    filename: str,
    datasets_dir: str = DATASETS_DIR,
) -> Dict[str, Any]:
    safe_name = os.path.basename(filename.strip())
    if not safe_name:
        safe_name = f'custom_cvrp{customer_size}-v{vehicle_num}.npy'
    if not safe_name.endswith('.npy'):
        safe_name += '.npy'

    os.makedirs(datasets_dir, exist_ok=True)
    target_path = os.path.join(datasets_dir, safe_name)

    torch.manual_seed(seed)
    generate_cvrp_balance(
        batch_size=num_instances,
        customer_size=customer_size,
        vehicle_num=vehicle_num,
        vehicle_capacity=vehicle_capacity,
        save_npy_path=target_path,
    )
    meta = build_dataset_meta(safe_name, target_path)
    return asdict(meta) if meta is not None else {'filename': safe_name}
