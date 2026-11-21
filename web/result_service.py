import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.mo_utils import hv_wfg, ndsort_ess
from web.dataset_service import (
    DEFAULT_REF_POINTS,
    RESULT_DIR,
    SAVED_MODEL_DIR,
    VRPLIB_PATTERN,
    resolve_dataset_path,
)
from route_solver.utils import zoom  # noqa: E402

VALUE_GEN_PATTERN = re.compile(r'^value-(\d+)\.json$')


def _safe_subdir(base_dir: str, rel_path: str) -> str:
    clean_rel = rel_path.strip().replace('/', os.sep).replace('\\', os.sep)
    target = os.path.abspath(os.path.join(base_dir, clean_rel))
    root = os.path.abspath(base_dir)
    if os.path.commonpath([root, target]) != root or target == root:
        raise ValueError('Đường dẫn kết quả không hợp lệ.')
    if not os.path.isdir(target):
        raise FileNotFoundError(f'Không tìm thấy thư mục kết quả: {rel_path}')
    return target


def _parse_result_id(result_id: str) -> Tuple[str, str]:
    if '::' not in result_id:
        raise ValueError('result_id không đúng định dạng.')
    kind, rel_path = result_id.split('::', 1)
    if kind not in ('eam', 'cmma'):
        raise ValueError(f'Loại kết quả không hỗ trợ: {kind}')
    return kind, rel_path


def _parse_eam_eval_txt(eval_txt_path: str) -> Dict[str, Any]:
    if not os.path.isfile(eval_txt_path):
        return {}
    with open(eval_txt_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read().strip()

    parsed: Dict[str, Any] = {'raw_text': text}
    for line in text.splitlines():
        line_s = line.strip()
        if line_s.startswith('hv:'):
            try:
                parsed['hv'] = float(line_s.split(':', 1)[1].strip())
            except ValueError:
                pass
        elif line_s.startswith('sample_hv:'):
            try:
                parsed['sample_hv'] = float(line_s.split(':', 1)[1].strip())
            except ValueError:
                pass
        elif line_s.startswith('avg_best:'):
            try:
                parsed['avg_best'] = json.loads(line_s.split(':', 1)[1].strip())
            except json.JSONDecodeError:
                pass
        elif line_s.startswith('avg_avg:'):
            try:
                parsed['avg_avg'] = json.loads(line_s.split(':', 1)[1].strip())
            except json.JSONDecodeError:
                pass
    return parsed


def _dataset_name_from_option(opt: Dict[str, Any]) -> Optional[str]:
    for key in ('val_dataset', 'load_file'):
        val = opt.get(key)
        if isinstance(val, str) and val.strip():
            return os.path.basename(val.strip())
    return None


def scan_results() -> Dict[str, Any]:
    eam_results: List[Dict[str, Any]] = []
    if os.path.isdir(SAVED_MODEL_DIR):
        for dirpath, _, filenames in os.walk(SAVED_MODEL_DIR):
            if 'objv.json' not in filenames:
                continue
            rel = os.path.relpath(dirpath, SAVED_MODEL_DIR).replace('\\', '/')
            opt: Dict[str, Any] = {}
            opt_path = os.path.join(dirpath, 'option.json')
            if os.path.isfile(opt_path):
                with open(opt_path, 'r', encoding='utf-8') as f:
                    opt = json.load(f)
            eval_info = _parse_eam_eval_txt(os.path.join(dirpath, 'eval.txt'))
            eam_results.append(
                {
                    'result_id': f'eam::{rel}',
                    'kind': 'eam',
                    'title': f'E-AM | {rel}',
                    'rel_path': rel,
                    'dataset': _dataset_name_from_option(opt),
                    'eval_type': opt.get('eval_type', ''),
                    'customer_size': opt.get('val_customer_size', opt.get('customer_size')),
                    'vehicle_num': opt.get('val_vehicle_num', opt.get('vehicle_num')),
                    'val_range': opt.get('val_range'),
                    'hv': eval_info.get('hv'),
                    'has_action': 'action.json' in filenames,
                    'mtime': os.path.getmtime(os.path.join(dirpath, 'objv.json')),
                }
            )
    eam_results.sort(key=lambda r: r['mtime'], reverse=True)

    cmma_results: List[Dict[str, Any]] = []
    if os.path.isdir(RESULT_DIR):
        for dirpath, _, filenames in os.walk(RESULT_DIR):
            gens: List[int] = []
            for name in filenames:
                m = VALUE_GEN_PATTERN.match(name)
                if m:
                    gens.append(int(m.group(1)))
            if not gens:
                continue
            gens.sort()
            latest_gen = gens[-1]
            rel = os.path.relpath(dirpath, RESULT_DIR).replace('\\', '/')
            opt = {}
            opt_path = os.path.join(dirpath, 'option.json')
            if os.path.isfile(opt_path):
                with open(opt_path, 'r', encoding='utf-8') as f:
                    opt = json.load(f)
            hv_val = None
            latest_val_path = os.path.join(dirpath, f'value-{latest_gen}.json')
            with open(latest_val_path, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
                hv_val = val_data.get('hv')

            cmma_results.append(
                {
                    'result_id': f'cmma::{rel}',
                    'kind': 'cmma',
                    'title': f'CMMA | {rel} (gen {latest_gen})',
                    'rel_path': rel,
                    'dataset': _dataset_name_from_option(opt),
                    'problem_index': opt.get('problem_index', 0),
                    'vehicle_num': opt.get('vehicle_num'),
                    'generations': gens,
                    'latest_gen': latest_gen,
                    'hv': hv_val,
                    'has_action': f'population-{latest_gen}.json' in filenames,
                    'mtime': os.path.getmtime(latest_val_path),
                }
            )
    cmma_results.sort(key=lambda r: r['mtime'], reverse=True)

    return {'eam': eam_results, 'cmma': cmma_results}


def _compute_pareto_stats(
    objs: List[List[float]], ref_point: Optional[List[float]] = None
) -> Dict[str, Any]:
    if not objs:
        return {
            'points': [],
            'nondominated_count': 0,
            'min_sum_idx': None,
            'min_max_idx': None,
            'knee_idx': None,
            'best_sum': None,
            'best_max': None,
            'hv_ratio': None,
        }

    arr = np.asarray(objs, dtype=np.float64)
    levels, _ = ndsort_ess(arr, need_level=1)
    is_nd = levels == 1

    points = [
        {
            'sol_idx': idx,
            'f1': float(arr[idx, 0]),
            'f2': float(arr[idx, 1]),
            'is_nondominated': bool(is_nd[idx]),
        }
        for idx in range(arr.shape[0])
    ]

    nd_indices = np.flatnonzero(is_nd)
    candidate_indices = nd_indices if nd_indices.size > 0 else np.arange(arr.shape[0])

    min_sum_idx = int(candidate_indices[int(np.argmin(arr[candidate_indices, 0]))])
    min_max_idx = int(candidate_indices[int(np.argmin(arr[candidate_indices, 1]))])

    # Knee point: closest to normalized ideal point on non-dominated front
    sub = arr[candidate_indices]
    f_min = sub.min(axis=0)
    f_max = sub.max(axis=0)
    span = np.where(f_max > f_min, f_max - f_min, 1.0)
    norm_dist = np.linalg.norm((sub - f_min) / span, axis=1)
    knee_idx = int(candidate_indices[int(np.argmin(norm_dist))])

    hv_ratio = None
    if ref_point is not None and len(ref_point) == 2 and ref_point[0] > 0 and ref_point[1] > 0:
        hv = hv_wfg(arr, np.asarray(ref_point, dtype=np.float64))
        hv_ratio = float(hv / (ref_point[0] * ref_point[1]))

    return {
        'points': points,
        'nondominated_count': int(np.sum(is_nd)),
        'min_sum_idx': min_sum_idx,
        'min_max_idx': min_max_idx,
        'knee_idx': knee_idx,
        'best_sum': float(arr[:, 0].min()),
        'best_max': float(arr[:, 1].min()),
        'hv_ratio': hv_ratio,
    }


def _load_eam_dir(rel_path: str) -> Tuple[str, Dict[str, Any], List[Any]]:
    run_dir = _safe_subdir(SAVED_MODEL_DIR, rel_path)
    opt: Dict[str, Any] = {}
    opt_path = os.path.join(run_dir, 'option.json')
    if os.path.isfile(opt_path):
        with open(opt_path, 'r', encoding='utf-8') as f:
            opt = json.load(f)
    with open(os.path.join(run_dir, 'objv.json'), 'r', encoding='utf-8') as f:
        objv = json.load(f)
    return run_dir, opt, objv


def _load_cmma_dir(
    rel_path: str, gen: Optional[int] = None
) -> Tuple[str, Dict[str, Any], int, List[int], Dict[str, Any]]:
    run_dir = _safe_subdir(RESULT_DIR, rel_path)
    opt: Dict[str, Any] = {}
    opt_path = os.path.join(run_dir, 'option.json')
    if os.path.isfile(opt_path):
        with open(opt_path, 'r', encoding='utf-8') as f:
            opt = json.load(f)

    gens = sorted(
        int(m.group(1))
        for name in os.listdir(run_dir)
        if (m := VALUE_GEN_PATTERN.match(name)) is not None
    )
    if not gens:
        raise FileNotFoundError('Không tìm thấy file value-*.json nào.')
    chosen_gen = gens[-1] if gen is None else int(gen)
    val_path = os.path.join(run_dir, f'value-{chosen_gen}.json')
    if not os.path.isfile(val_path):
        raise FileNotFoundError(f'Không tìm thấy value-{chosen_gen}.json')

    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    return run_dir, opt, chosen_gen, gens, val_data


def get_result_summary(result_id: str, gen: Optional[int] = None) -> Dict[str, Any]:
    kind, rel_path = _parse_result_id(result_id)

    if kind == 'eam':
        run_dir, opt, objv = _load_eam_dir(rel_path)
        eval_info = _parse_eam_eval_txt(os.path.join(run_dir, 'eval.txt'))
        val_range = opt.get('val_range', [0, len(objv)])
        has_action = os.path.isfile(os.path.join(run_dir, 'action.json'))
        cus = opt.get('val_customer_size', opt.get('customer_size', 20))
        ref_point = opt.get('ref_point') or list(DEFAULT_REF_POINTS.get(cus, (30.0, 3.0)))

        return {
            'result_id': result_id,
            'kind': 'eam',
            'rel_path': rel_path,
            'dataset': _dataset_name_from_option(opt),
            'val_range': val_range,
            'num_instances': len(objv),
            'customer_size': cus,
            'vehicle_num': opt.get('val_vehicle_num', opt.get('vehicle_num')),
            'eval_type': opt.get('eval_type'),
            'ref_point': ref_point,
            'hv': eval_info.get('hv'),
            'sample_hv': eval_info.get('sample_hv'),
            'avg_best': eval_info.get('avg_best'),
            'avg_avg': eval_info.get('avg_avg'),
            'raw_text': eval_info.get('raw_text', ''),
            'has_action': has_action,
            'generations': [],
            'chosen_gen': None,
        }

    run_dir, opt, chosen_gen, gens, val_data = _load_cmma_dir(rel_path, gen)
    hv_history = []
    for g in gens:
        with open(os.path.join(run_dir, f'value-{g}.json'), 'r', encoding='utf-8') as f:
            vd = json.load(f)
            hv_history.append({'gen': g, 'hv': vd.get('hv')})

    log_tail = ''
    log_path = os.path.join(run_dir, 'log.txt')
    if os.path.isfile(log_path):
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()
            log_tail = '\n'.join(lines[-25:])

    return {
        'result_id': result_id,
        'kind': 'cmma',
        'rel_path': rel_path,
        'dataset': _dataset_name_from_option(opt),
        'problem_index': opt.get('problem_index', 0),
        'num_instances': 1,
        'vehicle_num': opt.get('vehicle_num'),
        'population_size': opt.get('population_size'),
        'ref_point': val_data.get('ref') or opt.get('ref_point'),
        'hv': val_data.get('hv'),
        'hv_history': hv_history,
        'raw_text': log_tail,
        'has_action': os.path.isfile(os.path.join(run_dir, f'population-{chosen_gen}.json')),
        'generations': gens,
        'chosen_gen': chosen_gen,
    }


def get_instance_pareto(
    result_id: str, instance_offset: int = 0, gen: Optional[int] = None
) -> Dict[str, Any]:
    kind, rel_path = _parse_result_id(result_id)

    if kind == 'eam':
        run_dir, opt, objv = _load_eam_dir(rel_path)
        if instance_offset < 0 or instance_offset >= len(objv):
            raise IndexError(f'instance_offset phải nằm trong [0, {len(objv) - 1}].')
        cus = opt.get('val_customer_size', opt.get('customer_size', 20))
        ref_point = opt.get('ref_point') or list(DEFAULT_REF_POINTS.get(cus, (30.0, 3.0)))
        stats = _compute_pareto_stats(objv[instance_offset], ref_point)
        val_range = opt.get('val_range', [0, len(objv)])
        return {
            'result_id': result_id,
            'kind': 'eam',
            'dataset': _dataset_name_from_option(opt),
            'dataset_problem_index': int(val_range[0]) + instance_offset,
            'instance_offset': instance_offset,
            'ref_point': ref_point,
            'has_action': os.path.isfile(os.path.join(run_dir, 'action.json')),
            **stats,
        }

    run_dir, opt, chosen_gen, _, val_data = _load_cmma_dir(rel_path, gen)
    ref_point = val_data.get('ref') or opt.get('ref_point')
    stats = _compute_pareto_stats(val_data.get('obj', []), ref_point)
    return {
        'result_id': result_id,
        'kind': 'cmma',
        'dataset': _dataset_name_from_option(opt),
        'dataset_problem_index': int(opt.get('problem_index', 0)),
        'instance_offset': 0,
        'chosen_gen': chosen_gen,
        'ref_point': ref_point,
        'has_action': os.path.isfile(os.path.join(run_dir, f'population-{chosen_gen}.json')),
        **stats,
    }


def _clean_eam_vehicle_route(seq: List[int]) -> List[int]:
    """Collapse consecutive zeros and ensure a clean [0, ..., 0] tour if non-empty."""
    cleaned: List[int] = []
    for node in seq:
        n = int(node)
        if not cleaned or cleaned[-1] != n:
            cleaned.append(n)
    while cleaned and cleaned[0] == 0:
        cleaned.pop(0)
    while cleaned and cleaned[-1] == 0:
        cleaned.pop()
    if not cleaned:
        return [0, 0]
    return [0] + cleaned + [0]


def _split_cmma_1d_route(seq: List[int], vehicle_num: Optional[int]) -> List[List[int]]:
    """Split CMMA 1D giant route [0, ..., 0, ..., 0] into per-vehicle [0, ..., 0] tours."""
    routes: List[List[int]] = []
    cur: List[int] = []
    for raw_node in seq:
        node = int(raw_node)
        if node == 0:
            if cur:
                routes.append([0] + cur + [0])
                cur = []
        else:
            cur.append(node)
    if cur:
        routes.append([0] + cur + [0])

    if vehicle_num is not None:
        while len(routes) < vehicle_num:
            routes.append([0, 0])
    return routes


def _load_problem_coords(
    dataset_name: Optional[str], problem_index: int, zoom_on: bool = False
) -> Tuple[np.ndarray, Optional[float]]:
    if not dataset_name:
        raise FileNotFoundError('Không rõ tên file dataset trong option.json.')
    path = resolve_dataset_path(dataset_name)
    data = np.load(path)
    if problem_index < 0 or problem_index >= data.shape[0]:
        raise IndexError(
            f'Chỉ số bài toán {problem_index} vượt ngoài phạm vi dataset {dataset_name} ({data.shape[0]}).'
        )
    ins = data[problem_index].copy()
    ratio_val: Optional[float] = None
    if zoom_on or VRPLIB_PATTERN.search(dataset_name):
        tensor = torch.tensor(ins, dtype=torch.float32).unsqueeze(0)
        _, ratio = zoom(tensor)
        ratio_val = float(ratio.item())
    return ins, ratio_val


def _evaluate_vehicle_routes(
    problem: np.ndarray, routes: List[List[int]]
) -> Dict[str, Any]:
    xy = problem[:, :2]
    has_demand = problem.shape[1] >= 3
    demands = problem[:, 2] if has_demand else None

    vehicles: List[Dict[str, Any]] = []
    for v_idx, route in enumerate(routes):
        dist = 0.0
        for a, b in zip(route[:-1], route[1:]):
            dist += float(np.linalg.norm(xy[b] - xy[a]))
        customers = [n for n in route if n != 0]
        veh_demand = float(demands[customers].sum()) if demands is not None and customers else 0.0
        vehicles.append(
            {
                'vehicle_id': v_idx + 1,
                'route': route,
                'customer_count': len(customers),
                'distance': dist,
                'demand': veh_demand,
            }
        )

    distances = [v['distance'] for v in vehicles]
    active_distances = [d for d in distances if d > 0]
    obj_sum = float(sum(distances))
    obj_max = float(max(distances)) if distances else 0.0
    balance_gap = (
        float(max(active_distances) - min(active_distances))
        if len(active_distances) >= 2
        else 0.0
    )

    nodes = [
        {
            'id': idx,
            'role': 'Depot' if idx == 0 else 'Khách hàng',
            'x': float(problem[idx, 0]),
            'y': float(problem[idx, 1]),
            'demand': float(problem[idx, 2]) if has_demand else None,
        }
        for idx in range(problem.shape[0])
    ]

    return {
        'nodes': nodes,
        'has_demand': has_demand,
        'vehicles': vehicles,
        'objective_sum': obj_sum,
        'objective_max': obj_max,
        'balance_gap': balance_gap,
    }


def get_solution_routes(
    result_id: str,
    instance_offset: int = 0,
    sol_idx: int = 0,
    gen: Optional[int] = None,
) -> Dict[str, Any]:
    kind, rel_path = _parse_result_id(result_id)

    if kind == 'eam':
        run_dir, opt, objv = _load_eam_dir(rel_path)
        action_path = os.path.join(run_dir, 'action.json')
        if not os.path.isfile(action_path):
            raise FileNotFoundError(
                'Kết quả E-AM này không có `action.json` (có thể là baseline 1D hoặc chưa eval).'
            )
        with open(action_path, 'r', encoding='utf-8') as f:
            actions = json.load(f)
        if instance_offset < 0 or instance_offset >= len(actions):
            raise IndexError('instance_offset vượt ngoài phạm vi.')
        if sol_idx < 0 or sol_idx >= len(actions[instance_offset]):
            raise IndexError('sol_idx vượt ngoài phạm vi.')

        val_range = opt.get('val_range', [0, len(actions)])
        dataset_name = _dataset_name_from_option(opt)
        problem_idx = int(val_range[0]) + instance_offset
        problem, _ = _load_problem_coords(
            dataset_name, problem_idx, zoom_on=bool(opt.get('zoom_on', False))
        )

        raw_vehicle_routes = actions[instance_offset][sol_idx]
        routes = [_clean_eam_vehicle_route(r) for r in raw_vehicle_routes]
        eval_res = _evaluate_vehicle_routes(problem, routes)
        recorded = objv[instance_offset][sol_idx]
        return {
            'result_id': result_id,
            'kind': 'eam',
            'dataset': dataset_name,
            'problem_index': problem_idx,
            'sol_idx': sol_idx,
            'recorded_obj': recorded,
            **eval_res,
        }

    run_dir, opt, chosen_gen, _, val_data = _load_cmma_dir(rel_path, gen)
    pop_path = os.path.join(run_dir, f'population-{chosen_gen}.json')
    if not os.path.isfile(pop_path):
        raise FileNotFoundError(f'Không tìm thấy population-{chosen_gen}.json')
    with open(pop_path, 'r', encoding='utf-8') as f:
        population = json.load(f)
    if sol_idx < 0 or sol_idx >= len(population):
        raise IndexError('sol_idx vượt ngoài phạm vi quần thể.')

    dataset_name = _dataset_name_from_option(opt)
    problem_idx = int(opt.get('problem_index', 0))
    problem, _ = _load_problem_coords(dataset_name, problem_idx, zoom_on=False)
    routes = _split_cmma_1d_route(population[sol_idx], opt.get('vehicle_num'))
    eval_res = _evaluate_vehicle_routes(problem, routes)
    recorded = val_data.get('obj', [])[sol_idx] if sol_idx < len(val_data.get('obj', [])) else None
    return {
        'result_id': result_id,
        'kind': 'cmma',
        'dataset': dataset_name,
        'problem_index': problem_idx,
        'chosen_gen': chosen_gen,
        'sol_idx': sol_idx,
        'recorded_obj': recorded,
        **eval_res,
    }


def compare_results(
    result_a_id: str,
    inst_a: int = 0,
    gen_a: Optional[int] = None,
    result_b_id: Optional[str] = None,
    inst_b: int = 0,
    gen_b: Optional[int] = None,
) -> Dict[str, Any]:
    pareto_a = get_instance_pareto(result_a_id, inst_a, gen_a)
    pareto_b = (
        get_instance_pareto(result_b_id, inst_b, gen_b)
        if result_b_id
        else None
    )

    shared_ref = pareto_a.get('ref_point') or (
        pareto_b.get('ref_point') if pareto_b else None
    )
    if pareto_b and shared_ref and len(shared_ref) == 2:
        pts_a = [[p['f1'], p['f2']] for p in pareto_a['points']]
        pts_b = [[p['f1'], p['f2']] for p in pareto_b['points']]
        pareto_a['shared_hv'] = float(
            hv_wfg(np.asarray(pts_a, dtype=np.float64), np.asarray(shared_ref, dtype=np.float64))
            / (shared_ref[0] * shared_ref[1])
        )
        pareto_b['shared_hv'] = float(
            hv_wfg(np.asarray(pts_b, dtype=np.float64), np.asarray(shared_ref, dtype=np.float64))
            / (shared_ref[0] * shared_ref[1])
        )

    return {
        'primary': pareto_a,
        'secondary': pareto_b,
        'shared_ref': shared_ref,
    }
