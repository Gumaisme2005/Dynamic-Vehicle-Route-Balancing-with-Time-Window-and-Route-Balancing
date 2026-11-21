import codecs
from dataclasses import dataclass, field
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Tuple

from web.dataset_service import (
    DATASETS_DIR,
    DEFAULT_REF_POINTS,
    DEFAULT_VEHICLES,
    ROOT_DIR,
    SAVED_MODEL_DIR,
    build_dataset_meta,
    resolve_dataset_path,
)

POMO_TSP20_CKPT = os.path.join(
    ROOT_DIR,
    'src',
    'route_solver',
    'pomo_tsp_solver',
    'result',
    'saved_tsp20_model',
    'checkpoint-510.pt',
)


@dataclass
class JobRecord:
    job_id: str
    kind: str
    title: str
    command: List[str]
    command_preview: str
    status: str = 'running'
    pid: Optional[int] = None
    return_code: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    progress: str = ''
    logs: List[str] = field(default_factory=list)
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self, include_logs: bool = False) -> Dict[str, Any]:
        with self.lock:
            data = {
                'job_id': self.job_id,
                'kind': self.kind,
                'title': self.title,
                'command_preview': self.command_preview,
                'status': self.status,
                'pid': self.pid,
                'return_code': self.return_code,
                'started_at': self.started_at,
                'finished_at': self.finished_at,
                'progress': self.progress,
                'log_count': len(self.logs),
            }
            if include_logs:
                data['logs'] = list(self.logs)
            return data


_JOBS: Dict[str, JobRecord] = {}
_JOBS_LOCK = threading.Lock()
_TB_LOCK = threading.Lock()
_TB_PROC: Optional[subprocess.Popen] = None
_TB_PORT: int = 6006

MAX_LOG_LINES = 5000


def _format_cmd_preview(cmd: List[str]) -> str:
    parts = []
    for i, token in enumerate(cmd):
        if i == 0 and token == sys.executable:
            parts.append(r'.\.venv\Scripts\python.exe')
        elif ' ' in token or token == '':
            parts.append(f'"{token}"')
        else:
            parts.append(token)
    return ' '.join(parts)


def _resolve_saved_checkpoint(rel_or_abs: str) -> str:
    if not rel_or_abs or not rel_or_abs.strip():
        raise ValueError('Vui lòng chọn file checkpoint (.pt).')
    candidate = rel_or_abs.strip()
    if not os.path.isabs(candidate):
        candidate = os.path.join(ROOT_DIR, candidate)
    abs_target = os.path.abspath(candidate)
    abs_root = os.path.abspath(SAVED_MODEL_DIR)
    if os.path.commonpath([abs_root, abs_target]) != abs_root or abs_target == abs_root:
        raise ValueError('Đường dẫn checkpoint phải nằm trong thư mục saved_model/.')
    if not os.path.isfile(abs_target) or not abs_target.endswith('.pt'):
        raise FileNotFoundError(f'Không tìm thấy checkpoint: {rel_or_abs}')
    return os.path.relpath(abs_target, ROOT_DIR)


def scan_checkpoints() -> Dict[str, Any]:
    checkpoints: List[Dict[str, Any]] = []
    if os.path.isdir(SAVED_MODEL_DIR):
        for dirpath, _, filenames in os.walk(SAVED_MODEL_DIR):
            for name in filenames:
                if not name.endswith('.pt'):
                    continue
                full_path = os.path.join(dirpath, name)
                rel_path = os.path.relpath(full_path, ROOT_DIR)
                rel_parts = os.path.relpath(full_path, SAVED_MODEL_DIR).split(os.sep)
                checkpoints.append(
                    {
                        'path': rel_path.replace('\\', '/'),
                        'filename': name,
                        'problem_tag': rel_parts[0] if len(rel_parts) >= 3 else '',
                        'run_name': rel_parts[1] if len(rel_parts) >= 3 else '',
                        'size_bytes': os.path.getsize(full_path),
                        'mtime': os.path.getmtime(full_path),
                    }
                )
    checkpoints.sort(key=lambda c: c['mtime'], reverse=True)
    return {
        'checkpoints': checkpoints,
        'pomo_tsp20_ready': os.path.isfile(POMO_TSP20_CKPT),
        'pomo_tsp20_path': os.path.relpath(POMO_TSP20_CKPT, ROOT_DIR),
    }


def _check_pomo_requirement(problem: str) -> None:
    if '_re_' in problem and not os.path.isfile(POMO_TSP20_CKPT):
        raise ValueError(
            'Chế độ `_re_tsp` yêu cầu checkpoint POMO TSP tại '
            '`src/route_solver/pomo_tsp_solver/result/saved_tsp20_model/checkpoint-510.pt` '
            '(chưa tìm thấy file này).'
        )


def build_eam_train_cmd(params: Dict[str, Any]) -> Dict[str, Any]:
    problem = str(params.get('problem', 'mo_cvrp'))
    _check_pomo_requirement(problem)

    customer_size = int(params.get('customer_size', 20))
    vehicle_num = int(
        params.get('vehicle_num', DEFAULT_VEHICLES.get(customer_size, 4))
    )
    batch_size = int(params.get('batch_size', 256))
    epoch_size = int(params.get('epoch_size', 2560))
    epoch_end = int(params.get('epoch_end', 3))
    mo_reward_type = str(params.get('mo_reward_type', 'tch'))
    lr_model = float(params.get('lr_model', 1e-4))
    n_aug = int(params.get('N_aug', 8))
    n_pref = int(params.get('N_pref', 2))

    if epoch_size < batch_size or epoch_size % batch_size != 0:
        raise ValueError(
            f'epoch_size ({epoch_size}) phải lớn hơn hoặc bằng và chia hết cho batch_size ({batch_size}).'
        )

    no_eval = bool(params.get('no_eval_in_train', False))
    val_dataset_opt = str(params.get('val_dataset', 'auto')).strip()
    val_range_start = int(params.get('val_range_start', 0))
    val_range_end = int(params.get('val_range_end', 200))
    val_batch_size = int(params.get('val_batch_size', 100))
    eval_type = str(params.get('eval_type', 'pareto'))
    sample_times = int(params.get('sample_times', 0))
    quick_eval = bool(params.get('quick_eval', True))
    zoom = str(params.get('zoom', 'auto'))

    val_count = val_range_end - val_range_start
    if val_count < val_batch_size or val_count % val_batch_size != 0:
        raise ValueError(
            f'Số lượng bài toán validation ({val_count}) phải >= và chia hết cho val_batch_size ({val_batch_size}).'
        )

    cmd: List[str] = [
        sys.executable,
        os.path.join('src', 'main.py'),
        '--problem',
        problem,
        '--customer_size',
        str(customer_size),
        '--vehicle_num',
        str(vehicle_num),
        '--batch_size',
        str(batch_size),
        '--epoch_size',
        str(epoch_size),
        '--epoch_end',
        str(epoch_end),
        '--mo_reward_type',
        mo_reward_type,
        '--lr_model',
        str(lr_model),
        '--N_aug',
        str(n_aug),
        '--N_pref',
        str(n_pref),
        '--val_range',
        str(val_range_start),
        str(val_range_end),
        '--val_batch_size',
        str(val_batch_size),
        '--eval_type',
        eval_type,
        '--sample_times',
        str(sample_times),
    ]

    if val_dataset_opt == 'random':
        cmd.extend(['--val_dataset', 'random'])
        if zoom == 'auto':
            zoom = 'off'
    elif val_dataset_opt and val_dataset_opt != 'auto':
        resolve_dataset_path(val_dataset_opt)
        cmd.extend(['--val_dataset', f'datasets/{val_dataset_opt}'])
    elif not no_eval:
        # Check default auto dataset path in options.py
        if problem == 'cvrp':
            expected_name = f'cvrp{customer_size}.npy'
        elif 'cvrp' in problem:
            expected_name = f'cvrp{customer_size}-v{vehicle_num}.npy'
        else:
            expected_name = f'tsp{customer_size}.npy'
        if not os.path.isfile(os.path.join(DATASETS_DIR, expected_name)):
            raise FileNotFoundError(
                f'Không tìm thấy dataset mặc định `datasets/{expected_name}` cho validation. '
                'Hãy chọn `val_dataset` khác, chọn `random`, hoặc bật `no_eval_in_train`.'
            )

    cmd.extend(['--zoom', zoom])
    if quick_eval:
        cmd.append('--quick_eval')
    if no_eval:
        cmd.append('--no_eval_in_train')
    if bool(params.get('no_cuda', False)):
        cmd.append('--no_cuda')
    if bool(params.get('no_log', False)):
        cmd.append('--no_log')
    if bool(params.get('no_save', False)):
        cmd.append('--no_save')
    if bool(params.get('no_progress_bar', False)):
        cmd.append('--no_progress_bar')

    resume_path = str(params.get('resume', '')).strip()
    if resume_path:
        rel_ckpt = _resolve_saved_checkpoint(resume_path)
        cmd.extend(['--resume', rel_ckpt])

    run_name = str(params.get('run_name', '')).strip()
    if run_name:
        cmd.extend(['--run_name', run_name])

    return {
        'command': cmd,
        'command_preview': _format_cmd_preview(cmd),
        'title': f'E-AM Train ({problem}, {customer_size} khách, {vehicle_num} xe, {epoch_end} epoch)',
    }


def _suggested_max_parallel(customer_size: int) -> int:
    if customer_size <= 20:
        return 16384
    if customer_size <= 50:
        return 4096
    if customer_size <= 100:
        return 2048
    if customer_size <= 200:
        return 512
    return 128


def build_eam_eval_cmd(params: Dict[str, Any]) -> Dict[str, Any]:
    problem = str(params.get('problem', 'mo_cvrp'))
    _check_pomo_requirement(problem)

    load_path = _resolve_saved_checkpoint(str(params.get('load_path', '')))
    customer_size = int(params.get('customer_size', 20))
    vehicle_num = int(
        params.get('vehicle_num', DEFAULT_VEHICLES.get(customer_size, 4))
    )
    val_customer_size = int(params.get('val_customer_size', customer_size))
    val_vehicle_num = int(params.get('val_vehicle_num', vehicle_num))

    val_dataset_opt = str(params.get('val_dataset', '')).strip()
    if not val_dataset_opt:
        raise ValueError('Vui lòng chọn dataset để đánh giá.')

    val_range_start = int(params.get('val_range_start', 0))
    val_range_end = int(params.get('val_range_end', 200))
    val_batch_size = int(params.get('val_batch_size', 100))
    eval_type = str(params.get('eval_type', 'pareto_sample'))
    sample_times = int(params.get('sample_times', 32))
    val_n_aug = int(params.get('val_N_aug', 8))
    max_parallel = int(
        params.get('max_parallel') or _suggested_max_parallel(val_customer_size)
    )
    zoom = str(params.get('zoom', 'auto'))

    val_count = val_range_end - val_range_start
    if val_count < val_batch_size or val_count % val_batch_size != 0:
        raise ValueError(
            f'Số lượng bài toán ({val_count}) phải >= và chia hết cho val_batch_size ({val_batch_size}).'
        )

    cmd: List[str] = [
        sys.executable,
        os.path.join('src', 'main.py'),
        '--eval_only',
        '--problem',
        problem,
        '--customer_size',
        str(customer_size),
        '--vehicle_num',
        str(vehicle_num),
        '--val_customer_size',
        str(val_customer_size),
        '--val_vehicle_num',
        str(val_vehicle_num),
        '--val_range',
        str(val_range_start),
        str(val_range_end),
        '--val_batch_size',
        str(val_batch_size),
        '--eval_type',
        eval_type,
        '--sample_times',
        str(sample_times),
        '--val_N_aug',
        str(val_n_aug),
        '--max_parallel',
        str(max_parallel),
        '--load_path',
        load_path,
    ]

    if val_dataset_opt == 'random':
        cmd.extend(['--val_dataset', 'random'])
        if zoom == 'auto':
            zoom = 'off'
    else:
        path = resolve_dataset_path(val_dataset_opt)
        meta = build_dataset_meta(val_dataset_opt, path)
        if meta and val_range_end > meta.num_instances:
            raise ValueError(
                f'val_range_end ({val_range_end}) vượt quá số bài toán của {val_dataset_opt} ({meta.num_instances}).'
            )
        cmd.extend(['--val_dataset', f'datasets/{val_dataset_opt}'])

    cmd.extend(['--zoom', zoom])
    if bool(params.get('quick_eval', False)):
        cmd.append('--quick_eval')
    if bool(params.get('no_cuda', False)):
        cmd.append('--no_cuda')
    if bool(params.get('no_log', True)):
        cmd.append('--no_log')
    if bool(params.get('no_progress_bar', False)):
        cmd.append('--no_progress_bar')

    run_name = str(params.get('run_name', '')).strip()
    if run_name:
        cmd.extend(['--run_name', run_name])

    return {
        'command': cmd,
        'command_preview': _format_cmd_preview(cmd),
        'title': f'E-AM Eval ({val_dataset_opt}, {eval_type})',
    }


def build_cmma_cmd(params: Dict[str, Any]) -> Dict[str, Any]:
    filename = str(params.get('load_file', '')).strip()
    path = resolve_dataset_path(filename)
    meta = build_dataset_meta(filename, path)
    if meta is None:
        raise ValueError(f'Dataset {filename} không hợp lệ.')

    problem_index = int(params.get('problem_index', 0))
    if problem_index < 0 or problem_index >= meta.num_instances:
        raise ValueError(
            f'problem_index ({problem_index}) phải nằm trong [0, {meta.num_instances - 1}].'
        )

    vehicle_num = int(
        params.get('vehicle_num')
        or meta.vehicle_num
        or DEFAULT_VEHICLES.get(meta.customer_size, 4)
    )
    population_size = int(params.get('population_size', 50))
    if population_size < 2 or population_size % 2 != 0:
        raise ValueError('population_size trong CMMA phải là số chẵn >= 2 (do ghép cặp lai ghép).')

    n_gen = int(params.get('N_gen', 200))
    n_iter = int(params.get('Iter', 40))
    ls_iter = int(params.get('LS_iter', 300))
    tournament_tau = float(params.get('tournament_tau', 0.25))
    save_per_gen = int(params.get('save_per_gen', 50))
    capacity = float(params.get('capacity', 1.0))
    seed = int(params.get('seed', 0))

    cmd: List[str] = [
        sys.executable,
        os.path.join('src-CMMA', 'main.py'),
        '--load_file',
        f'datasets/{filename}',
        '--problem_index',
        str(problem_index),
        '--vehicle_num',
        str(vehicle_num),
        '--capacity',
        str(capacity),
        '--population_size',
        str(population_size),
        '--N_gen',
        str(n_gen),
        '--Iter',
        str(n_iter),
        '--LS_iter',
        str(ls_iter),
        '--tournament_tau',
        str(tournament_tau),
        '--save_per_gen',
        str(save_per_gen),
        '--seed',
        str(seed),
    ]

    if bool(params.get('use_ref_point', True)):
        default_ref = DEFAULT_REF_POINTS.get(meta.customer_size, (30.0, 3.0))
        ref_0 = float(params.get('ref_point_0', default_ref[0]))
        ref_1 = float(params.get('ref_point_1', default_ref[1]))
        cmd.extend(['--ref_point', f'{ref_0:g}', f'{ref_1:g}'])

    if bool(params.get('no_log', False)):
        cmd.append('--no_log')

    run_name = str(params.get('run_name', '')).strip()
    if run_name:
        cmd.extend(['--run_name', run_name])

    return {
        'command': cmd,
        'command_preview': _format_cmd_preview(cmd),
        'title': f'CMMA ({filename} #{problem_index}, {n_gen} gen)',
    }


def _append_log(job: JobRecord, line: str) -> None:
    with job.lock:
        job.logs.append(line)
        if len(job.logs) > MAX_LOG_LINES:
            del job.logs[: len(job.logs) - MAX_LOG_LINES]


TQDM_PCT_PATTERN = re.compile(r'^(?:.*?:)?\s*(\d+)%\|')


def _iter_pipe_segments(pipe: Any) -> Iterator[Tuple[str, bool]]:
    """Yield (text, is_progress) incrementally from raw binary stdout/stderr pipe."""
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    fd = pipe.fileno() if hasattr(pipe, 'fileno') else None
    buf = ''
    pending_cr = False

    def _make_segment(text: str, force_progress: bool) -> Optional[Tuple[str, bool]]:
        if not text.strip():
            return None
        return text, (force_progress or bool(TQDM_PCT_PATTERN.match(text)))

    while True:
        raw = os.read(fd, 4096) if fd is not None else pipe.read(1)
        if not raw:
            buf += decoder.decode(b'', final=True)
            seg = _make_segment(buf, pending_cr)
            if seg:
                yield seg
            break

        chunk = decoder.decode(raw, final=False)
        for ch in chunk:
            if pending_cr:
                pending_cr = False
                if ch == '\n':
                    # Windows CRLF (\r\n) is a standard newline, not a tqdm overwrite.
                    seg = _make_segment(buf, False)
                    buf = ''
                    if seg:
                        yield seg
                    continue
                seg = _make_segment(buf, True)
                buf = ''
                if seg:
                    yield seg

            if ch == '\r':
                if not buf.strip():
                    buf = ''
                elif TQDM_PCT_PATTERN.match(buf):
                    seg = _make_segment(buf, True)
                    buf = ''
                    if seg:
                        yield seg
                else:
                    pending_cr = True
            elif ch == '\n':
                seg = _make_segment(buf, False)
                buf = ''
                if seg:
                    yield seg
            else:
                buf += ch
                if ch == ']' and TQDM_PCT_PATTERN.match(buf):
                    # tqdm omits trailing \r/\n while working on the next step; flush at closing ']'.
                    seg = _make_segment(buf, True)
                    buf = ''
                    if seg:
                        yield seg


def _stream_reader(job: JobRecord) -> None:
    proc = job.process
    assert proc is not None and proc.stdout is not None
    last_logged_stage = ''
    last_logged_pct = -100
    last_logged_time = 0.0
    try:
        for text, is_progress in _iter_pipe_segments(proc.stdout):
            clean = text.strip()
            if not clean:
                continue
            if is_progress:
                with job.lock:
                    job.progress = clean
                now = time.time()
                m = TQDM_PCT_PATTERN.match(clean)
                if m:
                    stage = clean.split(':', 1)[0].strip() if ':' in clean else 'progress'
                    pct = int(m.group(1))
                    if (
                        stage != last_logged_stage
                        or pct == 100
                        or pct - last_logged_pct >= 10
                        or (now - last_logged_time) >= 3.0
                    ):
                        last_logged_stage = stage
                        last_logged_pct = pct
                        last_logged_time = now
                        _append_log(job, f'[TIẾN ĐỘ] {clean}')
                elif (now - last_logged_time) >= 2.0:
                    last_logged_time = now
                    _append_log(job, clean)
            else:
                _append_log(job, clean)
        proc.stdout.close()
    except Exception as exc:
        _append_log(job, f'[WEB RUNNER] Lỗi đọc log: {exc}')
    finally:
        rc = proc.wait()
        with job.lock:
            job.return_code = rc
            job.finished_at = time.time()
            if job.status == 'stopped':
                pass
            elif rc == 0:
                job.status = 'completed'
            else:
                job.status = 'failed'
        _append_log(
            job,
            f'[WEB RUNNER] Kết thúc tiến trình (trạng thái={job.status}, exit_code={rc}).',
        )


def start_job(kind: str, built: Dict[str, Any]) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:10]
    cmd = built['command']
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['TQDM_MININTERVAL'] = '0.5'

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )

    job = JobRecord(
        job_id=job_id,
        kind=kind,
        title=built['title'],
        command=cmd,
        command_preview=built['command_preview'],
        status='running',
        pid=proc.pid,
        process=proc,
        logs=[f'[WEB RUNNER] Khởi chạy PID={proc.pid}: {built["command_preview"]}'],
    )

    with _JOBS_LOCK:
        _JOBS[job_id] = job

    t = threading.Thread(target=_stream_reader, args=(job,), daemon=True)
    t.start()
    return job.to_dict(include_logs=True)


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == 'nt':
        # Try killing child tree first with a short timeout, then ensure main process dies.
        try:
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pass
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def stop_job(job_id: str) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise KeyError(f'Không tìm thấy job: {job_id}')

    proc = job.process
    if proc is not None and proc.poll() is None:
        with job.lock:
            job.status = 'stopped'
        _append_log(job, '[WEB RUNNER] Đang dừng tiến trình theo yêu cầu người dùng...')
        _terminate_process_tree(proc)

    return job.to_dict(include_logs=False)


def list_jobs() -> List[Dict[str, Any]]:
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    jobs.sort(key=lambda j: j.started_at, reverse=True)
    return [j.to_dict(include_logs=False) for j in jobs]


def get_job(job_id: str, include_logs: bool = True) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise KeyError(f'Không tìm thấy job: {job_id}')
    return job.to_dict(include_logs=include_logs)


def get_job_slice(
    job_id: str, start_idx: int
) -> Tuple[List[str], str, Optional[int], str]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise KeyError(f'Không tìm thấy job: {job_id}')
    with job.lock:
        lines = job.logs[start_idx:]
        return lines, job.status, job.return_code, job.progress


def tensorboard_status() -> Dict[str, Any]:
    with _TB_LOCK:
        running = _TB_PROC is not None and _TB_PROC.poll() is None
        pid = _TB_PROC.pid if running and _TB_PROC is not None else None
        return {
            'running': running,
            'pid': pid,
            'port': _TB_PORT,
            'url': f'http://127.0.0.1:{_TB_PORT}',
        }


def start_tensorboard(port: int = 6006) -> Dict[str, Any]:
    global _TB_PROC, _TB_PORT
    with _TB_LOCK:
        if _TB_PROC is not None and _TB_PROC.poll() is None:
            return {
                'running': True,
                'pid': _TB_PROC.pid,
                'port': _TB_PORT,
                'url': f'http://127.0.0.1:{_TB_PORT}',
            }
        os.makedirs(os.path.join(ROOT_DIR, 'log'), exist_ok=True)
        _TB_PORT = port
        _TB_PROC = subprocess.Popen(
            [
                sys.executable,
                '-m',
                'tensorboard.main',
                '--logdir',
                'log',
                '--host',
                '127.0.0.1',
                '--port',
                str(port),
            ],
            cwd=ROOT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {
            'running': True,
            'pid': _TB_PROC.pid,
            'port': _TB_PORT,
            'url': f'http://127.0.0.1:{_TB_PORT}',
        }


def stop_tensorboard() -> Dict[str, Any]:
    global _TB_PROC
    with _TB_LOCK:
        if _TB_PROC is not None and _TB_PROC.poll() is None:
            _terminate_process_tree(_TB_PROC)
        _TB_PROC = None
    return tensorboard_status()


def shutdown_all_processes() -> None:
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    for job in jobs:
        if job.process is not None and job.process.poll() is None:
            with job.lock:
                job.status = 'stopped'
            _terminate_process_tree(job.process)
    stop_tensorboard()
