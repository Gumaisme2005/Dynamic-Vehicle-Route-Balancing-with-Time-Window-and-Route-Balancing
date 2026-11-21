import asyncio
from contextlib import asynccontextmanager
import json
import os
import platform
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch

from web.dataset_service import (
    CAPACITIES,
    DATASETS_DIR,
    DEFAULT_REF_POINTS,
    DEFAULT_VEHICLES,
    LOG_DIR,
    RESULT_DIR,
    SAVED_MODEL_DIR,
    create_random_dataset,
    get_dataset_instance,
    get_dataset_summary,
    save_uploaded_dataset,
    scan_datasets,
)
from web.runner_service import (
    build_cmma_cmd,
    build_eam_eval_cmd,
    build_eam_train_cmd,
    get_job,
    get_job_slice,
    list_jobs,
    scan_checkpoints,
    shutdown_all_processes,
    start_job,
    start_tensorboard,
    stop_job,
    stop_tensorboard,
    tensorboard_status,
)
from web.result_service import (
    compare_results,
    get_instance_pareto,
    get_result_summary,
    get_solution_routes,
    scan_results,
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    shutdown_all_processes()


app = FastAPI(title='E-AM for VRPRB Dashboard', version='0.3.0', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<rect width="64" height="64" rx="14" fill="#0f172a"/>'
    '<circle cx="20" cy="20" r="6" fill="#ef4444"/>'
    '<circle cx="46" cy="24" r="5" fill="#38bdf8"/>'
    '<circle cx="38" cy="46" r="5" fill="#22c55e"/>'
    '<path d="M20 20 L46 24 L38 46 Z" fill="none" stroke="#94a3b8" stroke-width="3"/>'
    '</svg>'
)


class GenerateDatasetRequest(BaseModel):
    num_instances: int = Field(default=20, ge=1, le=2000)
    customer_size: int = Field(default=20, ge=10, le=1000)
    vehicle_num: int = Field(default=4, ge=2, le=100)
    vehicle_capacity: int = Field(default=-1)
    seed: int = Field(default=1234, ge=0)
    filename: str = Field(default='custom_cvrp20-v4.npy')


def _count_files(directory: str, suffix: Optional[str] = None) -> int:
    if not os.path.isdir(directory):
        return 0
    total = 0
    for _, _, files in os.walk(directory):
        for name in files:
            if suffix is None or name.endswith(suffix):
                total += 1
    return total


def _count_subdirs(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    return sum(1 for entry in os.scandir(directory) if entry.is_dir())


@app.get('/favicon.ico', include_in_schema=False)
def favicon() -> Response:
    return Response(content=FAVICON_SVG, media_type='image/svg+xml')


@app.get('/')
def serve_index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


@app.get('/api/system/status')
def system_status() -> dict:
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else 'CPU'
    gpu_count = torch.cuda.device_count() if cuda_available else 0

    return {
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'os': f'{platform.system()} {platform.release()}',
        'torch_version': torch.__version__,
        'cuda_available': cuda_available,
        'gpu_name': gpu_name,
        'gpu_count': gpu_count,
        'directories': {
            'datasets': {
                'exists': os.path.isdir(DATASETS_DIR),
                'count': len(scan_datasets()),
            },
            'saved_model': {
                'exists': os.path.isdir(SAVED_MODEL_DIR),
                'checkpoints': _count_files(SAVED_MODEL_DIR, '.pt'),
            },
            'result': {
                'exists': os.path.isdir(RESULT_DIR),
                'runs': _count_subdirs(RESULT_DIR),
            },
            'log': {
                'exists': os.path.isdir(LOG_DIR),
                'runs': _count_subdirs(LOG_DIR),
            },
        },
        'capacities': {str(k): v for k, v in CAPACITIES.items()},
        'default_vehicles': {str(k): v for k, v in DEFAULT_VEHICLES.items()},
        'default_ref_points': {str(k): list(v) for k, v in DEFAULT_REF_POINTS.items()},
    }


@app.get('/api/datasets')
def list_datasets() -> dict:
    return {'datasets': scan_datasets()}


@app.get('/api/datasets/{filename}/summary')
def dataset_summary(filename: str) -> dict:
    try:
        return get_dataset_summary(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/api/datasets/{filename}/instance/{problem_index}')
def dataset_instance(
    filename: str,
    problem_index: int,
    zoom: Optional[bool] = Query(default=None),
) -> dict:
    try:
        return get_dataset_instance(filename, problem_index, apply_zoom=zoom)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/datasets/generate')
def generate_dataset(req: GenerateDatasetRequest) -> dict:
    try:
        meta = create_random_dataset(
            num_instances=req.num_instances,
            customer_size=req.customer_size,
            vehicle_num=req.vehicle_num,
            vehicle_capacity=req.vehicle_capacity,
            seed=req.seed,
            filename=req.filename,
        )
        return {'status': 'ok', 'dataset': meta}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/datasets/upload')
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    try:
        content = await file.read()
        meta = save_uploaded_dataset(content, file.filename or '')
        return {'status': 'ok', 'dataset': meta}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------- Step 3: Runner & Checkpoints & TensorBoard ----------


@app.get('/api/checkpoints')
def list_checkpoints() -> dict:
    return scan_checkpoints()


@app.post('/api/jobs/eam/train/preview')
def preview_eam_train(params: Dict[str, Any]) -> dict:
    try:
        built = build_eam_train_cmd(params)
        return {'valid': True, 'command_preview': built['command_preview'], 'title': built['title']}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/jobs/eam/train')
def run_eam_train(params: Dict[str, Any]) -> dict:
    try:
        built = build_eam_train_cmd(params)
        return {'job': start_job('eam_train', built)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/jobs/eam/eval/preview')
def preview_eam_eval(params: Dict[str, Any]) -> dict:
    try:
        built = build_eam_eval_cmd(params)
        return {'valid': True, 'command_preview': built['command_preview'], 'title': built['title']}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/jobs/eam/eval')
def run_eam_eval(params: Dict[str, Any]) -> dict:
    try:
        built = build_eam_eval_cmd(params)
        return {'job': start_job('eam_eval', built)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/jobs/cmma/preview')
def preview_cmma(params: Dict[str, Any]) -> dict:
    try:
        built = build_cmma_cmd(params)
        return {'valid': True, 'command_preview': built['command_preview'], 'title': built['title']}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/jobs/cmma/run')
def run_cmma(params: Dict[str, Any]) -> dict:
    try:
        built = build_cmma_cmd(params)
        return {'job': start_job('cmma', built)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/api/jobs')
def jobs_list() -> dict:
    return {'jobs': list_jobs()}


@app.get('/api/jobs/{job_id}')
def job_detail(job_id: str) -> dict:
    try:
        return {'job': get_job(job_id, include_logs=True)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post('/api/jobs/{job_id}/stop')
def job_stop(job_id: str) -> dict:
    try:
        return {'job': stop_job(job_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get('/api/jobs/{job_id}/stream')
async def job_stream(job_id: str, offset: int = Query(default=0, ge=0)):
    try:
        get_job(job_id, include_logs=False)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def event_gen():
        cursor = offset
        last_progress = ''
        while True:
            lines, status, return_code, progress = get_job_slice(job_id, cursor)
            if lines or progress != last_progress:
                cursor += len(lines)
                last_progress = progress
                payload = json.dumps(
                    {
                        'lines': lines,
                        'next_offset': cursor,
                        'status': status,
                        'return_code': return_code,
                        'progress': progress,
                    },
                    ensure_ascii=False,
                )
                yield f'data: {payload}\n\n'
            if status != 'running':
                done_payload = json.dumps(
                    {
                        'done': True,
                        'status': status,
                        'return_code': return_code,
                        'next_offset': cursor,
                        'progress': progress,
                    },
                    ensure_ascii=False,
                )
                yield f'event: done\ndata: {done_payload}\n\n'
                break
            await asyncio.sleep(0.35)

    return StreamingResponse(event_gen(), media_type='text/event-stream')


@app.get('/api/tensorboard/status')
def tb_status() -> dict:
    return tensorboard_status()


@app.post('/api/tensorboard/start')
def tb_start(port: int = Query(default=6006, ge=1024, le=65535)) -> dict:
    return start_tensorboard(port=port)


@app.post('/api/tensorboard/stop')
def tb_stop() -> dict:
    return stop_tensorboard()


# ---------- Step 4: Results, Pareto Compare & Route Visualizer ----------


@app.get('/api/results')
def results_list() -> dict:
    return scan_results()


@app.get('/api/results/summary')
def result_summary(
    result_id: str = Query(...),
    gen: Optional[int] = Query(default=None),
) -> dict:
    try:
        return get_result_summary(result_id, gen=gen)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/api/results/pareto')
def result_pareto(
    result_id: str = Query(...),
    instance_offset: int = Query(default=0, ge=0),
    gen: Optional[int] = Query(default=None),
) -> dict:
    try:
        return get_instance_pareto(result_id, instance_offset=instance_offset, gen=gen)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/api/results/routes')
def result_routes(
    result_id: str = Query(...),
    instance_offset: int = Query(default=0, ge=0),
    sol_idx: int = Query(default=0, ge=0),
    gen: Optional[int] = Query(default=None),
) -> dict:
    try:
        return get_solution_routes(
            result_id, instance_offset=instance_offset, sol_idx=sol_idx, gen=gen
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/api/results/compare')
def result_compare(
    result_a: str = Query(...),
    inst_a: int = Query(default=0, ge=0),
    gen_a: Optional[int] = Query(default=None),
    result_b: Optional[str] = Query(default=None),
    inst_b: int = Query(default=0, ge=0),
    gen_b: Optional[int] = Query(default=None),
) -> dict:
    try:
        return compare_results(
            result_a_id=result_a,
            inst_a=inst_a,
            gen_a=gen_a,
            result_b_id=result_b or None,
            inst_b=inst_b,
            gen_b=gen_b,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == '__main__':
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description='E-AM for VRPRB Web Dashboard')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    cli_args = parser.parse_args()

    uvicorn.run('web.server:app', host=cli_args.host, port=cli_args.port, reload=False)
