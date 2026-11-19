import importlib.util
import os
from typing import Any


def _load_shared_mo_utils() -> Any:
    mo_utils_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'src',
        'utils',
        'mo_utils.py',
    )
    spec = importlib.util.spec_from_file_location('_shared_mo_utils', mo_utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load shared mo_utils from {mo_utils_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mo_utils = _load_shared_mo_utils()

ndsort_ess = _mo_utils.ndsort_ess
crowdis = _mo_utils.crowdis
hv_wfg = _mo_utils.hv_wfg
hv_indicator = _mo_utils.hv_indicator
