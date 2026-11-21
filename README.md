# E-AM-for-VRPRB

Mã nguồn triển khai bài báo **Deep Reinforcement Learning for the Vehicle Routing Problem with Route Balancing** (Bài toán Định tuyến Xe có Giới hạn Tải trọng và Cân bằng Lộ trình — VRPRB).

Hệ thống tối ưu đồng thời **2 mục tiêu**:
1. **Min-Sum ($f_1$)**: Tổng chiều dài quãng đường của toàn bộ các xe.
2. **Min-Max ($f_2$)**: Chiều dài quãng đường của xe đi dài nhất (cân bằng khối lượng công việc giữa các xe).

### Cấu trúc thư mục chính
* `src/`: Mô hình Học tăng cường sâu **E-AM** (Extended Attention Model với Hypernetwork, Vehicle Context Encoder, 2D Cross-Probability Decoder và tùy chọn tinh chỉnh tuyến con bằng POMO TSP).
* `src-CMMA/`: Thuật toán tiến hóa Memetic đa mục tiêu (**CMMA**) chạy trên CPU làm baseline đối chiếu.
* `web/`: Giao diện Web Dashboard (`FastAPI` + `HTML/CSS/JS`) điều khiển chạy/dừng thuật toán và trực quan hóa dữ liệu, Pareto Front, lộ trình xe.
* `datasets/`: Thư mục chứa dữ liệu đánh giá `.npy` (tải từ Releases).
* `saved_model/`: Thư mục lưu checkpoint mô hình (`.pt`) và kết quả đánh giá (`eval.txt`, `objv.json`, `action.json`, `pareto_front*.png`).
* `log/`: Thư mục lưu log cho TensorBoard.
* `result/`: Thư mục lưu kết quả đầu ra của thuật toán `src-CMMA`.

---

## Bước 1: Tạo môi trường & Cài đặt thư viện

Mã nguồn đã được nâng cấp để chạy tương thích trên **Python 3.10 – 3.15+** và **PyTorch $\ge$ 2.0**, đồng thời **không sử dụng 2 thư viện cũ `hvwfg` và `geatpy`** (đã được thay bằng cài đặt thuần NumPy tương đương trong [src/utils/mo_utils.py](src/utils/mo_utils.py)).

### 1.1. Danh sách thư viện cần thiết

| Thư viện | Mục đích | Bắt buộc cho |
| :--- | :--- | :--- |
| **`torch`** ($\ge 2.0$) | Xây dựng mạng Neural, huấn luyện RL, DDP | `src/` |
| **`numpy`** | Xử lý dữ liệu `.npy`, tính Pareto & Hypervolume | `src/` và `src-CMMA/` |
| **`tqdm`** | Thanh tiến trình khi train và eval | `src/` |
| **`matplotlib`** | Vẽ và lưu đồ thị mặt Pareto (`.png`) | `src/` và `src-CMMA/analyse_result.py` |
| **`tensorboard`** | Ghi log và trực quan hóa quá trình train (`SummaryWriter`) | `src/` *(tùy chọn nếu truyền `--no_log`)* |
| **`fastapi`**, **`uvicorn`** | Chạy giao diện Web Dashboard (`web/server.py`) | `web/` *(chỉ cần nếu dùng Web UI)* |

### 1.2. Khởi tạo môi trường ảo `.venv`
Mở PowerShell tại thư mục gốc của dự án (`E-AM-for-VRPRB`):

* **Cách 1: Dùng `uv` (khuyên dùng, cài nhanh)**
  ```powershell
  uv venv .venv --python 3.15
  ```
* **Cách 2: Dùng `python` có sẵn trên máy**
  ```powershell
  python -m venv .venv
  ```

*(Tùy chọn) Kích hoạt môi trường ảo trong PowerShell:*
```powershell
.\.venv\Scripts\Activate.ps1
```

### 1.3. Cài đặt các thư viện

1. **Kiểm tra GPU NVIDIA (nếu có)**:
   ```powershell
   nvidia-smi
   ```
   Xem mục `CUDA Version` ở góc trên bên phải để chọn đúng bản PyTorch CUDA (ví dụ với driver hỗ trợ CUDA 13.x trên Python 3.15 dùng index `cu132`; với CUDA 12.x dùng `cu124` / `cu128` tùy phiên bản Python).

2. **Cài PyTorch**:
   * **Nếu máy có GPU NVIDIA** (ví dụ CUDA `cu132`):
     ```powershell
     uv pip install --python .\.venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu132 torch
     ```
     *(Hoặc dùng `pip`: `.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu132`)*
   * **Nếu chỉ chạy CPU**:
     ```powershell
     uv pip install --python .\.venv\Scripts\python.exe torch
     ```

3. **Cài các thư viện còn lại**:
   ```powershell
   uv pip install --python .\.venv\Scripts\python.exe numpy tqdm matplotlib tensorboard fastapi uvicorn
   ```
   *(Hoặc dùng `pip`: `.\.venv\Scripts\python.exe -m pip install numpy tqdm matplotlib tensorboard fastapi uvicorn`)*

4. **Kiểm tra lại cài đặt**:
   ```powershell
   .\.venv\Scripts\python.exe -c "import torch, numpy, tqdm, matplotlib, tensorboard; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

---

## Bước 2: Tải & Chuẩn bị dữ liệu (`datasets/`)

### 2.1. Link tải dữ liệu và Pre-trained Model
Toàn bộ bộ dữ liệu đánh giá (`datasets.zip`) và các mô hình đã huấn luyện sẵn (pre-trained models) từ tác giả được lưu tại trang GitHub Releases của repo gốc:

👉 **Link tải chính thức:** https://github.com/dtkon/E-AM-for-VRPRB/releases

### 2.2. Giải nén dữ liệu vào đúng vị trí
Tạo thư mục `datasets` ở gốc dự án và giải nén toàn bộ các file `.npy` vào bên trong (`.\datasets\*.npy`):

```powershell
Expand-Archive -Path "<duong_dan_toi_file_datasets.zip>" -DestinationPath ".\datasets" -Force
```

Sau khi giải nén, thư mục `.\datasets\` sẽ gồm **19 file**:
* **12 bộ dữ liệu tổng hợp** (mỗi file gồm 200 bài toán, shape `(200, N+1, 3)`):
  * `cvrp20-v4.npy`, `cvrp20-v5.npy` (20 khách, 4 / 5 xe)
  * `cvrp50-v7.npy`, `cvrp50-v9.npy` (50 khách, 7 / 9 xe)
  * `cvrp100-v11.npy`, `cvrp100-v14.npy` (100 khách, 11 / 14 xe)
  * `cvrp200-v15.npy`, `cvrp200-v19.npy` (200 khách, 15 / 19 xe)
  * `cvrp500-v25.npy`, `cvrp500-v30.npy` (500 khách, 25 / 30 xe)
  * `cvrp1000-v34.npy`, `cvrp1000-v40.npy` (1000 khách, 34 / 40 xe)
* **7 bộ benchmark chuẩn VRPLIB** (mỗi file gồm 1 bài toán, shape `(1, N+1, 3)`):
  * `p1 E51-05e.npy` (`customer_size=50`, `vehicle_num=5`)
  * `p2 E76-10e.npy` (`customer_size=75`, `vehicle_num=10`)
  * `p3 E101-08e.npy` (`customer_size=100`, `vehicle_num=8`)
  * `p4 E151-12c.npy` (`customer_size=150`, `vehicle_num=12`)
  * `p5 E200-17c.npy` (`customer_size=199`, `vehicle_num=17` — có 200 node gồm cả depot)
  * `p6 E121-07c.npy` (`customer_size=120`, `vehicle_num=7`)
  * `p7 E101-10c.npy` (`customer_size=100`, `vehicle_num=10`)

> **Lưu ý nếu dùng chế độ `mo_cvrp_re_tsp` (bài toán quy mô lớn):** Chế độ này cần thêm checkpoint POMO TSP từ Releases đặt tại `src/route_solver/pomo_tsp_solver/result/saved_tsp20_model/checkpoint-510.pt`.

---

## Bước 3 (Khuyên dùng): Chạy qua Giao diện Web Dashboard

Thay vì gõ câu lệnh dòng lệnh dài dòng, bạn có thể mở giao diện Web Dashboard tại `http://127.0.0.1:8000` bằng một trong các cách sau:

* **Cách 1 (1-click trên Windows)**: Double-click vào file `run_web.bat` (hoặc chạy `.\run_web.ps1` trong PowerShell) — tự động khởi chạy server và mở trình duyệt.
* **Cách 2 (Chạy thủ công)**:
  ```powershell
  .\.venv\Scripts\python.exe -m web.server --port 8000
  ```

### Các tab chức năng trên Web Dashboard
1. **`1. Hệ thống & Dữ liệu`**: Kiểm tra Python/PyTorch/GPU, quét toàn bộ `.npy` trong `datasets/`, xem phân phối demand, tải lên/sinh mới dataset, và vẽ trực quan vị trí Depot + khách hàng.
2. **`2. Chạy E-AM`**: Chạy **Huấn luyện** (có sẵn Preset *Chạy thử nhanh 3 epoch*, *20/50/100 khách*) hoặc **Đánh giá (`--eval_only`)** với checkpoint `.pt`, xem trước lệnh CLI, thanh tiến độ `%` và Terminal Log realtime, nút **Dừng ngay (Stop)** và nút bật/tắt **TensorBoard**. *(Mẹo: để `Max parallel` cao — ví dụ `16384` cho 20 khách, `4096` cho 50 khách — sẽ giúp Eval trên GPU nhanh gấp 5–6 lần so với `256`.)*
3. **`3. Chạy CMMA`**: Chọn file `.npy` và `problem_index`, chạy baseline CMMA với Preset `200 / 1000 / 10000` thế hệ, theo dõi log và dừng bất cứ lúc nào.
4. **`4. Kết quả & Lộ trình`**: Vẽ mặt Pareto tương tác (click thẳng vào điểm nghiệm bất kỳ hoặc chọn nhanh `Min f₁`, `Min f₂`, `Knee`), so sánh 2 mặt Pareto (E-AM vs CMMA) trên cùng biểu đồ, và vẽ bản đồ lộ trình chi tiết của từng xe.

---

## Bước 4: Chạy thử nhanh bằng dòng lệnh (Quick Start CLI)

Để kiểm tra code hoạt động trơn tru trên máy trước khi chạy huấn luyện dài, hãy chạy thử với bộ dữ liệu nhỏ **`datasets/cvrp20-v4.npy`**.

### 4.1. Chạy thử huấn luyện E-AM (3 epoch, vài phút)
Lệnh dưới đây huấn luyện nhanh 3 epoch (mỗi epoch 2560 bài toán ngẫu nhiên) và tự đánh giá trên toàn bộ 200 bài toán của `datasets/cvrp20-v4.npy`:

```powershell
.\.venv\Scripts\python.exe .\src\main.py `
  --problem mo_cvrp `
  --customer_size 20 `
  --vehicle_num 4 `
  --batch_size 256 `
  --epoch_size 2560 `
  --epoch_end 3 `
  --val_range 0 200 `
  --val_batch_size 100 `
  --eval_type pareto `
  --sample_times 0 `
  --quick_eval `
  --run_name test_20_v4
```

*Sau khi chạy xong, checkpoint được lưu tại `saved_model\mo_cvrp_20-v4\test_20_v4_<thoi_gian>\epoch-2.pt`.*

### 4.2. Chạy thử đánh giá E-AM (`--eval_only`)
Dùng file `epoch-2.pt` vừa tạo (hoặc pre-trained model tải từ [Releases](https://github.com/dtkon/E-AM-for-VRPRB/releases)) để đánh giá chi tiết với 101 vector sở thích (`pareto_sample`):

```powershell
.\.venv\Scripts\python.exe .\src\main.py `
  --eval_only --no_log `
  --problem mo_cvrp `
  --customer_size 20 `
  --vehicle_num 4 `
  --val_range 0 200 `
  --val_batch_size 100 `
  --val_dataset "datasets/cvrp20-v4.npy" `
  --eval_type pareto_sample `
  --sample_times 32 `
  --run_name eval_20_v4 `
  --load_path "saved_model\mo_cvrp_20-v4\<thu_muc_run>\epoch-2.pt"
```

### 4.3. Chạy thử thuật toán baseline CMMA (`src-CMMA/main.py`)
Chạy thử CMMA trên bài toán đầu tiên (`--problem_index 0`) của `datasets/cvrp20-v4.npy` trong 200 thế hệ:

```powershell
.\.venv\Scripts\python.exe .\src-CMMA\main.py `
  --load_file "datasets/cvrp20-v4.npy" `
  --problem_index 0 `
  --population_size 50 `
  --N_gen 200 `
  --save_per_gen 50 `
  --ref_point 30 3 `
  --run_name test_cmma
```

---

## Chạy huấn luyện & Đánh giá đầy đủ (CLI)

### 5.1. Huấn luyện đầy đủ E-AM (`200` epochs, `128000` bài toán/epoch)

* **20 khách, 4 xe**:
  ```powershell
  .\.venv\Scripts\python.exe .\src\main.py --problem mo_cvrp --customer_size 20 --vehicle_num 4 --batch_size 512
  ```
* **50 khách, 7 xe**:
  ```powershell
  .\.venv\Scripts\python.exe .\src\main.py --problem mo_cvrp --customer_size 50 --vehicle_num 7 --batch_size 256
  ```
* **100 khách, 11 xe** *(trên 1 GPU 8GB VRAM như RTX 4060 Laptop nên để `--batch_size 64` hoặc `32`)*:
  ```powershell
  .\.venv\Scripts\python.exe .\src\main.py --problem mo_cvrp --customer_size 100 --vehicle_num 11 --batch_size 64
  ```
* **Quy mô lớn với `mo_cvrp_re_tsp`** *(cần sẵn checkpoint POMO TSP)*:
  ```powershell
  .\.venv\Scripts\python.exe .\src\main.py --problem mo_cvrp_re_tsp --customer_size 100 --batch_size 90 --epoch_size 128070
  ```
* **Chạy tiếp từ checkpoint (`--resume`)**:
  ```powershell
  .\.venv\Scripts\python.exe .\src\main.py --problem mo_cvrp --customer_size 20 --vehicle_num 4 --batch_size 256 --resume "saved_model\mo_cvrp_20-v4\<thu_muc_run>\epoch-50.pt"
  ```

*(Lưu ý quy tắc chia hết trong `src/options.py`: `epoch_size % batch_size == 0` và `(val_range[1] - val_range[0]) % val_batch_size == 0`. Khi dùng `--val_dataset random`, nhớ truyền thêm `--zoom off`.)*

### 5.2. Đánh giá nâng cao (Cross-size & Benchmark VRPLIB)

* **Đánh giá trên 200 bài toán 100 khách, 11 xe**:
  ```powershell
  .\.venv\Scripts\python.exe .\src\main.py --eval_only --no_log --problem mo_cvrp --customer_size 100 --vehicle_num 11 --val_range 0 200 --val_batch_size 100 --val_dataset "datasets/cvrp100-v11.npy" --eval_type pareto_sample --sample_times 32 --load_path "<duong_dan_model.pt>"
  ```
* **Đánh giá trên bài toán lớn 1000 khách, 40 xe** *(dùng `mo_cvrp_re_tsp`)*:
  ```powershell
  .\.venv\Scripts\python.exe .\src\main.py --eval_only --no_log --problem mo_cvrp_re_tsp --customer_size 1000 --vehicle_num 40 --val_range 0 20 --val_batch_size 4 --val_dataset "datasets/cvrp1000-v40.npy" --eval_type pareto --sample_times 0 --max_parallel 2 --load_path "<duong_dan_model.pt>"
  ```
* **Đánh giá trên 7 file benchmark chuẩn `p1..p7`** *(mỗi file chỉ có 1 bài toán nên đặt `--val_range 0 1 --val_batch_size 1`; `--zoom auto` mặc định tự chuẩn hóa tọa độ về $[0, 1]^2$)*:
  ```powershell
  # Ví dụ với p1 E51-05e.npy (50 khách, 5 xe)
  .\.venv\Scripts\python.exe .\src\main.py --eval_only --no_log --problem mo_cvrp --customer_size 50 --vehicle_num 5 --val_range 0 1 --val_batch_size 1 --val_dataset "datasets/p1 E51-05e.npy" --eval_type pareto_sample --sample_times 32 --load_path "<duong_dan_model_50.pt>"
  ```

### 5.3. Chạy CMMA đầy đủ cho `p1..p7` hoặc hàng loạt
* Với `p1..p7` (tên file không chứa `-v<số>.` nên bắt buộc truyền `--vehicle_num`):
  ```powershell
  .\.venv\Scripts\python.exe .\src-CMMA\main.py --load_file "datasets/p1 E51-05e.npy" --problem_index 0 --vehicle_num 5 --N_gen 1000 --save_per_gen 100
  ```
* Chạy song song toàn bộ tập dữ liệu theo cấu hình trong [src-CMMA/parallel_start.py](src-CMMA/parallel_start.py):
  ```powershell
  .\.venv\Scripts\python.exe .\src-CMMA\parallel_start.py
  ```

---

## Cách xem và đọc hiểu kết quả (CLI & TensorBoard)

### 6.1. Theo dõi trực quan bằng TensorBoard (cho `src/`)
Trong khi train hoặc sau khi chạy xong (khi không dùng `--no_log`), mở một terminal mới tại thư mục gốc và chạy:

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir log
```

Mở trình duyệt vào `http://localhost:6006`, các mục cần chú ý:
* **SCALARS**:
  * `training/loss`: Loss của thuật toán REINFORCE theo từng bước cập nhật (`--log_step`).
  * `evaluating/HV`: Chỉ số **Hypervolume chuẩn hóa** trên tập validation qua từng epoch (giá trị trong $[0, 1]$, càng cao càng tốt).
  * `evaluating/infeasible_rate_numerator` & `evaluating/infeasible_rate_denominator`: Tỉ lệ nghiệm vi phạm ràng buộc (mục tiêu tiến về `0`).
* **TEXT**:
  * `training/step`: Giá trị reward của từng mục tiêu ứng với các vector sở thích `pref` khi train.
  * `evaluating/avg_obj`: Giá trị trung bình 2 mục tiêu `(objective_sum, objective_max)` tại 5 mức preference (`0.00/1.00`, `0.25/0.75`, `0.50/0.50`, `0.75/0.25`, `1.00/0.00`).
* **IMAGES**:
  * `evaluating/pareto_front`: Biểu đồ mặt Pareto trung bình của toàn tập validation.
  * `evaluating/pareto_front_<idx>`: Biểu đồ mặt Pareto của từng bài toán đầu tiên (số lượng do `--pareto_log_num` quy định).

### 6.2. Xem các file kết quả của E-AM trên ổ đĩa
Kết quả mỗi lần chạy được lưu tại `saved_model\<problem>_<N>-v<V>\<run_name>\`:
* `option.json`: Toàn bộ siêu tham số đã dùng.
* `epoch-<k>.pt`: Checkpoint tại epoch `k`.
* `eval.txt`: Tóm tắt kết quả `hv`, `avg_best`, `avg_avg`, `sample_hv` (nếu dùng `pareto_sample`) và `avg_obj`.
* `objv.json`: Mảng `[val_size, num_solutions, 2]` chứa giá trị `[tổng quãng đường, quãng đường max]` của từng nghiệm.
* `action.json`: Danh sách lộ trình chi tiết của các xe cho từng nghiệm (đã bỏ các số `0` lặp liên tiếp).
* `pareto_front.png`, `pareto_front_0.png`, ...: Ảnh mặt Pareto được vẽ sẵn.

Bạn có thể kiểm tra nhanh số liệu từ `objv.json` và `action.json` bằng lệnh Python:
```powershell
.\.venv\Scripts\python.exe -c "import json; obj = json.load(open(r'saved_model\mo_cvrp_20-v4\<run_dir>\objv.json')); act = json.load(open(r'saved_model\mo_cvrp_20-v4\<run_dir>\action.json')); print('Số bài toán:', len(obj), '| Số nghiệm/bài:', len(obj[0])); print('Nghiệm 0 bài 0 (sum, max):', obj[0][0]); print('Lộ trình các xe:', act[0][0])"
```

### 6.3. Xem kết quả của CMMA (`src-CMMA/`)
Kết quả được lưu tại `result\<ten_dataset>\p<index>_<run_name>\`:
* `log.txt`: Log tiến độ và `HV` qua từng mốc thế hệ.
* `value-<gen>.json`: Chứa `obj` (các cặp `[tổng quãng đường, quãng đường max]`), `ref`, và `hv` tại thế hệ `<gen>`.
* `population-<gen>.json`: Danh sách lộ trình 1D của quần thể tại thế hệ `<gen>`.
* Vẽ nhanh đồ thị mặt Pareto từ `value-<gen>.json`:
  ```powershell
  .\.venv\Scripts\python.exe -c "import json, sys; sys.path.insert(0, 'src-CMMA'); from analyse_result import plt_pareto; d = json.load(open(r'result\cvrp20-v4\<run_dir>\value-200.json')); print('HV:', d['hv']); plt_pareto(d['obj'])"
  ```

---

## 7. Tra cứu toàn bộ tham số dòng lệnh

```powershell
.\.venv\Scripts\python.exe .\src\main.py -h
.\.venv\Scripts\python.exe .\src-CMMA\main.py -h
```

---

## Lời cảm ơn & Tài liệu tham khảo (Acknowledgements)

Xin chân thành cảm ơn các tác giả **Jianhua Xiao, Detian Kong, Zhiguang Cao và Jingyi Zhao** đã nghiên cứu, công bố bài báo khoa học và chia sẻ mã nguồn mở gốc của mô hình **Equity Attention Model (E-AM)** cùng bản cài đặt đối chiếu **CMMA** cho bài toán **VRPRB**. Toàn bộ phần thuật toán lõi trong `src/` và `src-CMMA/` được kế thừa trực tiếp từ kho mã nguồn chính thức của nhóm tác giả; repo này chỉ bổ sung nâng cấp tương thích môi trường mới (Python 3.10–3.15+, PyTorch $\ge 2.0$, thay thế `geatpy`/`hvwfg` bằng NumPy) và giao diện Web Dashboard (`web/`) để phục vụ học tập và nghiên cứu.

* **Kho mã nguồn gốc trên GitHub:** https://github.com/dtkon/E-AM-for-VRPRB
* **Bài báo gốc:** Jianhua Xiao, Detian Kong, Zhiguang Cao, Jingyi Zhao (2026). *"Deep reinforcement learning for the vehicle routing problem with route balancing"*, **Transportation Research Part E**, 208, 104632.
  * **DOI:** https://doi.org/10.1016/j.tre.2025.104632
* **Thuật toán đối chiếu CMMA (`src-CMMA/`):** Cài đặt lại bằng Python từ nghiên cứu của Zhang, Z., Sun, Y., Xie, H., Teng, Y., & Wang, J. (2019). *"GMMA: GPU-based multiobjective memetic algorithms for vehicle routing problem with route balancing"*, **Applied Intelligence**, 49, 63–78.

### Trích dẫn (BibTeX)
```bibtex
@article{xiao2026eam_vrprb,
  title   = {Deep reinforcement learning for the vehicle routing problem with route balancing},
  author  = {Xiao, Jianhua and Kong, Detian and Cao, Zhiguang and Zhao, Jingyi},
  journal = {Transportation Research Part E},
  volume  = {208},
  pages   = {104632},
  year    = {2026},
  doi     = {10.1016/j.tre.2025.104632},
  url     = {https://github.com/dtkon/E-AM-for-VRPRB}
}
```
