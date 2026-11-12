# Cách chạy

Giải nén file zip rồi vào thư mục vừa giải nén:

```bash
unzip asqp_llm_package.zip -d asqp_llm
cd asqp_llm
```

Kích hoạt env có sẵn torch (ví dụ `LLM`), trỏ tới thư mục chứa model, rồi chạy script setup:

```bash
conda activate LLM
export ASQP_CACHE_DIR=/duong/dan/toi/models
bash llm/setup.sh
```

Script sẽ tạo một venv riêng tên `.venv-asqp`, nên env `LLM` không bị đụng tới. Máy không có mạng cũng chạy được, vì thư viện đã có sẵn trong gói. Chạy xong nó in ra một loạt dòng kiểm tra; nếu tất cả đều là `PASS` thì đã cài xong.

Sau đó bật venv lên và chạy:

```bash
source .venv-asqp/bin/activate
nohup python llm/pipeline.py run > run_all.log 2>&1 &
```

Chạy hết mất khoảng 3 ngày. Muốn xem tiến độ thì `tail -f run_all.log`. Nếu bị ngắt giữa chừng, cứ chạy lại đúng lệnh trên, phần nào xong rồi sẽ tự bỏ qua.

Khi chạy xong sẽ có file `results_llm_*.zip` ở thư mục gốc, gửi lại file đó là được.
