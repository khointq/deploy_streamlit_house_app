# 🏙️ House Listing Streamlit App (Deploy Bundle)

Thư mục này đã được đóng gói để **đẩy lên GitHub** và **deploy bằng Streamlit**.

## Cấu trúc

- `app.py`: entrypoint cho Streamlit Cloud.
- `house_web_app.py`: logic chính của ứng dụng.
- `requirements.txt`: dependencies tối thiểu để chạy app.
- `runtime.txt`: version Python (`python-3.11`).
- `.streamlit/config.toml`: cấu hình Streamlit.
- `data/processed/`: dữ liệu cần cho app.
- `models/kmeans_by_district/`: model KMeans theo quận để dự đoán phân khúc.

## Chạy local

```powershell
streamlit run app.py
```

## Deploy Streamlit Community Cloud

1. Push thư mục này lên một GitHub repo.
2. Trên Streamlit Cloud, chọn repo và đặt:
   - **Main file path**: `app.py`
3. Deploy.

## Lưu ý

- App dùng đường dẫn tương đối nội bộ thư mục này, nên cần giữ nguyên cấu trúc thư mục.
- Nếu cập nhật model hoặc dữ liệu, commit lại các file trong:
  - `models/kmeans_by_district/`
  - `data/processed/`
