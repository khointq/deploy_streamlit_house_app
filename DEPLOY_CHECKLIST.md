# ✅ DEPLOY CHECKLIST (Streamlit)

Dùng checklist này trước khi bấm Deploy.

## 1) Kiểm tra cấu trúc thư mục

- [ ] Có `app.py` ở root repo.
- [ ] Có `house_web_app.py` ở root repo.
- [ ] Có `requirements.txt`.
- [ ] Có `runtime.txt`.
- [ ] Có `.streamlit/config.toml`.
- [ ] Có thư mục `data/processed/` với các file:
  - [ ] `non_outlier_samples_6cols.csv`
  - [ ] `house_template.csv`
  - [ ] `admin_new_houses.csv`
- [ ] Có thư mục `models/kmeans_by_district/` với các model `.joblib`.

## 2) Kiểm tra chạy local

- [ ] Cài dependencies từ `requirements.txt`.
- [ ] Chạy app local bằng `streamlit run app.py`.
- [ ] Test nhanh các luồng chính:
  - [ ] Người mua xem danh sách + xem chi tiết.
  - [ ] Người bán đăng trực tiếp.
  - [ ] Người bán import Excel.
  - [ ] Dự đoán phân khúc hiển thị đúng (không còn `nan`).

## 3) Push lên GitHub

- [ ] Đã tạo repo GitHub.
- [ ] Đã commit toàn bộ file cần deploy.
- [ ] Đã push branch chính (`main` hoặc `master`).

## 4) Deploy trên Streamlit Community Cloud

- [ ] Chọn đúng repository.
- [ ] Branch: `main` (hoặc branch bạn deploy).
- [ ] Main file path: `app.py`.
- [ ] Bấm **Deploy**.

## 5) Smoke test sau deploy

- [ ] App mở không lỗi.
- [ ] Trang Người mua hiển thị dữ liệu.
- [ ] Trang Người bán import Excel hoạt động.
- [ ] Trang chi tiết nhà hiển thị phân khúc.
- [ ] Model KMeans theo quận được load và dự đoán được.

## 6) Ghi chú vận hành

- [ ] Nếu cập nhật dữ liệu/model, nhớ commit lại:
  - `data/processed/*`
  - `models/kmeans_by_district/*`
- [ ] Không commit thư mục môi trường ảo (`.venv/`).

---

**Release note:** Deploy bundle đã được chuẩn bị tại `deploy_streamlit_house_app`.
