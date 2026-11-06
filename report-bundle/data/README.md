# Thư mục dữ liệu thực nghiệm

Đây là nguồn duy nhất của mọi hình và bảng số liệu trong báo cáo. Agent chỉ được đọc, không được tạo, sửa hay xóa file trong thư mục này.

## Quy tắc

1. Dữ liệu phải là kết quả thật từ thí nghiệm, khảo sát, hoặc nguồn công khai có trích dẫn.
2. Không đặt dữ liệu giả lập, dữ liệu mẫu, dữ liệu ngẫu nhiên, hay dữ liệu "minh họa" vào đây. `make_figures.py` sẽ từ chối các file có dấu hiệu như vậy.
3. Mỗi file `.csv` phải có file metadata đi kèm, cùng tên với đuôi `.meta.json`.
4. Dùng dấu chấm làm dấu thập phân trong CSV (0.87, không phải 0,87). Mã hóa UTF-8.
5. Tên file không dấu, không khoảng trắng.

## Ví dụ file CSV

`experiment-results.csv`:
```csv
mo_hinh,f1,thoi_gian_ms
Baseline,<giá trị từ log thí nghiệm>,<giá trị từ log>
De-xuat,<giá trị từ log thí nghiệm>,<giá trị từ log>
```
Thay các giá trị trong ngoặc bằng số thật lấy từ kết quả của bạn. Không điền số ước lượng.

## Ví dụ metadata: `experiment-results.csv.meta.json`

```json
{
  "nguon": "Chạy trên tập kiểm tra của đồ án, log tại results/run_2026-10-20/",
  "nguoi_tao": "Họ và tên sinh viên",
  "ngay_tao": "2026-10-20",
  "lenh_sinh": "python train.py --config configs/final.yaml --seed 42",
  "mo_ta_cot": {
    "mo_hinh": "Tên mô hình",
    "f1": "Độ đo F1 macro trên tập kiểm tra",
    "thoi_gian_ms": "Thời gian suy luận trung bình trên một mẫu, đo trên GPU RTX 3060"
  }
}
```

Các trường bắt buộc: `nguon`, `nguoi_tao`, `ngay_tao`, `lenh_sinh`. Lệnh sinh phải đủ để chạy lại thí nghiệm và tạo ra đúng file CSV này.
