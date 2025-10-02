<div align="center">
    
# Demo Document Extraction
# Trích Xuất Metadata Từ Tài Liệu

## 📌 Giới thiệu

Dự án này được xây dựng nhằm **tự động trích xuất metadata** (thông tin mô tả) từ các loại tài liệu, hình ảnh hoặc file PDF. Metadata có thể bao gồm:

* Tiêu đề
* Tác giả
* Ngày tạo / chỉnh sửa
* Định dạng file
* Thông tin nhận dạng nội dung (tags, keywords)

## ⚙️ Công nghệ sử dụng

* **Python** cho xử lý logic chính
* **Thư viện OCR** (như `PaddleOCR` hoặc `Tesseract`) để nhận dạng văn bản trong ảnh/PDF
* **Transformers (Donut, HuggingFace)** để phân tích và bóc tách thông tin
* **GitHub** để quản lý và chia sẻ mã nguồn

## 🚀 Tính năng chính

* Tải tài liệu đầu vào (PDF, ảnh, text)
* Tự động phân tích và trích xuất metadata
* Xuất kết quả dưới dạng JSON hoặc CSV để dễ dàng tích hợp với hệ thống khác
* Có thể tùy chỉnh để nhận diện các trường metadata riêng theo nhu cầu

## 🎯 Ứng dụng thực tế

* Quản lý kho tài liệu số
* Hỗ trợ số hoá văn bản trong doanh nghiệp
* Tìm kiếm thông minh dựa trên metadata
* Tiền xử lý dữ liệu cho hệ thống Machine Learning / AI

## 👤 Tác giả

Dự án được phát triển bởi **tlinh902**, với mục tiêu nghiên cứu và ứng dụng AI vào xử lý tài liệu.

```
