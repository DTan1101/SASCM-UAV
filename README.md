# 🛰️ SASCM-UAV: Triển khai các Module Nền tảng cho Định vị Thị giác Tuyệt đối

Đây là kho lưu trữ chứa triển khai các module/thuật toán cơ bản, đóng vai trò then chốt trong các hệ thống **Định vị Thị giác Tuyệt đối (Absolute Visual Localization)** tiên tiến, đặc biệt là kiến trúc khớp ảnh phân cấp **SASCM (Semantic-Aware and Structure-Constrained Matching)** cho UAV.

Các module này đại diện cho các khái niệm cốt lõi trong **Lý thuyết Học sâu** (Học Biểu diễn Tự giám sát) và **Thị giác Máy tính** (Ước tính Tham số Mạnh mẽ).

## 🌟 Tổng quan các Module được Triển khai trong Bối cảnh SASCM

Kho lưu trữ này hiện thực ba khái niệm/module nền tảng:

| Module | Phân loại | Khái niệm Lý thuyết chính | Vai trò trong Pipeline SASCM-UAV |
| :--- | :--- | :--- | :--- |
| **DINOv2** | Học tự giám sát (SSL) & ViT | **Đặc trưng Thị giác Chung (General-Purpose Visual Features)** | Cung cấp **Đặc trưng Ngữ nghĩa (Semantic Features)** cho **Coarse Matching (Khớp Thô)** để định hướng ban đầu. |
| **NCNet** | Mạng lưới Đối ứng (Correspondence Network) | **Đồng thuận Lân cận (Neighbourhood Consensus)** | Cung cấp khuôn khổ cho **Khớp Mật độ Cao (Dense Matching)**, đảm bảo các điểm đối ứng đáng tin cậy giữa ảnh UAV và bản đồ tham chiếu. |
| **RANSAC** | Thuật toán Model Fitting | **Ước lượng Tham số Mạnh mẽ (Robust Parameter Estimation)** | Lọc ra các điểm ngoại lai (outliers) và ước lượng **Tư thế/Vị trí tuyệt đối (Absolute Pose)** cuối cùng thông qua bài toán PnP. |

-----

## 1\. DINOv2 (Self-Supervised Vision Transformer) 🦖

### 🧠 Lý thuyết & Bối cảnh: Học Biểu diễn Ngữ nghĩa

[cite\_start]DINOv2 là một mô hình **Vision Transformer (ViT)** được huấn luyện bằng phương pháp **Học Tự giám sát (SSL)** trên tập dữ liệu đa dạng và lớn (LVD-142M) mà **không cần nhãn**.

  * **Tính Nền tảng:** DINOv2 tạo ra các đặc trưng thị giác hoạt động tốt **ngay cả khi bị đóng băng** (frozen), thể hiện tính **chuyển giao (transferability)** cao, đặc biệt hữu ích cho việc trích xuất thông tin ngữ nghĩa.
  * **Giá trị Lý thuyết:** Nó củng cố khái niệm rằng các mô hình lớn, được huấn luyện trên dữ liệu đa dạng có thể học được các biểu diễn **phân biệt (discriminative)** mạnh mẽ, có thể được áp dụng trực tiếp cho các tác vụ cấp độ pixel như Phân đoạn Ngữ nghĩa và Ước tính Chiều sâu.

### 🛠️ Vai trò trong Khớp Thô (Coarse Matching) của SASCM

Trong kiến trúc SASCM-UAV, DINOv2 đóng vai trò là xương sống (backbone) chính:

1.  **Trích xuất Đặc trưng Ngữ nghĩa:** DINOv2 được sử dụng để trích xuất **đặc trưng ngữ nghĩa dày đặc** từ cả ảnh UAV query và ảnh bản đồ vệ tinh reference.
2.  **Khớp Lớp Cao:** Các đặc trưng ngữ nghĩa này giúp giảm đáng kể sự mơ hồ do sự khác biệt giữa các nguồn (cross-source) và biến đổi thời gian (temporal variations), định vị ảnh UAV vào một khu vực thô trên bản đồ trước khi chuyển sang bước khớp hạt mịn hơn.

-----

## 2\. NCNet (Neighbourhood Consensus Network) 🌐

### 🧠 Lý thuyết & Bối cảnh: Đồng thuận Hình học

[cite\_start]NCNet là một kiến trúc CNN đầu-cuối học cách ước tính các điểm tương ứng (correspondences) dày đặc và đáng tin cậy.

  * **Cơ chế cốt lõi:** NCNet vượt qua giới hạn của phép khớp láng giềng gần nhất (Nearest Neighbour) bằng cách phân tích **mô hình đồng thuận lân cận** trong không gian 4D của tất cả các điểm khớp tiềm năng.
  * **Đồng thuận Lân cận:** [cite\_start]Nó hoạt động trên nguyên lý rằng một điểm tương ứng mơ hồ có thể được củng cố bởi sự đồng thuận hình học của các điểm tương ứng **chắc chắn, duy nhất** xung quanh nó.

### 🛠️ Vai trò trong SASCM (Cơ sở cho Khớp Hạt Mịn)

Mặc dù kiến trúc SASCM có thể sử dụng các mạng khớp nhẹ hơn (như **XFeat**), ý tưởng cốt lõi của NCNet là nền tảng:

1.  **Chuyển đổi 4D:** Phương pháp xử lý **Bản đồ Tương quan 4D** của NCNet là cơ sở cho các kỹ thuật khớp ảnh hiện đại, nơi mà việc tìm kiếm **đồng thuận (consensus)** là cần thiết để tạo ra các điểm đối ứng chính xác trong các khu vực khó.
2.  **Độ Tin cậy:** NCNet cung cấp một mô hình mạnh mẽ để lọc các điểm tương ứng kém tin cậy, chuyển giao các điểm khớp **chất lượng cao** cho giai đoạn Ước lượng Tư thế tiếp theo.

-----

## 3\. RANSAC (Random Sample Consensus) 🎯

### 🧠 Lý thuyết & Bối cảnh: Ước lượng Mạnh mẽ

[cite\_start]RANSAC là một **mô hình mới** để khớp mô hình với dữ liệu thực nghiệm, được thiết kế để xử lý dữ liệu chứa **tỷ lệ lỗi thô đáng kể (outliers)**.

  * **Khác biệt với Bình phương tối thiểu:** [cite\_start]Trong khi các phương pháp cổ điển như Bình phương tối thiểu (Least Squares) bị ảnh hưởng nặng nề bởi outliers, RANSAC đạt được sự mạnh mẽ (robustness) bằng cách chỉ sử dụng một **tập hợp con tối thiểu** ($n$) các điểm dữ liệu để khởi tạo mô hình.
  * **Bài toán LDP/PnP:** [cite\_start]Đây là thuật toán tiêu chuẩn để giải quyết **Bài toán Xác định Vị trí (LDP)** hay **Perspective-n-Point (PnP)**, mục tiêu là tìm vị trí **Trung tâm Phối cảnh (CP)** (vị trí camera) từ các điểm ảnh 2D và điểm mốc 3D đã biết.

### 🛠️ Vai trò trong Khớp Hạt Mịn (Fine-Grained Matching) của SASCM

RANSAC là bước cuối cùng nhưng quan trọng nhất trong việc tính toán vị trí tuyệt đối (pose) của UAV:

1.  **Lọc Hình học:** Sau khi các module khớp ảnh (như NCNet hoặc XFeat) tạo ra các điểm tương ứng, RANSAC được áp dụng để **loại bỏ các điểm khớp sai** (ngoại lai) và tìm ra ma trận Biến đổi (Transformation) hoặc Tư thế (Pose) nhất quán nhất.
2.  **Ước lượng Tư thế Cuối cùng:** Tập hợp đồng thuận (Consensus Set) được tìm thấy bởi RANSAC sau đó được sử dụng để tính toán tư thế camera (R, t) chính xác, hoàn thành nhiệm vụ **Định vị Tuyệt đối**.

-----

## 🚀 Kết nối các Module trong Pipeline SASCM-UAV

Kiến trúc **SASCM** tổng thể được xây dựng trên sự kết hợp giữa ngữ nghĩa (DINOv2) và hình học mạnh mẽ (NCNet/RANSAC):

1.  **DINOv2** cung cấp **Đặc trưng Ngữ nghĩa** cho **Khớp Thô (Coarse Matching)**.
2.  Các đặc trưng này sau đó được tinh chỉnh thông qua **Khớp Hạt Mịn** (dựa trên các nguyên tắc như **Đồng thuận Lân cận** của NCNet) để tìm ra các điểm tương ứng pixel-level chính xác.
3.  Cuối cùng, **RANSAC** áp dụng **Ràng buộc Cấu trúc/Hình học** để lọc các điểm ngoại lai và ước lượng tư thế UAV chính xác (PnP).

-----

## 🛠️ Yêu cầu và Cài đặt

**(Đây là phần bạn sẽ điền chi tiết về môi trường và cách chạy code)**

```bash
# Clone the repository
git clone [YOUR_REPO_URL]
cd [REPO_NAME]

# Cài đặt môi trường (ví dụ: với conda)
conda create -n sascm_deep_modules python=3.x
conda activate sascm_deep_modules

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# Để chạy các module/ví dụ cụ thể, vui lòng tham khảo thư mục tương ứng.
# Ví dụ:
# python examples/run_ransac_pnp.py
# python examples/run_ncnet_matcher.py
# python examples/run_dinov2_feature_extraction.py
```

-----