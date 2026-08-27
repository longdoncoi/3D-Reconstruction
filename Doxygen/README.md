# README - HƯỚNG DẪN SỬ DỤNG DOXYGEN + GRAPHVIZ

1. GIỚI THIỆU

---

Tài liệu này hướng dẫn cách cấu hình và sử dụng Doxygen để sinh tài liệu tự động
cho source code C++ trong project 3D-Reconstruction.

Doxygen sẽ:

* Phân tích code (.h, .cpp)
* Sinh tài liệu HTML
* Vẽ sơ đồ class, quan hệ, call graph (khi dùng Graphviz)

---

2. CÀI ĐẶT CÔNG CỤ

---

2.1. Cài Doxygen

* Tải tại: https://www.doxygen.nl/download.html
* Cài đặt bình thường

Kiểm tra:

> doxygen -v

---

2.2. Cài Graphviz (bắt buộc để vẽ sơ đồ)

* Tải tại: https://graphviz.org/download/
* Sau khi cài, thêm vào PATH:
  Ví dụ:
  C:\Program Files\Graphviz\bin

Kiểm tra:

> dot -V

---

3. CẤU HÌNH FILE DOXYFILE

---

3.1. Chỉ định thư mục source

INPUT = ../src/app ../src/core ../src/modules ../src/services ../src/utils

---

3.2. Không quét thư mục con

RECURSIVE = NO

(Nếu project có src/include riêng thì nên dùng RECURSIVE = YES)

---

3.3. Chỉ lấy file source

FILE_PATTERNS = *.h *.hpp *.cpp

---

3.4. Loại bỏ file auto generate (Qt)

EXCLUDE_PATTERNS = */moc_* */ui_* moc_* ui_*

---

3.5. Loại bỏ thư mục không cần thiết

EXCLUDE = build docs models doxygen

---

4. BẬT SƠ ĐỒ (GRAPHVIZ)

---

HAVE_DOT = YES

DOT_PATH = C:/Program Files/Graphviz/bin

CLASS_DIAGRAMS = YES
CLASS_GRAPH = YES
COLLABORATION_GRAPH = YES

CALL_GRAPH = YES
CALLER_GRAPH = YES

INCLUDE_GRAPH = YES
INCLUDED_BY_GRAPH = YES

DIRECTORY_GRAPH = YES
GROUP_GRAPHS = YES

UML_LOOK = YES

---

5. TỐI ƯU HIỂN THỊ

---

DOT_IMAGE_FORMAT = svg
INTERACTIVE_SVG = YES

DOT_GRAPH_MAX_NODES = 100
MAX_DOT_GRAPH_DEPTH = 5

HIDE_UNDOC_RELATIONS = NO

---

6. TRÍCH XUẤT TOÀN BỘ CODE

---

EXTRACT_ALL = YES
EXTRACT_PRIVATE = YES
EXTRACT_STATIC = YES

---

7. CÁCH GENERATE TÀI LIỆU

---

Cách 1: Dùng command line

> doxygen Doxyfile

Cách 2: Dùng Doxygen GUI

* Mở Doxywizard
* Load file Doxyfile
* Nhấn "Run Doxygen"

---

8. KẾT QUẢ

---

Sau khi chạy xong:

* Mở file:
  docs/html/index.html

Bạn sẽ thấy:

* Danh sách class
* Documentation
* Sơ đồ UML
* Call graph
* Include graph

---

9. LỖI THƯỜNG GẶP

---

9.1. Không có sơ đồ

* Kiểm tra HAVE_DOT = YES
* Kiểm tra dot -V

---

9.2. Không thấy class

* Thiếu comment /** */
* Chưa bật EXTRACT_ALL

---

9.3. Bị include file build/

* Chưa set EXCLUDE

---

9.4. Không thấy file .cpp

* Sai FILE_PATTERNS

---

10. KHUYẾN NGHỊ

---

* Nên tổ chức project:
  src/
  include/
  build/
  docs/

* Viết comment theo chuẩn:
  /**

  * @brief Mô tả class/hàm
    */

* Không generate trong thư mục build

---                                              
`
