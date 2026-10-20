# ADR 0028 — Cache RAG được mã hoá at-rest bằng chính key của state

## Bối cảnh

ADR 0027 mã hoá *state* của agent (task, pending action) — prompt, đối số tool,
kết quả model và event log. Nhưng còn một kho khác mà người dùng không nghĩ
tới, và nó **chứa bản sao thứ hai của toàn bộ corpus tài liệu**:

| Artifact | Chứa gì |
|---|---|
| `Cache/chunks.pkl` | `ChunkResult.text` (nguyên văn từng đoạn tài liệu), `source_path`, `parent_text`, và `image_b64` — **ảnh tài liệu ở dạng base64** |
| `Cache/bm25.pkl` | corpus đã token hoá (BM25Okapi giữ toàn bộ token của tài liệu) |
| `Cache/faiss_index.bin` | embedding của từng chunk |

Cả ba trước đây được ghi **plaintext**. `rag/cache_integrity.py` (nay đã đổi
tên thành `rag/cache_seal.py`) ký HMAC lên chúng — nhưng HMAC là
**integrity, không phải confidentiality**: nó chứng minh "file này là file tôi
đã viết", không ngăn ai đọc. Chính module đó đã nói thẳng trong docstring:
*"it is not a defence against a local attacker with full account access"*. Đúng
là vậy — nhưng nó cũng không ngăn **một bản sao volume dữ liệu** mang theo
corpus tài liệu ở dạng đọc được.

Nói cách khác: sau ADR 0027, `tasks.sqlite` đã không còn lộ nội dung, nhưng
`Cache/chunks.pkl` thì vẫn lộ. Một review độc lập sẽ gọi đây là "sửa một
chỗ, để lại chỗ kia" — và đúng là vậy.

Quan trọng: cache là **artifact dẫn xuất** — nó luôn dựng lại được từ tài
liệu nguồn. Nên hậu quả của việc loại bỏ nó là **thời gian**, không phải dữ
liệu. Đó là điều cho phép thiết kế fail-closed một cách dứt khoát.

## Quyết định

1. **Một key cho mọi thứ app ghi ra đĩa.** Ba artifact được seal bằng **đúng
   cipher và key** mà ADR 0027 dùng cho task/pending state
   (`AI_STORAGE_KEY` → `AI_STORAGE_KEY_FILE` → key sinh một lần ngoài
   `APP_DATA_DIR`). Không có key thứ hai: `<data_dir>/.secrets/cache_hmac_key`
   bị bỏ, và lợi thế của nó là **backup volume dữ liệu không mang theo key**.
2. **Envelope nhị phân riêng cho artifact nhị phân.** `seal_bytes` /
   `open_bytes` dùng cùng AEAD nhưng lưu `enc:v1:<nonce || ciphertext>` **raw**,
   không base64: `chunks.pkl` có thể vài chục MB, base64 sẽ phồng lên 1/3 mà
   không đổi gì. Bản text (`enc:v1:<base64>`) giữ nguyên để ADR 0027 không
   phải migrate.
3. **`FieldCipher` chuyển vào tầng port.** Protocol (kèm hierarchy lỗi
   `StorageCryptoError` / `StorageKeyError` / `StorageDecryptionError`) sống ở
   `ai_assistant/ports.py`, nên `rag` phụ thuộc **vào trong** (port), không
   import adapter. `RagModule` **nhận cipher qua constructor** — không có
   chế độ "không có key": cache không seal được thì không được ghi.
4. **Index FAISS không bao giờ chạm đĩa ở dạng rõ.** `faiss.write_index` sẽ
   đặt vector lên đĩa trước; thay vào đó index được **serialize trong bộ nhớ**
   (`faiss.serialize_index`) rồi seal, và dựng lại bằng
   `faiss.deserialize_index`. Không còn cửa sổ plaintext nào.
5. **HMAC bị bỏ, GCM thay thế.** GCM tự xác thực byte đã mã hoá, nên digest
   HMAC là thừa và là một nguồn key thứ hai. `metadata.json` ghi lại
   `sealed: {algo, version}`; cache **không có** marker này là **miss** (cache
   cũ chưa seal, hoặc cache của version khác) → dựng lại.
6. **Fail-closed tuyệt đối.** Cache không xác thực được — sai key, bị sửa,
   chưa seal, thiếu marker, thiếu file — đều là **miss**: bị bỏ và dựng lại,
   **không bao giờ** unpickle một payload chưa xác thực. Lỗi cũng được phân
   biệt trong log: *"not sealed"* (cache của version cũ) khác *"could not be
   opened"* (sai key / hỏng / bị can thiệp).

## Hệ quả

- **Bản sao thứ hai của corpus không còn đọc được** từ một bản sao volume dữ
  liệu: `chunks.pkl` trên đĩa chỉ là `enc:v1:<nonce||ciphertext>`.
- **Một lần rebuild sau khi nâng cấp.** Cache cũ (plaintext) không được
  "migrate" mà bị **thay** — đây là chủ ý: chỉ có cách đó mới bảo đảm không
  còn file plaintext nào sót lại. Cache là dẫn xuất nên chi phí chỉ là thời
  gian dựng lại.
- **Tên file giữ nguyên** (`chunks.pkl`, `bm25.pkl`, `faiss_index.bin`) để
  đường dẫn và `is_cache_valid` không đổi — nhưng **nội dung không còn là
  pickle**. Tài liệu phải nói rõ điều này, không được để tên file gợi ý sai.
- **Một key thay vì hai.** `cache_hmac_key` không còn tồn tại; toàn bộ state
  at-rest của app dùng chung một key, nên backup key cũng chỉ cần một thứ.
- **Chi phí:** serialize/deserialize thay vì đọc file FAISS trực tiếp (một
  lần copy trong RAM), và mỗi artifact lớn hơn 12 byte (nonce) + 16 byte
  (tag). Không đáng kể so với chi phí dựng lại index.
- **Ghim bằng test và mutation:** `tests/unit/test_rag_cache_seal.py` (hợp
  đồng format), `tests/unit/test_index.py` (artifact trên đĩa là envelope,
  cache legacy là miss, cache sai key là miss, artifact bị sửa là miss, marker
  là thứ quyết định cache có hợp lệ không), `tests/unit/test_storage_crypto.py`
  (envelope nhị phân: round trip, context binding, tamper, không có đường
  legacy). **5 mutant mới** trong `scripts/mutation_check.py` đảo từng quyết
  định: artifact ghi rõ, cache chưa seal bị nạp, marker bị mất, envelope
  nhị phân bị tamper, artifact chưa seal bị nhận làm plaintext.

## Không giải quyết

- **Tài liệu nguồn vẫn là file của người dùng.** ADR này phủ *bản sao* mà
  app tự tạo ra; việc mã hoá thư mục tài liệu gốc là việc của người dùng và
  vận hành (mã hoá toàn volume — xem `Docs/RUNBOOK.md` §1.1).
- **Embedding vẫn là hàm của tài liệu.** Cache được seal *trên đĩa*; việc suy
  ra văn bản từ vector là một bài toán khác, nằm ngoài phạm vi at-rest.
- **Xoay key, KMS/DPAPI**: như ADR 0027 — chưa implement, đã ghi rõ.
