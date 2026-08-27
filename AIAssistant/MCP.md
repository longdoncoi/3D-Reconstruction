# MCP cho AI Agent

AI server cung cấp MCP Streamable HTTP tại `http://127.0.0.1:8080/mcp` khi
chạy `StartChatbotServer.py`. Cài dependency một lần bằng:

```powershell
python -m pip install -r AIAssistant/requirements.txt
```

Endpoint công khai các tool không phá huỷ: `read_file`, `list_directory`,
`search_text`, `analyze_code`, `get_project_status`, `validate_file`,
`rag_search`, và `application_action`.

Các tool thay đổi file hoặc chạy lệnh (`write_file`, `patch_file`,
`create_directory`, `run_command`) vẫn yêu cầu phê duyệt từ giao diện Agent;
chúng không được công bố qua MCP để tránh MCP client ghi trực tiếp vào dự án.

Agent nội bộ gọi các tool an toàn bằng MCP thay vì gọi executor Python trực
tiếp. Client MCP bên ngoài có thể kết nối vào cùng endpoint để khám phá và gọi
những tool an toàn này.
