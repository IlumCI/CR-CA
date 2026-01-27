# Tools Overview

The `tools/` module provides external tool integration for CR-CA.

## Modules

- **[MCP Client](mcp-client.md)**: Model Context Protocol client utilities

## MCP Integration

CR-CA integrates with MCP (Model Context Protocol) servers for extended functionality.

## Usage

```python
from tools.mcp_client import MCPClient

client = MCPClient(server_url="http://localhost:8000")
result = client.call_tool("tool_name", params={})
```

## Next Steps

- [MCP Client](mcp-client.md) - MCP client utilities
