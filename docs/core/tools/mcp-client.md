# MCP Client

MCP (Model Context Protocol) client utilities for CR-CA.

## Overview

MCP client enables CR-CA agents to interact with MCP servers for extended tool capabilities.

## Usage

```python
from tools.mcp_client import MCPClient

client = MCPClient(
    server_url="http://localhost:8000",
    api_key="your-api-key"
)

# Call MCP tool
result = client.call_tool(
    tool_name="causal_analysis",
    params={"variables": ["X", "Y"]}
)
```

## Features

- Tool discovery
- Tool execution
- Protocol handling

## Next Steps

- [Branches](../branches/crca-q/overview.md) - Specialized branches
