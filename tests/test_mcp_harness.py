"""MCP test harness for testing server with different transports."""

import pytest
import subprocess
import time
import asyncio
from pathlib import Path


class MCPTestHarness:
    """Test harness for MCP server."""
    
    def __init__(self, transport="stdio", host="127.0.0.1", port=8000, path="/mcp"):
        self.transport = transport
        self.host = host
        self.port = port
        self.path = path
        self.process = None
    
    def start(self, extra_args=None):
        """Start the MCP server."""
        cmd = ["uv", "run", "jupyter-editor-mcp", "--transport", self.transport]
        
        if self.transport == "http":
            cmd.extend(["--host", self.host, "--port", str(self.port), "--path", self.path])
        
        if extra_args:
            cmd.extend(extra_args)
        
        if self.transport == "stdio":
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            self.process = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
            # Wait for HTTP server to start
            time.sleep(2)
    
    def stop(self):
        """Stop the MCP server."""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
    
    def list_tools_stdio(self):
        """List tools via stdio transport."""
        # First initialize
        init_request = '{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test-client", "version": "1.0.0"}}}\n'
        self.process.stdin.write(init_request)
        self.process.stdin.flush()
        init_response = self.process.stdout.readline()
        
        # Then list tools
        request = '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}\n'
        self.process.stdin.write(request)
        self.process.stdin.flush()
        
        response = self.process.stdout.readline()
        import json
        return json.loads(response)
    
    def list_tools(self):
        """List tools using appropriate transport."""
        if self.transport == "stdio":
            return self.list_tools_stdio()
        else:
            return asyncio.run(self.list_tools_http_async())
    
    async def list_tools_http_async(self):
        """List tools via HTTP transport using MCP client."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        # For HTTP, we need to use the HTTP client
        # But MCP SDK primarily supports stdio, so we'll use a workaround
        import httpx
        
        url = f"http://{self.host}:{self.port}{self.path}"
        
        async with httpx.AsyncClient() as client:
            # Initialize
            init_response = await client.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "1.0.0"}
                    }
                },
                headers={"Content-Type": "application/json"}
            )
            
            # List tools
            tools_response = await client.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {}
                },
                headers={"Content-Type": "application/json"}
            )
            
            return tools_response.json()


@pytest.fixture
def stdio_harness():
    """Fixture for stdio transport harness."""
    harness = MCPTestHarness(transport="stdio")
    yield harness
    harness.stop()


@pytest.fixture
def http_harness():
    """Fixture for HTTP transport harness."""
    harness = MCPTestHarness(transport="http", port=8001)
    yield harness
    harness.stop()


class TestMCPTransports:
    """Test MCP server with different transports."""
    
    def test_stdio_transport(self, stdio_harness):
        """Test server with stdio transport."""
        stdio_harness.start(["--no-banner"])
        response = stdio_harness.list_tools()
        
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) > 0
        
        # Verify some expected tools exist
        tool_names = [tool["name"] for tool in response["result"]["tools"]]
        assert "read_notebook" in tool_names
        assert "list_cells" in tool_names
    
    @pytest.mark.asyncio
    async def test_http_transport(self, http_harness):
        """Test server with Streamable HTTP transport."""
        http_harness.start(["--no-banner"])
        
        # Give server time to start
        await asyncio.sleep(1)
        
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        
        url = f"http://{http_harness.host}:{http_harness.port}{http_harness.path}"
        
        async with streamable_http_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                
                assert tools is not None
                assert len(tools.tools) > 0
                
                tool_names = [tool.name for tool in tools.tools]
                assert "read_notebook" in tool_names
                assert "list_cells" in tool_names
    
    @pytest.mark.asyncio
    async def test_custom_http_port(self):
        """Test Streamable HTTP transport with custom port."""
        harness = MCPTestHarness(transport="http", port=8002)
        harness.start(["--no-banner"])
        
        await asyncio.sleep(1)
        
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
            
            url = f"http://{harness.host}:{harness.port}{harness.path}"
            
            async with streamable_http_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    assert tools is not None
        finally:
            harness.stop()
    
    @pytest.mark.asyncio
    async def test_custom_http_path(self):
        """Test Streamable HTTP transport with custom path."""
        harness = MCPTestHarness(transport="http", port=8003, path="/custom")
        harness.start(["--no-banner"])
        
        await asyncio.sleep(1)
        
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
            
            url = f"http://{harness.host}:{harness.port}{harness.path}"
            
            async with streamable_http_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    assert tools is not None
        finally:
            harness.stop()
    
    def test_project_scope_parameter(self, stdio_harness, tmp_path):
        """Test --project parameter sets scope correctly."""
        stdio_harness.start(["--no-banner", "--project", str(tmp_path)])
        
        # Server should start successfully with project scope
        response = stdio_harness.list_tools()
        assert "result" in response
