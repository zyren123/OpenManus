from contextlib import AsyncExitStack
from typing import Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.tool_collection import ToolCollection


class MCPClientTool(BaseTool):
    """Represents a tool proxy that can be called on the MCP server from the client side."""

    session: Optional[ClientSession] = None
    # Add server_id to identify which server this tool belongs to
    server_id: str = "default"

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool by making a remote call to the MCP server."""
        if not self.session:
            return ToolResult(error="Not connected to MCP server")

        try:
            result = await self.session.call_tool(self.name, kwargs)
            content_str = ", ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            return ToolResult(output=content_str or "No output returned.")
        except Exception as e:
            return ToolResult(error=f"Error executing tool: {str(e)}")


class MCPClients(ToolCollection):
    """
    A collection of tools that connects to an MCP server and manages available tools through the Model Context Protocol.
    Supports multiple server connections with unique server IDs.
    """

    # Dictionary to store sessions mapped to server IDs
    sessions: Dict[str, ClientSession] = {}
    # Dictionary to store exit stacks mapped to server IDs
    exit_stacks: Dict[str, AsyncExitStack] = {}
    description: str = "MCP client tools for server interaction"

    # Track active server IDs
    active_servers: List[str] = []

    def __init__(self):
        super().__init__()  # Initialize with empty tools list
        self.name = "mcp"  # Keep name for backward compatibility
        self.sessions = {}
        self.exit_stacks = {}
        self.active_servers = []

    async def connect_sse(self, server_url: str, server_id: str = "default") -> None:
        """
        Connect to an MCP server using SSE transport.

        Args:
            server_url: URL of the MCP server
            server_id: Unique identifier for this server connection
        """
        if not server_url:
            raise ValueError("Server URL is required.")

        # Disconnect if the server_id already exists
        if server_id in self.sessions:
            await self.disconnect(server_id)

        # Create new exit stack for this connection
        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        streams_context = sse_client(url=server_url)
        streams = await exit_stack.enter_async_context(streams_context)
        session = await exit_stack.enter_async_context(ClientSession(*streams))

        # Store the session
        self.sessions[server_id] = session

        # Add to active servers list if not present
        if server_id not in self.active_servers:
            self.active_servers.append(server_id)

        await self._initialize_and_list_tools(server_id)

    async def connect_stdio(
        self, command: str, args: List[str], server_id: str = "default"
    ) -> None:
        """
        Connect to an MCP server using stdio transport.

        Args:
            command: Command to run the MCP server
            args: Arguments for the command
            server_id: Unique identifier for this server connection
        """
        if not command:
            raise ValueError("Server command is required.")

        # Disconnect if the server_id already exists
        if server_id in self.sessions:
            await self.disconnect(server_id)

        # Create new exit stack for this connection
        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        server_params = StdioServerParameters(command=command, args=args)
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await exit_stack.enter_async_context(ClientSession(read, write))

        # Store the session
        self.sessions[server_id] = session

        # Add to active servers list if not present
        if server_id not in self.active_servers:
            self.active_servers.append(server_id)

        await self._initialize_and_list_tools(server_id)

    async def _initialize_and_list_tools(self, server_id: str = "default") -> None:
        """
        Initialize session and populate tool map for a specific server.

        Args:
            server_id: ID of the server to initialize tools for
        """
        if server_id not in self.sessions:
            raise RuntimeError(f"Session for server '{server_id}' not initialized.")

        session = self.sessions[server_id]
        await session.initialize()
        response = await session.list_tools()

        # Create tool objects for this server
        server_tools = []
        for tool in response.tools:
            # Create a prefix for tool names to avoid conflicts between servers
            tool_name = (
                f"{server_id}_{tool.name}" if server_id != "default" else tool.name
            )

            server_tool = MCPClientTool(
                name=tool_name,
                description=f"[{server_id}] {tool.description}",
                parameters=tool.inputSchema,
                session=session,
                server_id=server_id,
            )
            self.tool_map[tool_name] = server_tool
            server_tools.append(server_tool)

        # Update tools tuple with all tools from all servers
        self.tools = tuple(self.tool_map.values())

        logger.info(
            f"Connected to server '{server_id}' with tools: {[tool.name for tool in response.tools]}"
        )

    async def disconnect(self, server_id: str = None) -> None:
        """
        Disconnect from MCP server(s) and clean up resources.

        Args:
            server_id: ID of the server to disconnect, or None to disconnect all
        """
        if server_id is None:
            # Disconnect all servers
            for sid in list(self.sessions.keys()):
                await self.disconnect(sid)
            return

        if server_id in self.sessions and server_id in self.exit_stacks:
            # Remove tools associated with this server
            tools_to_remove = [
                name
                for name, tool in self.tool_map.items()
                if getattr(tool, "server_id", None) == server_id
            ]

            for tool_name in tools_to_remove:
                del self.tool_map[tool_name]

            # Close the exit stack for this server
            await self.exit_stacks[server_id].aclose()

            # Remove server from sessions and exit_stacks
            del self.sessions[server_id]
            del self.exit_stacks[server_id]

            # Remove from active servers list
            if server_id in self.active_servers:
                self.active_servers.remove(server_id)

            # Update tools tuple
            self.tools = tuple(self.tool_map.values())

            logger.info(f"Disconnected from MCP server '{server_id}'")

    def get_server_ids(self) -> List[str]:
        """Return list of active server IDs."""
        return self.active_servers

    def get_server_tools(self, server_id: str) -> List[BaseTool]:
        """Return list of tools for a specific server."""
        return [
            tool for tool in self.tools if getattr(tool, "server_id", None) == server_id
        ]
