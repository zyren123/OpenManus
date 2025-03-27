from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.mcp import MULTIMEDIA_RESPONSE_PROMPT, NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import AgentState, Message
from app.tool.base import ToolResult
from app.tool.mcp import MCPClients


class MCPAgent(ToolCallAgent):
    """Agent for interacting with MCP (Model Context Protocol) servers.

    This agent connects to MCP servers using either SSE or stdio transport
    and makes the servers' tools available through the agent's tool interface.
    Supports connecting to multiple MCP servers simultaneously.
    """

    name: str = "mcp_agent"
    description: str = "An agent that connects to MCP servers and uses their tools."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Initialize MCP tool collection
    mcp_clients: MCPClients = Field(default_factory=MCPClients)
    available_tools: MCPClients = None  # Will be set in initialize()

    max_steps: int = 20
    connection_type: str = "stdio"  # "stdio" or "sse"

    # Track active server connections
    active_servers: List[str] = Field(default_factory=list)
    default_server_id: str = "default"

    # Track tool schemas to detect changes
    tool_schemas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    _refresh_tools_interval: int = 5  # Refresh tools every N steps

    # Special tool names that should trigger termination
    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate"])

    async def initialize(
        self,
        connection_type: Optional[str] = None,
        server_url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        server_id: str = "default",
    ) -> None:
        """Initialize the MCP connection for a specific server.

        Args:
            connection_type: Type of connection to use ("stdio" or "sse")
            server_url: URL of the MCP server (for SSE connection)
            command: Command to run (for stdio connection)
            args: Arguments for the command (for stdio connection)
            server_id: Unique identifier for this server connection
        """
        if connection_type:
            self.connection_type = connection_type

        # Connect to the MCP server based on connection type
        if self.connection_type == "sse":
            if not server_url:
                raise ValueError("Server URL is required for SSE connection")
            await self.mcp_clients.connect_sse(
                server_url=server_url, server_id=server_id
            )
        elif self.connection_type == "stdio":
            if not command:
                raise ValueError("Command is required for stdio connection")
            await self.mcp_clients.connect_stdio(
                command=command, args=args or [], server_id=server_id
            )
        else:
            raise ValueError(f"Unsupported connection type: {self.connection_type}")

        # Set available_tools to our MCP instance
        self.available_tools = self.mcp_clients

        # Add to active servers list if not already present
        if server_id not in self.active_servers:
            self.active_servers.append(server_id)

        # Store initial tool schemas
        await self._refresh_tools()

        # Add system message about available tools for this server
        server_tools = self.mcp_clients.get_server_tools(server_id)
        tool_names = [tool.name for tool in server_tools]
        tools_info = ", ".join(tool_names)

        # Add system prompt and available tools information
        self.memory.add_message(
            Message.system_message(
                f"{self.system_prompt}\n\nConnected to MCP server '{server_id}' with tools: {tools_info}"
            )
        )

    async def initialize_multiple_servers(
        self, servers_config: List[Dict[str, Any]]
    ) -> None:
        """Initialize connections to multiple MCP servers.

        Args:
            servers_config: List of server configurations, each containing:
                - connection_type: "stdio" or "sse"
                - server_url: (For SSE) URL of the MCP server
                - command: (For stdio) Command to run
                - args: (For stdio) Arguments for the command
                - server_id: Unique identifier for this server
        """
        all_tools = []

        for config in servers_config:
            server_id = config.get("server_id", f"server_{len(self.active_servers)}")
            connection_type = config.get("connection_type", self.connection_type)

            await self.initialize(
                connection_type=connection_type,
                server_url=config.get("server_url"),
                command=config.get("command"),
                args=config.get("args", []),
                server_id=server_id,
            )

            # Collect tool names from this server
            server_tools = self.mcp_clients.get_server_tools(server_id)
            all_tools.extend([tool.name for tool in server_tools])

        # Add a consolidated system message with all tools across servers
        if len(servers_config) > 1:
            self.memory.add_message(
                Message.system_message(
                    f"Connected to {len(servers_config)} MCP servers with combined tools: {', '.join(all_tools)}"
                )
            )

    async def _refresh_tools(self) -> Tuple[List[str], List[str]]:
        """Refresh the list of available tools from all MCP servers.

        Returns:
            A tuple of (added_tools, removed_tools)
        """
        # Check if any sessions are available
        if not self.mcp_clients.sessions:
            return [], []

        # Update tools from all active servers
        current_tools = {}

        for server_id in self.active_servers:
            if server_id not in self.mcp_clients.sessions:
                continue

            # Get current tool schemas directly from the server
            session = self.mcp_clients.sessions[server_id]
            response = await session.list_tools()

            # Use the server_id prefix as in MCPClients._initialize_and_list_tools
            for tool in response.tools:
                tool_name = (
                    f"{server_id}_{tool.name}" if server_id != "default" else tool.name
                )
                current_tools[tool_name] = tool.inputSchema

        # Determine added, removed, and changed tools
        current_names = set(current_tools.keys())
        previous_names = set(self.tool_schemas.keys())

        added_tools = list(current_names - previous_names)
        removed_tools = list(previous_names - current_names)

        # Check for schema changes in existing tools
        changed_tools = []
        for name in current_names.intersection(previous_names):
            if current_tools[name] != self.tool_schemas.get(name):
                changed_tools.append(name)

        # Update stored schemas
        self.tool_schemas = current_tools

        # Log and notify about changes
        if added_tools:
            logger.info(f"Added MCP tools: {added_tools}")
            self.memory.add_message(
                Message.system_message(f"New tools available: {', '.join(added_tools)}")
            )
        if removed_tools:
            logger.info(f"Removed MCP tools: {removed_tools}")
            self.memory.add_message(
                Message.system_message(
                    f"Tools no longer available: {', '.join(removed_tools)}"
                )
            )
        if changed_tools:
            logger.info(f"Changed MCP tools: {changed_tools}")

        return added_tools, removed_tools

    async def think(self) -> bool:
        """Process current state and decide next action."""
        # Check if any MCP sessions are still available
        if not self.mcp_clients.sessions or not self.mcp_clients.tool_map:
            logger.info("All MCP services are no longer available, ending interaction")
            self.state = AgentState.FINISHED
            return False

        # Refresh tools periodically
        if self.current_step % self._refresh_tools_interval == 0:
            await self._refresh_tools()
            # All tools removed indicates shutdown
            if not self.mcp_clients.tool_map:
                logger.info("All MCP services have shut down, ending interaction")
                self.state = AgentState.FINISHED
                return False

        # Use the parent class's think method
        return await super().think()

    async def _handle_special_tool(self, name: str, result: Any, **kwargs) -> None:
        """Handle special tool execution and state changes"""
        # First process with parent handler
        await super()._handle_special_tool(name, result, **kwargs)

        # Handle multimedia responses
        if isinstance(result, ToolResult) and result.base64_image:
            self.memory.add_message(
                Message.system_message(
                    MULTIMEDIA_RESPONSE_PROMPT.format(tool_name=name)
                )
            )

    def _should_finish_execution(self, name: str, **kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        # Terminate if the tool name is 'terminate'
        return name.lower() == "terminate"

    async def cleanup(self, server_id: str = None) -> None:
        """Clean up MCP connection when done.

        Args:
            server_id: ID of the server to disconnect, or None for all servers
        """
        if server_id:
            # Disconnect specific server
            await self.mcp_clients.disconnect(server_id)
            if server_id in self.active_servers:
                self.active_servers.remove(server_id)
            logger.info(f"MCP connection to server '{server_id}' closed")
        else:
            # Disconnect all servers
            await self.mcp_clients.disconnect()
            self.active_servers = []
            logger.info("All MCP connections closed")

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with cleanup when done."""
        try:
            result = await super().run(request)
            return result
        finally:
            # Ensure cleanup happens even if there's an error
            await self.cleanup()

    def get_active_servers(self) -> List[str]:
        """Return list of active server connections."""
        return self.active_servers
