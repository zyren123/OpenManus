#!/usr/bin/env python
import argparse
import asyncio
import json
import sys
from typing import Dict, List

from app.agent.mcp import MCPAgent
from app.config import config
from app.logger import logger


class MCPRunner:
    """Runner class for MCP Agent with proper path handling and configuration."""

    def __init__(self):
        self.root_path = config.root_path
        self.server_reference = "app.mcp.server"
        self.agent = MCPAgent()

    async def initialize(
        self,
        connection_type: str,
        server_url: str | None = None,
        server_id: str = "default",
    ) -> None:
        """Initialize the MCP agent with the appropriate connection for a single server.

        Args:
            connection_type: Type of connection to use ("stdio" or "sse")
            server_url: URL of the MCP server for SSE connection
            server_id: Unique identifier for this server connection
        """
        logger.info(
            f"Initializing MCPAgent with {connection_type} connection to server '{server_id}'..."
        )

        if connection_type == "stdio":
            await self.agent.initialize(
                connection_type="stdio",
                command=sys.executable,
                args=["-m", self.server_reference],
                server_id=server_id,
            )
        else:  # sse
            await self.agent.initialize(
                connection_type="sse",
                server_url=server_url,
                server_id=server_id,
            )

        logger.info(f"Connected to MCP server '{server_id}' via {connection_type}")

    async def initialize_multiple(
        self,
        servers_config: List[Dict],
    ) -> None:
        """Initialize the MCP agent with connections to multiple servers.

        Args:
            servers_config: List of server configurations
        """
        logger.info(
            f"Initializing MCPAgent with {len(servers_config)} server connections..."
        )

        await self.agent.initialize_multiple_servers(servers_config)

        logger.info(f"Connected to {len(servers_config)} MCP servers")

    async def run_interactive(self) -> None:
        """Run the agent in interactive mode."""
        active_servers = self.agent.get_active_servers()
        print(
            f"\nMCP Agent Interactive Mode with {len(active_servers)} server(s): {', '.join(active_servers)}"
        )
        print("Type 'exit' to quit\n")

        while True:
            user_input = input("\nEnter your request: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            response = await self.agent.run(user_input)
            print(f"\nAgent: {response}")

    async def run_single_prompt(self, prompt: str) -> None:
        """Run the agent with a single prompt."""
        await self.agent.run(prompt)

    async def run_default(self) -> None:
        """Run the agent in default mode."""
        prompt = input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        await self.agent.run(prompt)
        logger.info("Request processing completed.")

    async def cleanup(self) -> None:
        """Clean up agent resources."""
        await self.agent.cleanup()
        logger.info("Session ended")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the MCP Agent")

    # Connection mode group
    connection_group = parser.add_mutually_exclusive_group()
    connection_group.add_argument(
        "--connection",
        "-c",
        choices=["stdio", "sse"],
        default="stdio",
        help="Connection type for a single server: stdio or sse",
    )
    connection_group.add_argument(
        "--servers",
        "-s",
        help="JSON string or path to JSON file with multiple server configurations",
    )

    # General arguments
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000/sse",
        help="URL for SSE connection (single server mode)",
    )
    parser.add_argument(
        "--server-id",
        default="default",
        help="Server ID for single server connection",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument("--prompt", "-p", help="Single prompt to execute and exit")

    return parser.parse_args()


def parse_servers_config(servers_arg: str) -> List[Dict]:
    """Parse the servers configuration from a JSON string or file.

    Args:
        servers_arg: JSON string or path to JSON file

    Returns:
        List of server configurations
    """
    if servers_arg.endswith(".json"):
        # Read from file
        try:
            with open(servers_arg, "r") as f:
                servers_config = json.load(f)
        except Exception as e:
            logger.error(f"Error reading servers configuration file: {str(e)}")
            sys.exit(1)
    else:
        # Parse JSON string
        try:
            servers_config = json.loads(servers_arg)
        except json.JSONDecodeError:
            logger.error("Invalid JSON string for servers configuration")
            sys.exit(1)

    # Validate configuration format
    if not isinstance(servers_config, list):
        logger.error("Servers configuration must be a list of server configurations")
        sys.exit(1)

    # Ensure each server has a unique ID
    server_ids = set()
    for i, server in enumerate(servers_config):
        if "server_id" not in server:
            server["server_id"] = f"server_{i}"
        elif server["server_id"] in server_ids:
            logger.error(f"Duplicate server ID: {server['server_id']}")
            sys.exit(1)
        server_ids.add(server["server_id"])

    return servers_config


async def run_mcp() -> None:
    """Main entry point for the MCP runner."""
    args = parse_args()
    runner = MCPRunner()

    try:
        # Initialize agent connections
        if args.servers:
            # Multiple servers mode
            servers_config = parse_servers_config(args.servers)
            await runner.initialize_multiple(servers_config)
        else:
            # Single server mode
            await runner.initialize(
                args.connection,
                args.server_url,
                args.server_id,
            )

        # Run the agent
        if args.prompt:
            await runner.run_single_prompt(args.prompt)
        elif args.interactive:
            await runner.run_interactive()
        else:
            await runner.run_default()

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Error running MCPAgent: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(run_mcp())
