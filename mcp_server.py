from typing import List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return f"It's sunny in {location} with 70 degrees and a 10mph wind"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")