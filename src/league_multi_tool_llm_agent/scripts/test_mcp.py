import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main() -> None:
    async with streamablehttp_client("https://mcp-api.op.gg/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("\n--- Tools ---")
            print(tools)

            result = await session.call_tool(
                "lol_get_summoner_profile",
                {
                    "region": "na",
                    "game_name": "Pobelter",
                    "tag_line": "NA1",
                    "desired_output_fields": [
                        "data.summoner.{game_name,tagline,level}",
                        "data.summoner.league_stats[].{game_type,win,lose}",
                    ],
                },
            )

            print("\n--- Result ---")
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
