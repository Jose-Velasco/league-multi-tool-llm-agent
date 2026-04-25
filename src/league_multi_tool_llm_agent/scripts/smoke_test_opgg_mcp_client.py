import asyncio

from league_multi_tool_llm_agent.integrations.opgg import OPGGMCPClient


async def main() -> None:
    client = OPGGMCPClient()

    registry = await client.refresh_tool_registry()
    print(f"Loaded {len(registry)} tools")

    profile = await client.get_summoner_profile(
        riot_id="Pobelter#NA1",
        region="na",
    )
    print("\nPROFILE\n")
    print(client.extract_text(profile))

    # analysis = await client.get_champion_analysis(
    #     champion="AHRI",
    #     position="mid",
    # )
    # print("\nANALYSIS\n")
    # print(client.extract_text(analysis))

    recent_matches = await client.list_summoner_matches(
        riot_id="Pobelter#NA1", region="na", limit=5
    )
    print("\nRecent matches\n")
    print(client.extract_text(recent_matches))

    # print("\nTool metadata\n")
    # for name, tool in registry.items():
    #     print(
    #         f"{name}: "
    #         f"input_schema={'yes' if tool.input_schema else 'no'}, "
    #         f"output_schema={'yes' if getattr(tool, 'output_schema', None) else 'no'}"
    #     )

    # for name, tool in registry.items():
    #     print(name, "->", "JSON" if tool.input_schema else "TEXT")


if __name__ == "__main__":
    asyncio.run(main())
