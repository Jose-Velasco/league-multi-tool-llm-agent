import asyncio
import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from litellm import acompletion
from PIL import Image
from tqdm import tqdm

from league_multi_tool_llm_agent.models.champion import Champion, ChampionSkin
from league_multi_tool_llm_agent.models.rag_configs import VisionSettings


def champion_to_profile_doc(champion: Champion) -> dict[str, Any]:
    abilities = champion.abilities

    content = "\n".join(
        [
            f"Champion: {champion.name}",
            f"Subtitle: {champion.subtitle}",
            f"Main role: {champion.main_role}",
            f"Play styles: {', '.join(champion.play_styles)}",
            (
                "Difficulty: "
                f"{champion.difficulty.difficulty} "
                f"({champion.difficulty.difficulty_name})"
            ),
            f"Description: {champion.description}",
            f"OP.GG summary: {champion.op_gg_summary}",
            "Abilities:",
            (
                f"Passive: {abilities.passive.name} - "
                f"{abilities.passive.description_high_level}"
            ),
            f"Q: {abilities.Q.name} - {abilities.Q.description_high_level}",
            f"W: {abilities.W.name} - {abilities.W.description_high_level}",
            f"E: {abilities.E.name} - {abilities.E.description_high_level}",
            f"R: {abilities.R.name} - {abilities.R.description_high_level}",
        ]
    )

    return {
        "doc_type": "champion_profile",
        "champion_name": champion.name,
        "skin_name": None,
        "main_role": champion.main_role,
        "difficulty": champion.difficulty.difficulty,
        "source_url": str(champion.official_lol_profile_details_website_url),
        "content": content,
        "meta_json": {
            "subtitle": champion.subtitle,
            "play_styles": champion.play_styles,
            "difficulty_name": champion.difficulty.difficulty_name,
            "op_gg_link": str(champion.op_gg_link),
        },
    }


def encode_image_to_base64(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


# vision_settings = VisionSettings()


# def image_path_to_data_url(image_path: str) -> str:
#     path = Path(image_path)
#     mime_type, _ = mimetypes.guess_type(path.name)
#     mime_type = mime_type or "image/png"
#     encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
#     return f"data:{mime_type};base64,{encoded}"


# def image_path_to_data_url(image_path: Path) -> str:
#     # path = Path(image_path)
#     mime_type, _ = mimetypes.guess_type(image_path.name)
#     mime_type = mime_type or "image/png"
#     encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
#     return f"data:{mime_type};base64,{encoded}"


def image_path_to_resized_data_url(image_path: Path, max_side: int = 512) -> str:
    # path = Path(image_path)

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=80, optimize=True)
        image_bytes = buffer.getvalue()

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


async def generate_skin_description(
    *,
    image_path: str,
    champion_name: str,
    skin_name: str,
    vision_settings: VisionSettings,
    root_skin_imgs_dir: Path,
) -> str:
    # image_data_url = image_path_to_data_url(root_skin_imgs_dir / image_path)
    image_data_url = image_path_to_resized_data_url(
        root_skin_imgs_dir / image_path,
        max_side=512,  # 👈 tweak here only
    )

    prompt = f"""
        Champion: {champion_name}
        Skin: {skin_name}

        Write one concise but detailed paragraph describing the visual appearance of this League of Legends skin for semantic retrieval.

        Focus on:
        - dominant colors
        - outfit, armor, or costume
        - theme or aesthetic (fantasy, cyber, spirit, dark, elegant, cute, celestial, infernal, etc.)
        - visible motifs, magical effects, weapons, animal or elemental cues
        - overall mood and vibe

        Rules:
        - return plain text only
        - do not say "this image shows"
        - do not mention camera framing, UI, or logos
        - do not speculate about lore beyond what is visually apparent
        - aim for 80 to 140 words
        """.strip()

    response = await acompletion(
        model=vision_settings.VISION_MODEL,
        api_base=vision_settings.VISION_API_BASE,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            }
        ],
        temperature=vision_settings.VISION_TEMPERATURE,
        max_tokens=vision_settings.VISION_MAX_TOKENS,
    )

    return response["choices"][0]["message"]["content"].strip()  # pyright: ignore[reportIndexIssue]


async def process_skin_description(
    *,
    champion: Any,
    skin: Any,
    semaphore: asyncio.Semaphore,
    vision_settings: VisionSettings,
    root_skin_imgs_dir: Path,
) -> None:
    if skin.skin_description:
        return
    if not skin.img_path:
        return

    async with semaphore:
        try:
            skin.skin_description = await generate_skin_description(
                image_path=skin.img_path,
                champion_name=champion.name,
                skin_name=skin.skin_name,
                vision_settings=vision_settings,
                root_skin_imgs_dir=root_skin_imgs_dir,
            )
        except Exception:
            # Safe fallback so ingestion keeps moving
            skin.skin_description = f"{skin.skin_name} is a skin for {champion.name}."


async def enrich_all_skin_descriptions(
    *,
    champions: list[Champion],
    vision_settings: VisionSettings,
    root_skin_imgs_dir: Path,
) -> None:
    # cfg = settings
    semaphore = asyncio.Semaphore(vision_settings.VISION_MAX_CONCURRENCY)

    all_pairs = [(champion, skin) for champion in champions for skin in champion.skins]

    progress = tqdm(total=len(all_pairs), desc="Generating skin descriptions")

    async def wrapped(champion: Any, skin: Any) -> None:
        try:
            await process_skin_description(
                champion=champion,
                skin=skin,
                semaphore=semaphore,
                vision_settings=vision_settings,
                root_skin_imgs_dir=root_skin_imgs_dir,
            )
        finally:
            progress.update(1)
            progress.set_postfix({"champion": champion.name})

    await asyncio.gather(*(wrapped(champion, skin) for champion, skin in all_pairs))
    progress.close()


# async def generate_skin_description(
#     *,
#     image_path: str,
#     champion_name: str,
#     skin_name: str,
#     agent: Agent[None, SkinDescriptionOutput],
#     skin_imgs_dir: Path,
# ) -> str:

#     image_bytes = (skin_imgs_dir / image_path).read_bytes()

#     prompt = f"""
#         Champion: {champion_name}
#         Skin: {skin_name}

#         Respond ONLY in JSON:
#         {{
#         "description": "..."
#         }}
#     """

#     result = await agent.run(
#         [
#             # f"""
#             # Champion: {champion_name}
#             # Skin: {skin_name}
#             # Generate a detailed visual description of this skin for semantic retrieval.
#             # """,
#             prompt,
#             BinaryContent(
#                 data=image_bytes,
#                 media_type="image/png",  # adjust if needed
#             ),
#         ]
#     )

#     return result.output.description


# async def generate_skin_description(
#     *,
#     image_path: str,
#     champion_name: str,
#     skin_name: str,
#     agent: Agent[None, SkinDescriptionOutput],
# ) -> str:
#     image_b64 = encode_image_to_base64(image_path)

#     prompt = f"""
#         Champion: {champion_name}
#         Skin: {skin_name}

#         Generate a detailed visual description of this skin for semantic retrieval.
#     """

#     result = await agent.run(
#         prompt,
#         images=[image_b64],
#     )

#     return result.output.description


# #####
async def skin_to_doc(
    champion: Champion,
    skin: ChampionSkin,
    # skin_description_agent: Agent[None, SkinDescriptionOutput],
    # skin_imgs_dir: Path,
) -> dict[str, Any]:
    # if not skin.skin_description and skin.img_path:
    #     skin.skin_description = await generate_skin_description(
    #         image_path=skin.img_path,
    #         champion_name=champion.name,
    #         skin_name=skin.skin_name,
    #         agent=skin_description_agent,
    #         skin_imgs_dir=skin_imgs_dir,
    #     )

    parts = [
        f"Champion: {champion.name}",
        f"Main role: {champion.main_role}",
        f"Skin: {skin.skin_name}",
    ]

    if skin.skin_description:
        parts.append(f"Skin description: {skin.skin_description}")

    return {
        "doc_type": "champion_skin",
        "champion_name": champion.name,
        "skin_name": skin.skin_name,
        "main_role": champion.main_role,
        "difficulty": champion.difficulty.difficulty,
        "source_url": str(skin.img_url),
        "content": "\n".join(parts),
        "meta_json": {
            "img_url": str(skin.img_url),
            "img_path": skin.img_path,
            "skin_description": skin.skin_description,
        },
    }


# async def skin_to_doc(champion: dict[str, Any], skin: dict[str, Any]) -> dict[str, Any]:
#     skin_verified = ChampionSkin.model_validate(skin)
#     skin_description = await ensure_skin_description(
#         skin_verified, skin_verified.champion_name
#     )

#     parts = [
#         f"Champion: {champion['name']}",
#         f"Main role: {champion['main_role']}",
#         f"Skin: {skin['skin_name']}",
#     ]

#     if skin.get("skin_description"):
#         parts.append(f"Skin description: {skin_description}")

#     content = "\n".join(parts)

#     return {
#         "doc_type": "champion_skin",
#         "champion_name": champion["name"],
#         "skin_name": skin["skin_name"],
#         "main_role": champion["main_role"],
#         "difficulty": champion["difficulty"]["difficulty"],
#         "source_url": str(skin["img_url"]),
#         "content": content,
#         "metadata": {
#             "champion_name": champion["name"],
#             "img_url": str(skin["img_url"]),
#             "img_path": skin.get("img_path"),
#             "skin_description": skin.get("skin_description"),
#         },
#     }


#############
async def build_rag_docs(
    # champions: list[dict[str, Any]],
    champions: list[Champion],
    # skin_description_agent: Agent[None, SkinDescriptionOutput] | None,
    skin_imgs_dir: Path,
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    # total_skins = sum(len(c.skins) for c in champions)
    total_skins = len(champions)

    with tqdm(total=total_skins, desc="Ingesting champion and skin data") as pbar:
        for champion in champions:
            pbar.set_postfix({"champion": champion.name})

            docs.append(champion_to_profile_doc(champion))
            for skin in champion.skins:
                # pbar.set_postfix({"champion": champion.name, "skin": skin.skin_name})

                docs.append(
                    await skin_to_doc(
                        champion,
                        skin,
                        # skin_description_agent,
                        # skin_imgs_dir=skin_imgs_dir,
                    )
                )
            pbar.update(1)

    return docs


def load_jsonl(path: str | Path) -> list[Champion]:
    # rows: list[dict[str, Any]] = []
    rows: list[Champion] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(Champion.model_validate_json(line))
    return rows


def save_champions_jsonl(champions: list[Champion], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for champion in champions:
            f.write(champion.model_dump_json())
            f.write("\n")
