from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ChampionRole(StrEnum):
    TOP = "top"
    JUNGLE = "jungle"
    MIDDLE = "middle"
    BOTTOM = "bottom"
    SUPPORT = "support"


class ChampionDifficulty(BaseModel):
    difficulty: int = Field(le=3, ge=0)
    difficulty_name: str


class ChampionSkin(BaseModel):
    champion_name: str
    skin_name: str
    img_url: HttpUrl
    img_path: str | None = None
    skin_description: str | None = None


class ChampionAbility(BaseModel):
    name: str
    subtitle: str
    img_icon: HttpUrl
    img_icon_path: str | None = None
    # called description in official lol site
    description_high_level: str
    # TODO: can get it from opgg, on hold for now until i can parse it
    description_details: str | None


class ChampionAbilities(BaseModel):
    passive: ChampionAbility
    Q: ChampionAbility
    W: ChampionAbility
    E: ChampionAbility
    R: ChampionAbility


class Champion(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    name: str
    profile_card_image_url: HttpUrl
    profile_card_image_path: str | None = None
    official_lol_profile_details_website_url: HttpUrl
    # below in champion details page
    subtitle: str
    play_styles: list[str]
    difficulty: ChampionDifficulty
    # img_backdrop_url: HttpUrl
    description: str
    op_gg_link: HttpUrl
    op_gg_summary: str
    abilities: ChampionAbilities
    skins: list[ChampionSkin]
    main_role: ChampionRole
