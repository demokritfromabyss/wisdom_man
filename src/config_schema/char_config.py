from typing import List
from pydantic import BaseModel, Field


class CharacterNames(BaseModel):
    """
    Модель для хранения информации о именах персонажа в разных платформах
    """
    Default: str
    Minecraft: str
    Aliases: List[str] = Field(default_factory=list)


class CharacterSocials(BaseModel):
    """
    Модель для хранения информации о социальных сетях персонажа
    """
    TwitchChannelName: str


class CharacterConfig(BaseModel):
    """
    Модель для десериализации данных о персонаже
    """
    Names: CharacterNames
    Socials: CharacterSocials
    GodUsers: List[str] = Field(default_factory=list)
    # Character never attacks players (refuses chase/strike actions)
    Pacifist: bool = False


class CharacterConfigManager:
    """
    Класс для управления конфигурацией персонажа
    """
    def __init__(self, config_data: dict):
        self.config = CharacterConfig(**config_data)
    
    def get_config(self) -> CharacterConfig:
        return self.config
