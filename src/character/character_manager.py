from typing import Dict, Optional, Tuple
import os
from config_schema.char_config import CharacterConfig, CharacterConfigManager, CharacterNames, CharacterSocials
from utils.yaml_utils import YamlReader
from config_schema.settings import Settings
from data_schema.structure_templates import REPO_RESOURCE_PATH


class CharacterManager:
    """
    Класс для управления персонажем: чтение конфига и промптов
    """

    @staticmethod
    def get_main_character():
        """Считывает конфиг и возвращает менеджера для основного персонажа (указан в конфиге)"""
        
        character = CharacterManager.get_cached_instance(Settings.get().main_character.name)
        return character

    # Словарь для кэширования уже созданных экземпляров CharacterManager
    _cache: Dict[str, 'CharacterManager'] = {}

    def __init__(self, character_name: str, force_new: bool = False):
        self.character_name: str = character_name
        self.base_dir: str = os.path.join(REPO_RESOURCE_PATH, "Characters", self.character_name)
        self.voice_refs_dir: str = os.path.join(self.base_dir, "Voice")
        self.voice_refs_manifest_path: str = os.path.join(self.voice_refs_dir, "refs_manifest.yml")
        self.config_manager: Optional[CharacterConfigManager] = None

        # Если мы не должны создавать новый экземпляр и такой уже есть в кэше, возвращаем его
        if not force_new and character_name in CharacterManager._cache:
            cached_instance = CharacterManager._cache[character_name]
            # Копируем все атрибуты из кэшированного экземпляра
            self.__dict__.update(cached_instance.__dict__)
            return

        # Загружаем конфигурацию
        self._load_character_config()

        CharacterManager._cache[character_name] = self
    
    def _load_character_config(self):
        """Загружает конфигурацию персонажа из YAML файла"""
        config_path = os.path.join(self.base_dir, "char_config.yml")
        
        try:
            config_data = YamlReader.read_yaml(config_path)
            self.config_manager = CharacterConfigManager(config_data)
        except Exception as e:
            raise Exception(f"Ошибка загрузки конфигурации персонажа {self.character_name}: {str(e)}")
    
    def get_character_config(self) -> CharacterConfig:
        """Возвращает конфигурацию персонажа"""
        if self.config_manager is None:
            raise Exception("Конфигурация персонажа не загружена")
        return self.config_manager.get_config()
    
    def get_names(self) -> CharacterNames:
        """Возвращает информацию об именах персонажа в разных платформах"""
        config = self.get_character_config()
        return config.Names
    
    def get_socials(self) -> CharacterSocials:
        """Возвращает информацию о социальных сетях персонажа"""
        config = self.get_character_config()
        return config.Socials
    
    def is_character_alias(self, username: str) -> bool:
        """
        Проверяет является ли указанный username псевдонимом персонажа
        
        Args:
            username (str): Проверяемый username
        Returns:
            bool: True, если username является псевдонимом персонажа
        """
        config = self.get_character_config()
        return username in config.Names.Aliases


    def is_user_godmode(self, username: str) -> bool:
        """
        Проверяет присутствует ли указанный username в списке супер-пользователей
        
        Args:
            username (str): Проверяемый username
        Returns:
            bool: True, если username находится в списке супер-пользователей
        """
        config = self.get_character_config()
        return username in config.GodUsers

    def read_character_prompt(self, personality_name: str = "streamer_abstract") -> str:
        """Загружает промпт для персонажа по имени персональности"""
        personalities_path = os.path.join(self.base_dir, "personalities.yml")
        try:
            personalities = YamlReader.read_yaml(personalities_path)
            return personalities.get(personality_name, "")
        except Exception as e:
            raise Exception(f"Ошибка загрузки промпта персонажа {self.character_name}: {str(e)}")
    
    def read_character_ideas(self) -> Dict:
        """Загружает идеи персонажа"""
        ideas_path = os.path.join(self.base_dir, "ideas.yml")
        try:
            return YamlReader.read_yaml(ideas_path)
        except Exception as e:
            raise Exception(f"Ошибка загрузки идей персонажа {self.character_name}: {str(e)}")
    
    def read_character_user_status_tags(self) -> Dict:
        """Загружает теги статусов пользователей для персонажа"""
        status_tags_path = os.path.join(self.base_dir, "user_status_tags.yml")
        try:
            return YamlReader.read_yaml(status_tags_path)
        except Exception as e:
            raise Exception(f"Ошибка загрузки тегов статусов пользователей персонажа {self.character_name}: {str(e)}")

    def resolve_voice_ref(self, emotion: str) -> Tuple[str, str]:
        """
        Разрешает ссылку на голосовой файл по эмоции.

        Загружает манифест голосовых ссылок и возвращает текст и путь к WAV-файлу
        для указанной эмоции. Если эмоция не найдена, возвращается нейтральная эмоция.

        Args:
            emotion (str): Название эмоции (например, "happy", "sad", "angry")

        Returns:
            Tuple[str, str]: Кортеж из текста и пути к WAV-файлу (text, wav_file_path)

        Example:
            >>> text, wav_path = character_manager.resolve_voice_ref("happy")
            >>> print(f"Text: {text}, WAV: {wav_path}")
        """
        voice_refs = YamlReader.read_yaml(self.voice_refs_manifest_path)
        
        ref_data = voice_refs.get(emotion, voice_refs["neutral"])
        ref_text = ref_data.get("text", "")
        ref_rel_path = ref_data["wav"]
        wav_file = os.path.join(self.voice_refs_dir, ref_rel_path + ".wav")

        return (ref_text, wav_file)

    @classmethod
    def get_cached_instance(cls, character_name: str) -> 'CharacterManager':
        """Получить кэшированный экземпляр или создать новый"""
        if character_name not in cls._cache:
            return cls(character_name)
        return cls._cache[character_name]

    @classmethod
    def clear_cache(cls, character_name: Optional[str] = None):
        """Очистить кэш для конкретного персонажа или все"""
        if character_name:
            cls._cache.pop(character_name, None)
        else:
            cls._cache.clear()


if __name__ == "__main__":
    from tests.framework.test_helper import TestHelper

    TestHelper.announce_testing(__file__)

    char_name = Settings.get().main_character.name

    print(f"Кэш персонажей: {CharacterManager._cache}")

    character = CharacterManager.get_cached_instance(char_name)
    print(f"Кэш персонажей: {CharacterManager._cache}")

    print(f"Базовый промпт для персонажа {character.character_name}:")
    print(character.read_character_prompt())

    char2 = CharacterManager.get_cached_instance(char_name)
    print(f"Кэш персонажей: {CharacterManager._cache}")

    TestHelper.test(char2).should_be(character, "Проверка кэша")
    TestHelper.test(char2.character_name).should_equal(character.character_name, "Проверка кэша")
