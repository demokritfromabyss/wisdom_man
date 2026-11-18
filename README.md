# Мудрец (Wisdom Man)

> «А вы знаете, что фундамент закладывают один раз, а потом всю жизнь на него жалуются? Это ещё Гегель заметил. Или прораб.»

ИИ-персонаж для Minecraft, трек 1 курса [AI Persona Lab](https://course.personalabs.ru). Псевдоинтеллигент: неточно цитирует философов, в каждом блоке видит метафору бытия и время от времени спрашивает «А вы знаете?..». Играет на публичном сервере, ходит сам, говорит голосом и дублирует сказанное в чат.

Собран на харнесе [NetTyan](https://github.com/3ndetz/NetTyan) — это его форк с персонажем и правками под игру. Подробный журнал работы — [docs/wisdom-man/progress.md](docs/wisdom-man/progress.md).

## Что умеет

- **Говорит своим голосом.** Qwen3-TTS локально на Mac (MLX) клонирует голос Мудреца, под каждую эмоцию свой образец. 2–5 с на фразу, при запущенной игре 6–11 с.
- **Ходит сам, по схеме курса** (вижу → иду → действую). В тишине смотрит, кто рядом и чем занято тело, и выбирает: подойти и заговорить, пойти в другое место или промолчать. Мод только исполняет. Одна команда движения за ход. Никого нет — уходит искать людей и помнит, где уже было пусто.
- **Пишет в чат то, что говорит голосом**, чтобы игроки видели реплики персонажа.
- **Слышит игроков** из голосового чата Minecraft (Parakeet) и **видит** окно игры через OBS.
- **Общается по ролям**, без привязки к никам: создатель, новичок, спорщик, грубиян. На практические вопросы отвечает правильно — ирония только в подаче.
- **Не зацикливается и не буянит**: не больше трёх ответов одному собеседнику в минуту, не повторяет одно и то же, ни на кого не нападает, не ходит дальше 300 блоков, пароль от сервера не вводит.

## Как устроен

| Часть | Чем сделано | Где |
|---|---|---|
| Мозг | Mistral (`mistral-medium-latest`, запасная `mistral-small-latest`) | `src/agent/multi/chat_agent.py` |
| Характер | промпт, роли собеседников, идеи для тихих моментов | [`resources/Characters/WisdomMan/`](resources/Characters/WisdomMan/) |
| Голос | Qwen3-TTS `1.7B-Base-6bit` на MLX, тот же API, что у Fish TTS | `qwen_tts_server.py` |
| Слух | Parakeet (sherpa-onnx) в Docker | `docker/stt/parakeet/` |
| Тело | мод [UnionClef](https://github.com/3ndetz/unionclef) (Alto Clef + Tungsten), py4j и MCP | `src/data_collectors/minebridge.py`, `src/agent/tools/minecraft_tools.py` |
| Зрение и запись | OBS: окно Minecraft + голос из VB-Cable | `src/utils/vision_helper.py` |

## Запуск

macOS, Apple Silicon. Нужны Docker, Python-окружение проекта (`.venv`), запущенный Minecraft с модом UnionClef.

```bash
docker compose up -d          # db, redis, parakeet

# голос: отдельное окружение для Qwen3-TTS (один раз)
uv venv --python 3.12 ../qwen3-tts/.venv && source ../qwen3-tts/.venv/bin/activate
uv pip install mlx-audio==0.6.3 gradio==6.30.0 soundfile scipy
QWEN_TTS_PORT=22231 python qwen_tts_server.py    # первый запуск качает модель ~1.5 ГБ

# агент
cd src && source ../.venv/bin/activate
python main.py -cof
```

Интерфейс: <http://127.0.0.1:22228>. Порт 22231 занимает и контейнер simple_tts — если он запущен, сначала `docker stop nettyan_simple_tts`.

Главное в `.env` (полный пример — [`.env.example`](.env.example)):

| Параметр | Значение |
|---|---|
| `NETTYAN_CHARACTER_NAME` | `"WisdomMan"` |
| `CORE_LLM_MODEL_NAME` | `mistral/mistral-medium-latest` |
| `TTS_ENDPOINT_URL` | `"http://127.0.0.1:22231/"` |
| `INITIATIVE_SEC` | `30` — тишина до тихого момента, `0` выключает |
| `VOICE_TO_CHAT` | `"TRUE"` — дублировать реплики в чат |
| `UNIONCLEF_MCP_TOKEN` | `mcpAuthToken` из `altoclef_settings.json` — мост сам поднимает упавший шлюз мода |

Остановка: гасить и главный процесс, и дочерние (мост, TTS, STT), иначе они остаются висеть.

## Известные проблемы

- Игровой аккаунт общий с создателем.
- Может выбрать точку в воде и утонуть по прибытии; маршруты мода иногда идут по обрывам.
- Голосом иногда повторяет одну и ту же фразу.
- Память о собеседниках (RAG) не включена — нужны локальные эмбеддинги.

Полный список и история правок — в [progress.md](docs/wisdom-man/progress.md). Точки восстановления — теги `restore/*`.

## Основа: NetTyan

Харнес — [NetTyan](https://github.com/3ndetz/NetTyan): обёртка над языковой моделью с распознаванием и синтезом речи, сбором сообщений с площадок, модерацией и аватаром. Его документация: <https://3ndetz.github.io/NetTyan/> и [`docs/help/`](docs/help/). Старый README харнеса — [README_OLD.md](README_OLD.md).

NetTyan вырос из [NeuroDeva](https://github.com/3ndetz/NeuroDeva) того же автора. Авторы харнеса — в [AUTHORS](AUTHORS).

## Лицензия

MIT, см. [LICENSE](LICENSE). Звуковые файлы в `resources/` лицензией не покрываются.
