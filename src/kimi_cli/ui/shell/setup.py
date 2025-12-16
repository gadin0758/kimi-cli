from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, NamedTuple

import aiohttp
from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts.choice_input import ChoiceInput
from pydantic import SecretStr

from kimi_cli.config import (
    LLMModel,
    LLMProvider,
    MoonshotFetchConfig,
    MoonshotSearchConfig,
    load_config,
    save_config,
)
from kimi_cli.ui.shell.console import console
from kimi_cli.ui.shell.metacmd import meta_command
from kimi_cli.utils.aiohttp import new_client_session

if TYPE_CHECKING:
    from kimi_cli.ui.shell import Shell


class _Platform(NamedTuple):
    id: str
    name: str
    base_url: str
    search_url: str | None = None
    fetch_url: str | None = None
    allowed_prefixes: list[str] | None = None


_PLATFORMS = [
    _Platform(
        id="kimi-for-coding",
        name="Kimi for Coding",
        base_url="https://api.kimi.com/coding/v1",
        search_url="https://api.kimi.com/coding/v1/search",
        fetch_url="https://api.kimi.com/coding/v1/fetch",
    ),
    _Platform(
        id="moonshot-cn",
        name="Moonshot AI 开放平台 (moonshot.cn)",
        base_url="https://api.moonshot.cn/v1",
        allowed_prefixes=["kimi-k2-"],
    ),
    _Platform(
        id="moonshot-ai",
        name="Moonshot AI Open Platform (moonshot.ai)",
        base_url="https://api.moonshot.ai/v1",
        allowed_prefixes=["kimi-k2-"],
    ),
    _Platform(
        id="local-ollama",
        name="Local Ollama (localhost:11434)",
        base_url="http://localhost:11434/v1",
    ),
]


@meta_command
async def setup(app: Shell, args: list[str]):
    """Setup Kimi CLI"""
    result = await _setup()
    if not result:
        # error message already printed
        return

    config = load_config()
    config.providers[result.platform.id] = LLMProvider(
        type="kimi",
        base_url=result.platform.base_url,
        api_key=result.api_key,
    )
    config.models[result.model_id] = LLMModel(
        provider=result.platform.id,
        model=result.model_id,
        max_context_size=result.max_context_size,
    )
    config.default_model = result.model_id

    if result.platform.search_url:
        config.services.moonshot_search = MoonshotSearchConfig(
            base_url=result.platform.search_url,
            api_key=result.api_key,
        )

    if result.platform.fetch_url:
        config.services.moonshot_fetch = MoonshotFetchConfig(
            base_url=result.platform.fetch_url,
            api_key=result.api_key,
        )

    save_config(config)
    console.print("[green]✓[/green] Kimi CLI has been setup! Reloading...")
    await asyncio.sleep(1)
    console.clear()

    from kimi_cli.cli import Reload

    raise Reload


class _SetupResult(NamedTuple):
    platform: _Platform
    api_key: SecretStr
    model_id: str
    max_context_size: int


async def _setup() -> _SetupResult | None:
    # select the API platform
    platform_name = await _prompt_choice(
        header="Select the API platform",
        choices=[platform.name for platform in _PLATFORMS],
    )
    if not platform_name:
        console.print("[red]No platform selected[/red]")
        return None

    platform = next(platform for platform in _PLATFORMS if platform.name == platform_name)

        # ===== Handle Local Ollama with auto-detection + fallback =====
    if platform.id == "local-ollama":
        ollama_models: list[str] = []
        base_url = "http://localhost:11434"

        # Try to fetch models from Ollama
        try:
            async with new_client_session() as session:
                async with session.get(f"{base_url}/api/tags", timeout=3) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        ollama_models = [m["name"] for m in data.get("models", [])]
                        console.print(f"[green]✓[/green] Found {len(ollama_models)} model(s) in Ollama.")
                    else:
                        console.print(f"[yellow]⚠[/yellow] Ollama returned status {resp.status}")
        except asyncio.TimeoutError:
            console.print("[yellow]⚠[/yellow] Timeout connecting to Ollama (is it running?)")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed to fetch models from Ollama: {type(e).__name__}")

        # Fallback to common models if none detected
        if not ollama_models:
            fallback_models = ["qwen:7b", "llama3", "phi3", "mistral", "gemma2:2b"]
            console.print("[dim]Using common model suggestions (pull with `ollama pull <model>`)[/dim]")
            ollama_models = fallback_models

        ollama_models.append("custom...")

        model_choice = await _prompt_choice(
            header="Select a local model",
            choices=ollama_models,
        )
        if not model_choice:
            return None

        if model_choice == "custom...":
            model_id = await _prompt_text("Enter custom model name (e.g., qwen:14b)")
            if not model_id:
                return None
            max_ctx = 32768
        else:
            model_id = model_choice
            ctx_map = {
                "qwen:7b": 32768, "qwen:14b": 32768, "qwen:32b": 32768,
                "llama3": 8192, "llama3:70b": 8192,
                "phi3": 4096,
                "mistral": 8192,
                "gemma2:2b": 4096, "gemma2:9b": 4096,
            }
            max_ctx = ctx_map.get(model_id, 8192)

        return _SetupResult(
            platform=platform,
            api_key=SecretStr("sk-local"),
            model_id=model_id,
            max_context_size=max_ctx,
        )

    # ===== 原有云平台逻辑（保持不变）=====

    # enter the API key
    api_key = await _prompt_text("Enter your API key", is_password=True)
    if not api_key:
        return None

    # list models
    models_url = f"{platform.base_url}/models"
    try:
        async with (
            new_client_session() as session,
            session.get(
                models_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
                raise_for_status=True,
            ) as response,
        ):
            resp_json = await response.json()
    except aiohttp.ClientError as e:
        console.print(f"[red]Failed to get models: {e}[/red]")
        return None

    model_dict = {model["id"]: model for model in resp_json["data"]}

    # select the model
    model_ids: list[str] = [model["id"] for model in resp_json["data"]]
    if platform.allowed_prefixes is not None:
        model_ids = [
            model_id
            for model_id in model_ids
            if model_id.startswith(tuple(platform.allowed_prefixes))
        ]

    if not model_ids:
        console.print("[red]No models available for the selected platform[/red]")
        return None

    model_id = await _prompt_choice(
        header="Select the model",
        choices=model_ids,
    )
    if not model_id:
        console.print("[red]No model selected[/red]")
        return None

    model = model_dict[model_id]

    return _SetupResult(
        platform=platform,
        api_key=SecretStr(api_key),
        model_id=model_id,
        max_context_size=model["context_length"],
    )


async def _prompt_choice(*, header: str, choices: list[str]) -> str | None:
    if not choices:
        return None

    try:
        return await ChoiceInput(
            message=header,
            options=[(choice, choice) for choice in choices],
            default=choices[0],
        ).prompt_async()
    except (EOFError, KeyboardInterrupt):
        return None


async def _prompt_text(prompt: str, *, is_password: bool = False) -> str | None:
    session = PromptSession[str]()
    try:
        return str(
            await session.prompt_async(
                f" {prompt}: ",
                is_password=is_password,
            )
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return None


@meta_command
def reload(app: Shell, args: list[str]):
    """Reload configuration"""
    from kimi_cli.cli import Reload

    raise Reload
