import asyncio
from concurrent.futures import TimeoutError
import aiohttp
from rich.console import Console

# Initialize rich console for colored output
console = Console()

# Constants
DEFAULT_TIMEOUT = 60  # seconds
MAX_RETRIES = 1

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        console.print(f"[blue]OpenRouterClient initialized with API key: {api_key[:10]}...[/blue]")
        
    async def generate(self, model: str, messages: list, timeout: int = DEFAULT_TIMEOUT) -> str:
        console.print(f"[cyan]Generating response for model: {model}[/cyan]")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/your-username/your-repo",
            "X-Title": "LLM Instruction Following Test Suite",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            # "temperature": 0.2,
            # "max_tokens": 1000
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                async with aiohttp.ClientSession() as session:
                    console.print("[cyan]Making API request...[/cyan]")
                    async with session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=timeout
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        console.print("[green]API response received[/green]")
                        return data['choices'][0]['message']['content']
            except TimeoutError:
                if attempt == MAX_RETRIES - 1:
                    raise
                console.print(f"[yellow]Timeout occurred. Retrying ({attempt + 2}/{MAX_RETRIES})[/yellow]")
                await asyncio.sleep(1)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    console.print(f"[red]Error in generate: {str(e)}[/red]")
                    raise
                console.print(f"[yellow]Error occurred. Retrying ({attempt + 2}/{MAX_RETRIES})[/yellow]")
                await asyncio.sleep(1)
