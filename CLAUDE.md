# Claude Code – Python Best Practices

## Function Arguments

- Always pass arguments as keyword arguments (kwargs), never positional.
- Exception: single-argument functions where meaning is obvious (e.g., `len(items)`).

```python
# Good
send_email(to="alice@example.com", subject="Hello", body="Hi there")

# Bad
send_email("alice@example.com", "Hello", "Hi there")
```

## Single Responsibility Principle

- Every function does exactly one thing. If you need the word "and" to describe what it does, split it.
- Every class owns one concept. If it's growing a second job, extract a new class.
- Every module groups one cohesive set of functionality.

## Constants and Helpers Belong in Dedicated Files

- Never define environment variables, constants, or config values at the top of a file that consumes them. They go in a dedicated constants or config module (e.g., `constants.py`, `config.py`).
- Never scatter helper/utility functions across random files. They belong in a shared `helpers.py` or `utils.py` module — one per logical domain, not one per file that happens to need a helper.
- If a constant or helper is used by multiple files, it must live in a shared module. No duplicating definitions.

```python
# Good – constants.py
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
API_BASE_URL = os.environ["API_BASE_URL"]

# Good – helpers.py
def format_timestamp(dt: datetime) -> str:
    """Format a datetime as ISO 8601."""
    return dt.isoformat()

# Good – service.py (imports from shared modules)
from myapp.constants import MAX_RETRIES, DEFAULT_TIMEOUT
from myapp.helpers import format_timestamp

# Bad – service.py (everything dumped at the top of the consumer)
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
API_BASE_URL = os.environ["API_BASE_URL"]

def format_timestamp(dt: datetime) -> str:
    return dt.isoformat()

def do_actual_work():
    ...
```

## Runtime Config Comes from Environment Variables, Not Class Params

- Values like API keys, base URLs, model names, and connection strings are runtime configuration — they change per environment (dev, staging, prod). They must be loaded from environment variables, not passed as config class parameters.
- A config class should read these from `os.environ` (or a settings module that does), not expect them to be supplied by the caller.
- This prevents secrets and environment-specific values from leaking into code, YAML files, or constructor calls.

```python
# Good – loaded from env vars
@dataclass(frozen=True)
class LLMConfig:
    model_name: str = os.environ.get("LLM_MODEL_NAME", "gpt-4o")
    base_url: str = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    api_key: str = os.environ["LLM_API_KEY"]
    temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0.0"))

# Bad – caller has to know and pass runtime values
@dataclass(frozen=True)
class LLMConfig:
    model_name: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0

# Bad – now every call site looks like this
config = LLMConfig(
    model_name="gpt-4o",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
)
```

## Lazy Imports in Config and Build Modules

- If a config or factory module gets imported widely (e.g., `config.py`, `settings.py`, `app_factory.py`), any heavy or build-specific imports inside it must go **inside the function**, not at the top of the file.
- Top-level imports in a widely-imported module get executed every time anything imports it — even if the expensive code path is never called. This kills startup and test import times.
- Rule of thumb: if the module is imported for its constants or simple config, but also has a function that builds/initializes something heavy, that function's imports go inside the function body.

```python
# Good – heavy imports inside the build function
# config.py

DATABASE_URL = os.environ["DATABASE_URL"]
DEBUG = os.environ.get("DEBUG", "false") == "true"

def build_engine() -> Engine:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import QueuePool

    return create_engine(DATABASE_URL, poolclass=QueuePool)

# Bad – heavy imports at top level, paid on every import of config
# config.py

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

DATABASE_URL = os.environ["DATABASE_URL"]
DEBUG = os.environ.get("DEBUG", "false") == "true"

def build_engine() -> Engine:
    return create_engine(DATABASE_URL, poolclass=QueuePool)
```

## Separate Configs from the Modules They Configure

- Config classes and the modules/objects they configure must live in separate files. Never put the config and the thing it configures in the same file.
- **A config only describes params. It never builds anything.** No `build_module()`, no factory methods, no imports of the thing it configures. A config is pure data plus validation, so it can be passed around, serialized, inspected, and diffed without pulling in heavy dependencies.
- **The module takes the config, not the other way round.** The dependency points one way: module → config. A config that knows how to construct its module inverts that, drags the heavy import back into the lightweight file, and makes the config impossible to use for inspection alone.
- Every module names its config class in a `__config__` class attribute and accepts `config` as a **keyword-only** argument defaulting to `None`. When it is `None`, the module constructs `__config__()` itself, so the common case needs no config at all.
- Validate the config type at the top of `__init__`. A mismatched config must fail there with a message naming both classes, not later as an `AttributeError` on a missing field.
- The public entry point is a plain, importable factory function with a concrete return type. Callers get exact types from an ordinary import — no registry lookup, no string keys, no `Any`.
- Heavy imports live in the module file, or inside the function that needs them (see the lazy-import rule).

```python
# Good – config is pure data, module takes it, factory is the entry point

# configs/openai_model_config.py (lightweight: safe to import anywhere)
@pydantic_dataclass(frozen=True)
class OpenAICompatibleModelConfig(ModelConfig):
    """Config for an OpenAI-compatible chat model endpoint."""

    model_name: str = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
    base_url: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key: str = os.environ.get("OPENAI_API_KEY", "")
    temperature: float = float(os.environ.get("OPENAI_TEMPERATURE", "0.0"))

# models/openai.py (heavy imports live here)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

class OpenAIChatModule(ChatModule[OpenAICompatibleModelConfig]):
    """Chat model backed by an OpenAI-compatible endpoint."""

    __config__ = OpenAICompatibleModelConfig

    def __init__(self, *, config: OpenAICompatibleModelConfig | None = None) -> None:
        """Build the model.

        Args:
            config: Endpoint params. Defaults to `__config__()`.

        Raises:
            TypeError: If `config` is not an OpenAICompatibleModelConfig.
        """
        super().__init__(config=config)
        provider = OpenAIProvider(base_url=self.config.base_url, api_key=self.config.api_key)
        self._model = OpenAIChatModel(
            self.config.model_name,
            provider=provider,
            settings={"temperature": self.config.temperature},
        )

def openai_compatible(
    model_name: str = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o"),
) -> OpenAIChatModule:
    """Build a chat module for an OpenAI-compatible endpoint."""
    return OpenAIChatModule(config=OpenAICompatibleModelConfig(model_name=model_name))

# Callers – exact types, plain imports
model = openai_compatible()                                   # -> OpenAIChatModule
model = OpenAIChatModule()                                    # default config
model = OpenAIChatModule(config=OpenAICompatibleModelConfig(temperature=0.7))

# Bad – config builds its own module: heavy import dragged into the config file,
#       dependency inverted, config no longer safe to import for inspection
@pydantic_dataclass(frozen=True)
class OpenAICompatibleModelConfig(ModelConfig):
    model_name: str
    temperature: float = 0.0

    def build_module(self) -> OpenAIChatModel:
        from atria_llm.models.openai import build_openai_model
        return build_openai_model(config=self)

# Bad – config and module in one file, env vars as caller-supplied params,
#       heavy imports at top level, conversation-referencing docstring
from pydantic_ai.models.openai import OpenAIChatModel

@pydantic_dataclass(frozen=True)
class OpenAICompatibleModelConfig(ModelConfig):
    """Config for any OpenAI-compatible endpoint -- real OpenAI, or a
    local vllm serve HTTP server. Pure config plumbing on top of
    pydantic-ai's own OpenAIChatModel/OpenAIProvider; no custom HTTP
    client code."""

    model_name: str
    api_key: str | None = None
```

## Registries: Prefer Plain Imports

- Do not build name→class registries with decorators and string keys. Python's import system already maps names to objects, and it does so with exact static types.
- Expose each buildable thing as an importable factory function. `atria_datasets.mnist()` returns a known concrete type; `create("mnist")` can only ever return the base class.
- When a name genuinely arrives as data (a CLI flag, a config file), resolve it as a dotted import path with `importlib` rather than maintaining a parallel registry dict.

```python
# Good – importable factory, exact type
from atria_datasets import mnist
dataset = mnist(config_name="mnist")          # -> MNIST

# Good – name as data, resolved by import path
module_name, _, attribute = "atria_datasets.mnist".rpartition(".")
factory = getattr(importlib.import_module(module_name), attribute)

# Bad – string registry: returns the base class, params unchecked
dataset = create_dataset("mnist", config_name="mnist")   # -> Dataset[Any, Any]
```

## Keep the Directory Structure Simple

- If a directory contains only a single file, it does not need to be a directory. Just put the file in the parent. A folder exists to group multiple related files — one file is not a group.
- Only create directories that represent a real, distinct domain or bounded context. Not every noun needs its own package.
- Don't nest directories more than 2–3 levels deep unless the project genuinely demands it.
- Helper functions go in the module they're closest to, or in a shared `utils`/`helpers` module — not in a new `helpers/` directory with subdirectories.
- When in doubt, fewer directories is better. A flat structure with clear file names beats a deep tree where you have to hunt through five folders to find anything.

```
# Good – flat and obvious
myapp/
    constants.py
    config.py
    helpers.py
    models.py
    services.py
    routes.py
    tests/
        test_services.py
        test_models.py

# Bad – single file in a folder for no reason
myapp/
    services/
        user_service.py      <-- just put this in myapp/user_service.py
    config/
        settings.py           <-- just put this in myapp/settings.py

# Bad – over-engineered directory hell
myapp/
    core/
        base/
            abstract_service.py
        config/
            settings/
                env.py
    helpers/
        string_helpers/
            formatters.py
        date_helpers/
            parsers.py
    services/
        user/
            user_service.py
```

## Naming

- Use descriptive, intention-revealing names. Avoid abbreviations.
- Functions: `verb_noun` – `calculate_total`, `fetch_user`, `validate_email`.
- Booleans: `is_`, `has_`, `should_` – `is_valid`, `has_permission`.
- Constants: `UPPER_SNAKE_CASE`.
- Avoid single-letter variables except in trivial comprehensions or lambdas.

## Type Hints – Strict Typing Everywhere

- Add type hints to **every** function signature (params and return), **every** variable that isn't immediately obvious, and **every** class attribute.
- Use `| None` instead of `Optional`.
- Use built-in generics (`list[str]`, `dict[str, int]`) not `typing.List`, `typing.Dict`.
- Mostly you should just prefer to use ruff for formatting which should automatically handle it.
- No duck typing. Ever. Don't rely on "it has a `.read()` method so it's file-like." Use the explicit type. Every object must have a known, declared type.
- Never use `Any` as a type. If you think you need `Any`, you need to define a proper type, union, or dataclass instead.
- Never use `getattr()` or `setattr()`. Access attributes directly through typed references. If you feel you need dynamic attribute access, the design is wrong — refactor to use a dataclass or explicit method.
- Never use `hasattr()` to check for capabilities. Use `isinstance()` checks against a typed class.
- Never use `TypedDict`. Always use dataclasses or Pydantic models for structured data — no exceptions.
- Never define `Protocol` classes unless explicitly asked. Use concrete types, base classes, or ABCs.
- Any time you're reaching for a plain `dict` to represent structured data, use a dataclass instead.

```python
# Good – everything explicitly typed
def get_users(active_only: bool = True) -> list[User]:
    ...

user_cache: dict[str, User] = {}
result: UserResponse = fetch_data(endpoint="/users")

# Good – dataclass for structured data
@dataclass
class UserResponse:
    users: list[User]
    total_count: int
    has_more: bool

# Bad – TypedDict
class UserResponse(TypedDict):
    users: list[User]
    total_count: int

# Bad – raw dict
def get_response() -> dict[str, Any]:
    return {"users": [], "total_count": 0}

# Bad – duck typing, no types, dynamic access
def process_stream(source):
    if hasattr(source, 'read'):
        return source.read()

# Bad – getattr/setattr
value = getattr(obj, "name", None)
setattr(obj, "name", "alice")

# Good – direct typed access
value: str = obj.name
obj.name = "alice"
```

## Mypy – Strict Mode, Zero Errors

- All code must pass `mypy --strict` with zero errors. No exceptions, no `# type: ignore` unless there is a genuine mypy bug and a comment explaining why.
- Run mypy before committing. If mypy complains, fix the code — don't loosen the config or suppress the error.
- Never use `# type: ignore` to shut mypy up. If the types are wrong, fix them. If mypy is wrong about a third-party library, add a stub or a minimal cast with a comment explaining the library gap.
- Keep `disallow_untyped_defs`, `disallow_any_explicit`, and `warn_return_any` enabled.
- If a third-party library has no stubs, don't set `ignore_missing_imports` globally — add a per-module override in `mypy.ini` for that specific library only.

```ini
# mypy.ini
[mypy]
strict = True
disallow_untyped_defs = True
disallow_any_explicit = True
warn_return_any = True
warn_unused_ignores = True

# Only if a specific library has no stubs
[mypy-some_untyped_library.*]
ignore_missing_imports = True
```

## Error Handling

- Catch specific exceptions, never bare `except:`.
- Fail fast – validate inputs at the top of a function.
- Use custom exception classes for domain errors.
- Don't use exceptions for control flow.

```python
# Good
except ValueError as e:
    logger.error("Invalid input: %s", e)

# Bad
except:
    pass
```

## Return Early

- Use early returns to eliminate nesting. Avoid deep `if/else` trees.

```python
# Good
def process(user: User) -> Result:
    if not user.is_active:
        return Result.inactive()
    if not user.has_permission:
        return Result.forbidden()
    return do_work(user=user)

# Bad
def process(user: User) -> Result:
    if user.is_active:
        if user.has_permission:
            return do_work(user=user)
        else:
            return Result.forbidden()
    else:
        return Result.inactive()
```

## Data Handling

- Use dataclasses or Pydantic models instead of raw dicts for structured data.
- Keep data transformations as pure functions (no side effects).

```python
@dataclass
class User:
    name: str
    email: str
    is_active: bool = True
```

## Constants Over Magic Values

- No magic numbers or strings. Extract them into named constants.

```python
# Good
MAX_RETRIES = 3
for attempt in range(MAX_RETRIES):
    ...

# Bad
for attempt in range(3):
    ...
```

## Logging Over Print

- Use `logging` instead of `print`.
- For this you should prefer always `get_logger` from atria.logging.

## Testing

- Write tests for public interfaces, not implementation details.
- One assertion per test when practical.
- Use descriptive test names: `test_expired_token_returns_401`.
- Arrange-Act-Assert structure in every test.

## Don't Repeat Yourself

- If you copy-paste code, extract it into a shared function.
- But don't over-abstract – two occurrences are fine; three means extract.

## Docstrings

- Add a docstring to every public function and class.
- Keep them concise: one-liner for simple functions, summary + args/returns for complex ones.
- Write docstrings as if you have zero context about any prior conversation, task, or decision. A docstring describes what the function or class does — not why we built it, what we discussed, or what problem we were solving in chat.
- A developer reading the docstring for the first time, with no awareness of our conversation, should fully understand the purpose from the docstring alone.
- Never reference refactors, prior implementations, user requests, or conversation context in any docstring or comment.
- If you encounter a bad, misleading, conversation-referencing, or outdated docstring in existing code while working on it, fix or remove it. Don't leave garbage docstrings in place just because they were already there.

```python
# Good – standalone, self-explanatory
def calculate_tax(amount: float, rate: float = 0.2) -> float:
    """Calculate tax for the given amount at the specified rate."""
    return amount * rate

# Good
class InvoiceService:
    """Handles invoice creation, validation, and delivery."""

# Bad – references conversation context or decisions
def calculate_tax(amount: float, rate: float = 0.2) -> float:
    """Calculate tax as discussed. Moved from utils to here per refactor."""
    return amount * rate

# Bad
class InvoiceService:
    """Refactored from the old OrderProcessor to separate concerns
    as we agreed. Handles invoices now instead of orders."""
```

## Duplication vs. Abstraction

- Prefer human readability over abstraction. Similar-looking code in different, conceptually
  independent places (e.g. per-backend chat-message builders, per-task helper methods, per-task
  input/output/metadata models) should generally stay local and repeated rather than merged into
  a shared helper or base class, even when the current implementations are identical — they
  represent independent concepts that may diverge, and a shared abstraction can quickly turn into
  a maintenance/readability burden ("abstraction nightmare") once they do.
- Two or three occurrences of a duplicated block is not automatically a signal to extract a
  shared function/base class if the callers are conceptually distinct entities (e.g. two
  different model backends, two different eval tasks). Judge by whether the concepts are truly
  the same thing, not by textual similarity alone.
- Shared *default parameter values* across sibling config classes (e.g. `max_tokens`,
  `temperature` defaults across model configs) belong on the shared base config class as
  inherited fields — not as standalone named module-level constants.
- Don't invent parallel identification schemes (e.g. string-literal "system" tags) to distinguish
  between a fixed set of known classes when `isinstance()` against the actual classes does the
  job directly.

## Script Output

- Scripts that produce run artifacts (eval reports, metrics, etc.) must take a general
  `output_dir` input rather than writing uniquely-named files per experiment. Uniquely-named
  output files tend to get committed to git by accident since generic `.gitignore` patterns
  don't anticipate every name; a single configurable output directory (itself gitignored) avoids
  that class of mistake entirely.