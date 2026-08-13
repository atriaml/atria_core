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

## Naming

- Use descriptive, intention-revealing names. Avoid abbreviations.
- Functions: `verb_noun` – `calculate_total`, `fetch_user`, `validate_email`.
- Booleans: `is_`, `has_`, `should_` – `is_valid`, `has_permission`.
- Constants: `UPPER_SNAKE_CASE`.
- Avoid single-letter variables except in trivial comprehensions or lambdas.

## Type Hints

- Add type hints to all function signatures (params and return).
- Use `| None` instead of `Optional`.
- Use built-in generics (`list[str]`, `dict[str, int]`) not `typing.List`, `typing.Dict`.
- Mostly you should just prefer to use ruff for formatting which should automatically handle it 

```python
def get_users(active_only: bool = True) -> list[User]:
    ...
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

- Use `logging` instead of 
- For this you should prefer always `get_logger` from atria.logging

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

```python
def calculate_tax(amount: float, rate: float = 0.2) -> float:
    """Calculate tax for the given amount at the specified rate."""
    return amount * rate
```