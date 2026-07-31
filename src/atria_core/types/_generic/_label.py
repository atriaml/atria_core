class Label:
    def __init__(self, value: int, name: str) -> None:
        self.value = value
        self.name = name

    def __repr__(self) -> str:
        return f"Label(value={self.value}, name={self.name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Label):
            return NotImplemented
        return self.value == other.value and self.name == other.name
