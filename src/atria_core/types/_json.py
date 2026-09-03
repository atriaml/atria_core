"""JSON-safe primitive types, shared by data-instance metadata and module configs."""

JSON_PRIMITIVES = (str, int, float, bool, type(None))

JSONPrimitive = str | int | float | bool | None
ParamDict = dict[str, JSONPrimitive]
ParamList = list[JSONPrimitive]
