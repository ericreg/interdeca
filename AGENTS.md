# InterDeCA core

- Keep imports at the top of each Python file.
- Never use `from __future__ import annotations`. Use modern Python typing, including `typing.Self` for methods that return their own type.
- Keep core operations independent of Slicer, Qt, and execution managers.
- Use static methods on the domain namespaces for core operations.
- Use `typing.NamedTuple` for internal data records. Keep dictionaries for actual key/value lookup tables, not objects with fixed fields.
- Use Pydantic models in `serde.py` for boundary validation, serialization, and deserialization. Numerical algorithms consume internal records.
- Document public contracts and mathematical steps for an undergraduate math/CS audience.
- Throughout the package, add plain-English line comments above logically related statements explaining why the group is needed. Cover assumptions, choices, and invariants; do not merely restate the operations or comment every trivial line. Keep comments accurate when code changes.
- Preserve mesh correspondence: vertex order, triangle order, and per-corner UVs matter.
- The original implementation in `../3d_color_package/` is reference material; do not modify it.
- Tests are deferred for this initial implementation at the user's request.
