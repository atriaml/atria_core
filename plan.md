# Design Principles: Refactoring Guide (portable reference)

## Context

Over this session, `atria_core.registry`, `atria_core.types`, and parts of `atria_core.datasets`/`extractors`/`visualizers`/`serialization` were refactored/ported from an older, more tangled codebase (`atria_registry`, `atria_datasets`). Several recurring design decisions emerged — not as one-off fixes, but as a repeatable way of judging "is this the right shape?" The user wants those decisions distilled into a **standalone, project-agnostic principles document**, separate from any concrete task list, to carry into future refactoring work on other parts of the project (starting with `atria_core.datasets` Phase 2, but written generically enough for anything).

This document is the deliverable itself — not a step-by-step implementation plan. It should be dropped into a new chat as a reference the assistant reads before proposing a design.

## The document

### 1. Single Responsibility Principle
Each class does exactly one job; when a class both builds *and* caches *and* transforms, that's a signal to split it.
- **Concrete precedent**: `atria_datasets.Dataset`/`CachedDataset` conflate downloading, raw iteration, disk caching (msgpack/Delta Lake), and transform application into two classes with bidirectional coupling (`Dataset.cache()` calls `CachedDataset.save_snapshot()`; `CachedDataset` reaches back into `_dataset_builders` for transform classes). The identified fix (deferred to a later phase, not yet done): split into independent `Downloader`, `Cacher`, `Storer`, `Transformer` components orchestrated around a `Dataset` that only builds.
- **Test when refactoring**: can you describe the class in one sentence without "and"? If not, find the seam.

### 2. Transformations are always applied from the outside
Data/model classes hold state; they do not know how to transform themselves. Transform logic lives in free functions or standalone transformer classes that take an instance and return a new one.
- **Concrete precedent**: `PreprocessTransform`/`atria_core.transforms.functional` — resize, bbox ops, dataset preprocessing are all functions applied *to* immutable `@dataclass(frozen=True)` instances, never methods on `Image`, `BoundingBox`, etc.
- **Why it matters**: keeps the data model minimal and serializable, keeps transform logic independently testable/composable, and avoids classes accumulating unrelated "convenience" methods that only one call site needs.
- **Test when refactoring**: if you're tempted to add a `.normalize()` or `.augment()` method onto a data model, write it as a function taking that model instead.

### 3. Config → build pattern stays consistent
Every `ModuleConfig` subclass implements its own `build_module()` that directly constructs and returns whatever it configures. No stored `target: ClassVar[type]`, no base-class dispatch flags, no reflection-based instantiation, no "build-and-immediately-run" double duty.
- **Concrete precedent**: `atria_core.registry.ModuleConfig`/`ConfigurableModule` — replaced an older pattern (`DatasetConfig.build(**kwargs)`) that constructed an object *and* invoked ~15 runtime-only kwargs in one call. The fix: `build_module()` takes no arguments and only constructs; any runtime invocation (`.load(...)`, `.cache(...)`) is a separate explicit call at the use site.
- **Test when refactoring**: does the config's build step take any parameter that isn't itself config data? If yes, that parameter belongs to a separate method call, not folded into construction.

### 4. Bias toward simplicity over complexity
Prefer the plainer shape once added complexity isn't earning its keep. Collapsing an abstraction is as valid a refactor as introducing one.
- **Concrete precedent**: `DocumentInstance.document`-wrapping field was collapsed into direct subclasses `SinglePageDocumentInstance`/`MultiPageDocumentInstance` once it was clear nothing downstream ever treated the wrapper polymorphically. Similarly, `Image`/`PdfPage` share a `load()`/`require_content()` shape via structural duck-typing rather than a shared abstract base class — polymorphism without a formal hierarchy.
- **Test when refactoring**: for every abstraction (wrapper, base class, generic parameter), ask "does any real caller need this to vary?" If not, inline/collapse it.

### 5. Complex dunder/reflection mechanics = design smell
Needing `__new__`, `__setattr__` overrides, metaclasses, or reflection-based magic to make something work is a sign the design itself is wrong, not a problem to engineer around.
- **Rationale established this session**: pydantic dataclasses are used narrowly, only where `atria_core.registry` needs config validation; everything else (`BaseDataModel`, `BaseDataInstance`, and all of `atria_core.types`) deliberately stays on plain stdlib `@dataclass(frozen=True)` to avoid metaclass machinery entirely.
- **Test when refactoring**: if implementing a feature requires overriding object construction/attribute-setting internals to route around `frozen=True`, or requires `hasattr`-based capability probing (as `SplitIterator`'s generic wrapper does against an arbitrary `base_iterator`) — stop and redesign the type hierarchy or the call contract instead. The fix is almost always: make the concrete class implement the behavior directly, rather than a generic wrapper that inspects what it's given at runtime.

### 6. Simple user-facing APIs for registry-based extension points
When users define their own types or register their own datasets/modules, the surface they touch should be minimal and obvious — a decorator plus a plain class — while the underlying behavior (validation, uniqueness, build dispatch) stays intact.
- **Concrete precedent**: `Registry.group("datasets")` + `@datasets.register("name")` on a `ModuleConfig` subclass with a `build_module()` method is the entire user-facing contract; no required base-class ceremony beyond implementing `build_module()`.
- **Test when refactoring**: could a new user write a working extension by copy-pasting one example and changing names, without reading the registry's internals? If they need to understand dispatch internals to succeed, the API is leaking complexity it shouldn't.

## How to use this document
For each area being refactored: identify the God-object/mixed-responsibility class (principle 1), check whether transform-like logic has crept into a data model (principle 2), check whether any config/build path takes runtime args mixed with construction args (principle 3), look for abstractions with only one real implementation (principle 4), grep for dunder overrides or `hasattr`/`getattr` capability probing (principle 5), and sanity-check the extension-point API against "could a new user copy one example and succeed" (principle 6).

No verification section — this document is reference material, not an implementation task.
