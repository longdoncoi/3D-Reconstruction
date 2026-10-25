# ADR-0032: Plugin Unload Lifecycle and Zero-Leak QObject Contract

## Status
**Accepted** (Enforced by `TestPluginSmoke` & `PluginHost`)

## Context

In Qt applications, dynamic plugin loading is managed via `QPluginLoader`. A common expectation in modular architectures is that unloading a plugin (`QPluginLoader::unload()`) completely unmaps the underlying shared library (`.dll` / `.so`) from the host process address space.

However, modern Qt (and `QPluginLoader` specifically) intentionally sets or enforces `QLibrary::PreventUnloadHint` for loaded plugins:
1. **Virtual Method Tables and RTTI:** C++ plugins frequently instantiate classes whose vtables, `typeinfo`, and template singletons reside in the plugin's code section. If a background worker, asynchronous signal, or Qt meta-object registration accesses code from an unmapped library, a fatal access violation (`0xC0000005`) occurs immediately.
2. **Qt Meta-Object System:** `Q_OBJECT` metadata, signal/slot connection tables, and custom Qt property registrations are global within `QCoreApplication`. Unmapping code pages while metatypes or type registries hold references creates dangling function pointers.
3. **One-Way Flag:** `QLibrary` load hints can only be bitwise OR-ed; `PreventUnloadHint` cannot be cleared once set.

Consequently, calling `QPluginLoader::unload()` destroys the root instance of the plugin (`QObject` hierarchy), but **the library binary remains mapped in memory** for the lifetime of the application session.

### The Architectural Hazard

Because the code pages remain mapped, failure to clean up UI widgets does **not** immediately trigger a page-fault crash on access. Instead, widgets (dock widgets, ribbon tabs, menu items, event filters) installed by a plugin into the host shell remain present as **"dead UI"** — widgets that cannot respond, continue to receive event dispatches, or reference destroyed plugin services.

## Decision

We decouple the **physical library mapping** from the **logical plugin component lifecycle**, establishing an explicit **Zero-Leak QObject Contract**:

### 1. Two-Phase Teardown Architecture
- **Phase 1: Deterministic Synchronous Cleanup (`IPlugin::cleanup()`):**
  - Plugins must synchronously delete every UI artifact and hook installed during `initialize()`.
  - Dock widgets must be deleted via `IShellService::deleteDockWidget()`.
  - Ribbon panels must be removed via `delete m_panel` (the `AbstractRibbonPanel` destructor unlinks itself from the tab panel).
  - Event filters installed on `qApp` or shell windows must be explicitly removed with `removeEventFilter()`.
  - Note: Deferred deletion (`deleteLater()`) is prohibited in teardown paths because pending events may not process before application shutdown or subsequent plugin operations.
- **Phase 2: Component Destruction & Metric Accounting (`PluginHost::shutdown()`):**
  - `PluginHost` iterates through all registered plugins in **reverse initialization order** (satisfying dependency graphs).
  - It calls `plugin->cleanup()`, invokes `loader->unload()`, and verifies the destruction of the root component.
  - The number of released plugin instances is tracked via `PluginHost::releasedPluginCount()`.

### 2. Automated Contract Enforcement (`TestPluginSmoke`)
This contract is guarded continuously in CI via `tests/TestPluginSmoke.cpp`:
- The test dynamically loads each production plugin into a test shell.
- It verifies that UI artifacts were successfully registered (non-vacuous check).
- It triggers `PluginHost::shutdown()`.
- It performs a recursive inspection of the shell's `QObject` tree: **zero** objects belonging to the plugin's namespace or installed by the plugin may remain.

## Consequences

### Positive
- **No Dangling Pointers or Memory Leaks:** UI elements are cleanly removed and destroyed, preventing memory accumulation across reload cycles.
- **Fail-Safe Crash Prevention:** Keeping binary code pages mapped prevents catastrophic faults if delayed events or background threads briefly overlap with teardown.
- **Continuous Architectural Verification:** Any regression in plugin teardown immediately fails `TestPluginSmoke` in CTest.

### Negative / Trade-offs
- Process address space for unloaded plugin `.dll`s is not reclaimed until process exit. In desktop 64-bit architectures with 128 TB virtual address spaces, this overhead is negligible (~a few megabytes per library).
