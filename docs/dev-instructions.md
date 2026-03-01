# 🤖 MXFrame — Development Instructions

> Guidelines for Copilot (and human developers) working on this project.

> GPU custom-op development must follow: `docs/gpu-custom-op-guidelines.md`.

---

## 1. Project Structure Rules

### nbdev is the Source of Truth

All Python code lives in **notebooks first**, then gets exported to `.py` files.

```
nbs/01_lazy_expr.ipynb       →  mxframe/lazy_expr.py
nbs/02_lazy_frame.ipynb      →  mxframe/lazy_frame.py
nbs/03_compiler.ipynb        →  mxframe/compiler.py
nbs/04_custom_ops.ipynb      →  mxframe/custom_ops.py
nbs/05_xxx.ipynb             →  mxframe/xxx.py       (future)
```

**Never edit `.py` files directly.** Always edit the notebook, then run `nbdev_export`.

### File Naming Convention

| Type | Pattern | Example |
|------|---------|---------|
| **Notebook (code)** | `nbs/NN_module_name.ipynb` | `nbs/05_sort.ipynb` |
| **Notebook (bench)** | `nbs/bench_*.ipynb` | `nbs/bench_q1.ipynb` |
| **Mojo kernel** | `mxframe/kernels/kernel_name.mojo` | `mxframe/kernels/group_min.mojo` |
| **Doc** | `docs/topic.md` | `docs/roadmap.md` |
| **Script** | `scripts/purpose.py` | `scripts/build_kernels.sh` |

### Module Numbering

Keep notebooks numbered sequentially by dependency order:

```
01_lazy_expr       (no deps within mxframe)
02_lazy_frame      (depends on 01)
03_compiler        (depends on 01, 02)
04_custom_ops      (depends on 01, 02, 03)
05_...             (depends on prior)
```

---

## 2. Notebook Best Practices

### Keep Notebooks Small

Each notebook should cover **one module** with **one clear purpose**. Target:
- **< 10 code cells** for the exported module
- **< 5 test cells** at the bottom
- **< 300 lines** of total code

If a notebook grows beyond this, split it.

### Required Cell Structure

Every code notebook must follow this pattern:

```
Cell 1:  Markdown — Title with emoji, one-line description
Cell 2:  Code    — #| default_exp module_name
Cell 3:  Code    — #| export \n imports
Cell 4:  Markdown — Section header explaining the class/concept
Cell 5:  Code    — #| export \n class/function definition
...repeat sections as needed...
Cell N-1: Markdown — ## Tests 🧪
Cell N:   Code    — test code (NOT exported)
```

### nbdev Directives

| Directive | When to use |
|-----------|------------|
| `#| default_exp module_name` | First code cell of every notebook |
| `#| export` | Any cell whose code should go into the `.py` module |
| `#| hide` | Helper cells that shouldn't appear in docs |
| `#| eval: false` | Cells that shouldn't run during `nbdev_test` (e.g., GPU-only) |

### Export Workflow

```sh
# After editing a notebook:
pixi run nbdev_export          # notebooks → .py files
pixi run nbdev_test            # run all notebook tests
pixi run nbdev_docs            # build documentation site
```

### Documentation Style

- Use **emojis** in markdown headers: 🚀 🛠️ 🧪 ⚡ 📐 🔥 📊 💡
- Write **why**, not just what. Explain the design decision.
- Include **ASCII diagrams** for data flow when helpful.
- Add **type hints** to all function signatures.
- Write **docstrings** for every public class and method.

---

## 3. Coding Standards

### Python

- **Type hints everywhere**: `def foo(x: int, y: str) -> pa.Table:`
- **Dataclasses for plan nodes**: `@dataclass class Sort(LogicalPlan):`
- **No global state**: compilers are instantiated, not singletons
- **PyArrow in, PyArrow out**: the public API always accepts/returns `pa.Table`
- **f-strings for errors**: `raise NotImplementedError(f"Unsupported: {op}")`

### Mojo Kernels

- One kernel per `.mojo` file
- Register with `@compiler.register` decorator
- Always provide **both CPU and GPU** implementations when possible
- Use `alias dtype = DType.float32` at the top — keep it consistent
- Document the kernel signature in a comment block at the top of the file
- Re-export from `__init__.mojo`

### Testing

- **Every notebook has tests** at the bottom (after the `## Tests 🧪` section)
- Tests should be self-contained: create small PyArrow tables inline
- Assert **exact values** for small inputs, **approximate** for floating point
- Print a `✅ Test N passed: description` line for each test
- End with `print("\nAll XYZ tests passed! 🎉")`

---

## 4. Architecture Rules for Copilot

### The Golden Rule

> **Everything goes through the MAX Graph.**
> Python builds the plan. The compiler translates it to a graph. Mojo kernels execute it.
> Python never touches the data in the hot path.

### When Adding a New Operation

1. **Add the `Expr` method** in `01_lazy_expr.ipynb` (e.g., `.count()`)
2. **Add the plan node** in `02_lazy_frame.ipynb` if needed (e.g., `Sort`)
3. **Add the compiler visitor** in `03_compiler.ipynb` (built-in MAX ops)
4. **Add the Mojo kernel** in `mxframe/kernels/` if a custom kernel is needed
5. **Wire the kernel** in `04_custom_ops.ipynb`
6. **Add tests** in the same notebook
7. **Run `nbdev_export`** to update `.py` files
8. **Run `nbdev_test`** to verify everything

### When Adding a New Mojo Kernel

1. Create `mxframe/kernels/kernel_name.mojo`
2. Add `from .kernel_name import *` to `mxframe/kernels/__init__.mojo`
3. Rebuild: `bash scripts/build_kernels.sh`
4. Wire it in `04_custom_ops.ipynb` via `ops.custom("kernel_name", ...)`
5. Test in the notebook

### Compiler Design Principles

- **`GraphCompiler`** uses only built-in `ops.*` — no custom kernels. It's the reference implementation.
- **`CustomOpsCompiler`** extends `GraphCompiler` and overrides specific visitors to dispatch to Mojo kernels.
- The custom compiler should **always fall through** to the parent for anything it doesn't explicitly handle.
- **Never do computation in Python** during `compile_and_run`. PyArrow is only used for:
  - Reading the source table (`Scan`)
  - Building group IDs (dictionary encoding) — this is metadata, not data processing
  - Converting results back to `pa.Table`

---

## 5. Kernels Path Resolution

The `kernels.mojopkg` path must resolve correctly in two contexts:

| Context | How `KERNELS_PATH` resolves |
|---------|-----------------------------|
| **Installed package** (`import mxframe`) | `Path(__file__).parent / "kernels.mojopkg"` → `mxframe/kernels.mojopkg` |
| **Notebook** (`nbs/04_custom_ops.ipynb`) | `__file__` is not defined → use absolute path or workspace-relative |

In test cells within notebooks, always use:

```python
kernels_path = str(Path('/home/ablearn/mxdf/mxframe/kernels.mojopkg'))
```

Or dynamically:

```python
import mxframe
kernels_path = str(Path(mxframe.__file__).parent / 'kernels.mojopkg')
```

---

## 6. Git Workflow

```sh
# Before committing:
pixi run nbdev_export          # sync notebooks → .py
pixi run nbdev_test            # run all tests
pixi run nbdev_clean           # strip output from notebooks

# Commit message format:
# feat: add group_min kernel and compiler support
# fix: correct filter to use gather instead of mask multiply
# docs: update roadmap with Q1 benchmark results
# refactor: split compiler into base + custom
```

---

## 7. Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Editing `.py` files directly | Always edit the notebook. Run `nbdev_export`. |
| `kernels.mojopkg` not found | Rebuild: `bash scripts/build_kernels.sh` |
| Notebook kernel hangs | Select "Python 3 (ipykernel)" from pixi env |
| `__file__` undefined in notebook | Use absolute path or `import mxframe; Path(mxframe.__file__)` |
| MAX Graph type mismatch | Check `_max_dtype()` mapping; cast with `ops.cast()` |
| GPU kernel fails on CPU | Check `DeviceRef.CPU()` vs `DeviceRef.GPU()` in `TensorType` and `ops.custom()` |

---

## 8. Reference Resources 📚

We have two valuable reference resources in the `llm-ctx/` folder. Use them as lookup aids — don't copy, but peek when stuck.

### 8.1 Mojo & MAX API Context Files

Pre-built LLM context files following the [llmstxt.org](https://llmstxt.org) standard:

| File | Lines | Contents |
|------|-------|----------|
| `llm-ctx/llms-mojo.txt` | ~63K | Mojo language API docs |
| `llm-ctx/llms-python.txt` | ~35K | MAX Python API docs (Graph, Engine, etc.) |
| `llm-ctx/llms-full.txt` | ~223K | Combined Mojo + Python (everything) |

**When to use:** Look up exact function signatures, `ops.*` available operations, `TensorType` constructors, `DeviceRef` options, or any Mojo stdlib API you're unsure about.

### 8.2 MojoFrame — Academic Reference Implementation

`llm-ctx/MojoFrame/` contains a separate academic project: a pure-Mojo DataFrame library that has **implemented all 22 TPC-H benchmark queries**.

**Key facts:**
- **Pure Mojo** — no MAX Graph API, no PyArrow, no Python
- Located at `llm-ctx/MojoFrame/Mojoframe/` with `core/` (Arrays, Calculations, DataFrame) + `main.mojo` (all 22 query implementations)
- `Calculations.mojo` (~3900 lines) has filter, groupby, sort, join, aggregation logic — all in raw Mojo
- `main.mojo` (~5400 lines) has `test_query_1()` through `test_query_22()`
- They hit **string handling limitations** in Mojo and built workarounds — we don't need those since we use PyArrow for string columns

**When to use:**
- Peeking at **how a specific TPC-H query is decomposed** into operations (e.g., Q5's 6-way join order)
- Understanding **what aggregations/operations each query needs** before implementing
- Checking **algorithmic approaches** for sort, join, groupby in Mojo
- Seeing **what edge cases** they encountered (especially with data types)

**When NOT to copy:**
- Their architecture is fundamentally different (no Graph compilation, no lazy evaluation)
- Their string workarounds — we use PyArrow string arrays
- Their I/O — they parse CSV manually; we use PyArrow readers
- Their memory management — we let MAX Graph handle tensor lifetimes

**Example usage:** Before implementing Phase 5 (Joins), read their join implementation in `Calculations.mojo` to understand hash join patterns in Mojo. Then implement it as a MAX Graph custom op with our `ops.custom()` registration pattern.