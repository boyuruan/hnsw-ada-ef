# Agent instructions

<!-- inkan -->
<!-- inkan-protocol: 7 -->
<!-- inkan-lang: en -->

## Agent protocol: sealed outcomes

This repository uses Inkan (`inkan`, alias `ink`). Inkan keeps a trustworthy record of what the work was meant to deliver and what was declared at close. It does not inspect commits, run tests, or judge the result; the repository's own checks do that. Write outcome prose in en. This block states the policy; `inkan help` gives the command syntax.

1. **Seal before durable changes.** Before changing code, configuration, documentation, or dependencies, run `inkan status`; if it shows an open outcome that is not your work, follow rule 4 first. Then run `inkan begin` with the outcome, one observable acceptance criterion at a time, and every decision record the work is bound by. File the outcome by lane only when the repository already files outcomes by lane.
2. **The seal is a fact.** Deliver what it says. If circumstances change, do not reinterpret it: run `inkan amend` with the reason and the added or withdrawn criteria. The original text stays. Never question why the outcome was sealed the way it was at the time.
3. **Close with dispositions, then commit.** Run `inkan end` with a disposition, met or unmet, for every live criterion and a note on what happened. Commit the outcome record with the work. Include the printed `Inkan-Outcome: <id>` trailer in the final paragraph of the landing commit message, beside any other trailers with no blank line between them. Never report success without closing the outcome.
4. **Re-anchor after context loss.** Run `inkan status` and `inkan log -n 3`. An open outcome that is the work you were asked to do is your task: continue it, or close it with a note. An open outcome that is not your work belongs to another session: leave it alone. Never close, amend, or abandon an outcome you did not work on, and do not judge why it is still open. Before beginning your own outcome beside it, stop and tell the person it is there, and ask whether your work should run in its own git worktree, because separate worktrees keep each session's edits apart.
5. **Closed outcomes are final.** Reviewing the log is reading, not re-checking. Never re-verify, re-attest, or re-close a closed outcome. If a past declaration now looks wrong, that is a new outcome with its own seal. When reading history, use commit trailers only as references. Missing trailers or unavailable referenced records are missing information, not failed outcomes or a reason to verify delivery or repair history.

Decision records live in `.inkan/decisions/`. Their Context and Decision sections record the scenario at the time and are never edited. To challenge one, run `inkan decision update` with the new status and the reason, or add a new record that supersedes it.

Outcome log: `.inkan/outcomes/<id>.jsonl`, one append-only file per outcome. Commit `.inkan/` with the code. Do not edit these files by hand.
<!-- /inkan -->

## Project layout

- `hnswlib/` — header-only HNSW plus Ada-ef (`adaptive_ef.h`, `distribution.h`, `sketch.h`).
- `experiments_driver/` — experiment and index-build drivers:
  - `fvecs_io.h` — TEXMEX fvecs/ivecs I/O (no HDF5).
  - `util.h` — fvecs load, HNSW build/search helpers (no HDF5).
  - `run_fvecs.cpp` — `run_fvecs` CLI: build/search from fvecs/ivecs.
  - `hdf5_io.h` / `run.cpp` — paper experiments; need HDF5.
- `tests/cpp/fvecs_io_smoke.cpp` — fvecs roundtrip smoke test.
- `python_bindings/` — Python package around hnswlib.
- `examples/` — small C++/Python examples.
- `benchmarking-darth/` — DARTH / FAISS comparison (separate CMake).

## Compile

Needs C++17, CMake, Eigen 3.4, Boost (math headers). Default paths in `CMakeLists.txt` match Debian/Ubuntu packages (`libeigen3-dev`, `libboost-dev`). Override `EIGEN3_INCLUDE_DIR` / `BOOST_ROOT` if needed.

**Without HDF5** (this is the usual machine that only builds indexes from fvecs):

```bash
cmake -S . -B build -DHNSWLIB_WITH_HDF5=OFF
cmake --build build -j --target run_fvecs fvecs_io_smoke
```

```bash
./build/run_fvecs build --base BASE.fvecs --index OUT.hnsw [--metric l2|cd] [--M 16] [--efc 500] [--threads N]
./build/run_fvecs online --base BASE.fvecs --query QUERY.fvecs --neighbors GT.ivecs --index INDEX.hnsw \
    --dataset NAME [--metric l2|cd] [--k 100]
```

`online` still needs `EXPERIMENTS_ROOT` for Ada-ef estimator / adaptor files.

**With HDF5** (paper experiment suite). Also needs `libhdf5-dev`. Adjust `HDF5_INCLUDE_DIR` / `HDF5_LIB_DIR` if headers are not under `/usr/include/hdf5/serial`.

```bash
cmake -S . -B build -DHNSWLIB_WITH_HDF5=ON
cmake --build build -j --target run run_fvecs
export EXPERIMENTS_ROOT=/path/to/experiments
./build/run
```

If `H5Cpp.h` is missing, CMake turns `HNSWLIB_WITH_HDF5` off and only builds `run_fvecs`.

