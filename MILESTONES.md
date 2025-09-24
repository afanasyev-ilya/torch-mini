# minitorch-cpp — Milestone Roadmap

A compact C++23 learning & research project building a tiny PyTorch-like stack from scratch: tensors ➜ autograd ➜ modules ➜ data/tokenization ➜ tiny-GPT ➜ polish & performance.

Each milestone ends with a **working demo + tests**.

---

## Table of Contents
- [Project Standards](#project-standards)
- [Quick Start (Deps & Build)](#quick-start-deps--build)
- [Repository Layout (suggested)](#repository-layout-suggested)
- [Milestones](#milestones)
  - [M0 — Project scaffold](#m0--project-scaffold)
  - [M1 — Tensor & storage (CPU)](#m1--tensor--storage-cpu)
  - [M2 — Autograd (reverse-mode)](#m2--autograd-reverse-mode)
  - [M3 — NN module system](#m3--nn-module-system)
  - [M4 — Data & tokenization](#m4--data--tokenization)
  - [M5 — Tiny-GPT (GPT-2–ish, minimal)](#m5--tiny-gpt-gpt-2ish-minimal)
  - [M6 — Engineering polish](#m6--engineering-polish)
  - [M7 — Performance & extensions](#m7--performance--extensions)
  - [M8 — “Use like a library”](#m8--use-like-a-library)
- [CI / Formatting / Linting](#ci--formatting--linting)
- [License](#license)

---

## Project Standards

- **Language:** C++23  
- **Build:** CMake
- **Testing:** doctest _or_ Catch2
- **Benchmarks:** Google Benchmark
- **Pkg mgr:** vcpkg _or_ Conan
- **Style:** clang-format, clang-tidy
- **CI:** GitHub Actions (build, test, format check)

Minimal CMake baseline:

```cmake
cmake_minimum_required(VERSION 3.24)
project(minitorch-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optional: tighten warnings
if (MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
