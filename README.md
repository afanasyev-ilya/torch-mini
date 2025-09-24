# NimbleML


### Installation

```
brew install ccache
brew install cmake
brew install ninja
```

### Build

```
cmake --preset dev-debug
cmake --build --preset dev-debug -j
```

### Test

```
ctest --preset dev-debug --output-on-failure
```
