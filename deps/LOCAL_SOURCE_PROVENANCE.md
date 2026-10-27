# Offline dependency sources

These source-only copies were recovered from `build_avxon_backup_20261025` on 2026-10-25. No object files, archives, shared libraries, or CMake caches were copied.

- `hexl-1.2.5`: Intel HEXL, revision `f95acf1507b246946a34cda446f3a028fafb1c4b` (v1.2.5), matching SEAL 4.1.1 `cmake/ExternalIntelHEXL.cmake`.
- `cpu_features`: Google cpu_features, revision `32b49eb5e7809052a28422cfde2f2745fbb0eb76`, matching HEXL 1.2.5 `cmake/third-party/cpu-features/CMakeLists.txt.in`.

License files are retained in each directory. The older `deps/hexl` checkout is left unchanged. The build script uses these local sources when present and accepts `HEXL_SOURCE_DIR` / `CPU_FEATURES_SOURCE_DIR` overrides.
