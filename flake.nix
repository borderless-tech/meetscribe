{
  description = "meetscribe — offline dual-track meeting recorder + diarized transcriber (CPU-only)";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # flake-compat lets a plain `nix-shell` reuse the flake's devShell (shell.nix).
    flake-compat.url = "https://flakehub.com/f/edolstra/flake-compat/1.tar.gz";

    # uv2nix stack — builds a reproducible Python env from pyproject.toml + uv.lock.
    # NOTE: these live under the `pyproject-nix` org (not `adisbladis`).
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, flake-utils, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        inherit (nixpkgs) lib;
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312; # must satisfy requires-python in pyproject.toml

        # Load the uv workspace (pyproject.toml + uv.lock) and turn it into an overlay.
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
        overlay = workspace.mkPyprojectOverlay {
          # MANDATORY: prefer wheels. A source build drags in CMake + half of ONNX Runtime.
          sourcePreference = "wheel";
        };

        # Native-wheel fixup. pyproject.nix's wheel builder already runs autoPatchelfHook on
        # binary wheels; we only supply the libs it can't otherwise resolve.
        #
        # sherpa-onnx is split across two wheels:
        #   - sherpa-onnx-core : ships libonnxruntime.so + libsherpa-onnx-*-api.so
        #   - sherpa-onnx      : the pybind extension `_sherpa_onnx*.so`, which is NEEDED-linked
        #                        against libonnxruntime.so — but that lib lives in the OTHER
        #                        wheel, and autoPatchelf can't see a sibling package.
        # So the extension override must add the built core package to its patch path (empirically
        # verified: without it autoPatchelf fails on `libonnxruntime.so not found`). Both wheels
        # additionally need libstdc++/libgcc_s (via stdenv.cc.cc.lib).
        pyprojectOverrides = final: prev: {
          sherpa-onnx-core = prev.sherpa-onnx-core.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.stdenv.cc.cc.lib ];
          });
          sherpa-onnx = prev.sherpa-onnx.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.stdenv.cc.cc.lib ];
            # libonnxruntime.so ships in the sibling core wheel and can't be resolved at
            # build time. Defer it, and pin an explicit runpath to the core package's lib
            # dir so the loader finds it at runtime (both wheels merge into sherpa_onnx/lib).
            autoPatchelfIgnoreMissingDeps = [ "libonnxruntime.so" ];
            appendRunpaths =
              [ "${final.sherpa-onnx-core}/${python.sitePackages}/sherpa_onnx/lib" ];
          });
        };

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope
            (lib.composeManyExtensions [
              # `.wheel` (not `.default`, which is sdist) to match sourcePreference = "wheel".
              pyproject-build-systems.overlays.wheel
              overlay
              pyprojectOverrides
            ]);

        # ---- Editable dev environment ------------------------------------------------
        # src/meetscribe is mounted editable so the red-green TDD loop needs no rebuild;
        # every third-party dep (incl. the sherpa-onnx native stack) comes from Nix.
        editableOverlay = workspace.mkEditablePyprojectOverlay { root = "$REPO_ROOT"; };
        editablePythonSet = pythonSet.overrideScope (lib.composeManyExtensions [
          editableOverlay
          (final: prev: {
            meetscribe = prev.meetscribe.overrideAttrs (old: {
              src = lib.fileset.toSource {
                root = old.src;
                fileset = lib.fileset.unions [
                  (old.src + "/pyproject.toml")
                  (old.src + "/README.md")
                  (old.src + "/src")
                ];
              };
              nativeBuildInputs = old.nativeBuildInputs
                ++ final.resolveBuildSystem { editables = [ ]; };
            });
          })
        ]);

        # deps.all includes the `dev` dependency-group (pytest); deps.default is runtime-only.
        devVenv = editablePythonSet.mkVirtualEnv "meetscribe-dev-env" workspace.deps.all;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            devVenv
            pkgs.uv
            pkgs.ffmpeg-headless
          ];
          env = {
            # uv is present for lockfile maintenance only — never to sync/run inside the shell.
            UV_NO_SYNC = "1";
            UV_PYTHON = python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";
          };
          shellHook = ''
            unset PYTHONPATH
            export REPO_ROOT=$(git rev-parse --show-toplevel)
          '';
        };

        # The runtime package (wrapper), apps, models, overlays and the Home-Manager module
        # are added in chunks C7/C8. Kept out of C0 so this flake evaluates + builds today.
      });
}
