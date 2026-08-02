# Lets a plain `nix-shell` reuse the flake's devShell (parity with `nix develop`).
# We reference devShells.<system>.default directly rather than flake-compat's `.shellNix`
# wrapper, whose generated shell-shim is broken with this flake-compat release.
(
  import
    (
      let lock = builtins.fromJSON (builtins.readFile ./flake.lock);
      in fetchTarball {
        url = lock.nodes.flake-compat.locked.url
          or "https://github.com/edolstra/flake-compat/archive/${lock.nodes.flake-compat.locked.rev}.tar.gz";
        sha256 = lock.nodes.flake-compat.locked.narHash;
      }
    )
    { src = ./.; }
).defaultNix.devShells.${builtins.currentSystem}.default
