# Home-Manager module: installs the meetscribe CLI for a user.
#
# (An optional systemd/launchd auto-processing service is intentionally left out for v1 —
# auto-processing semantics aren't pinned down yet, so shipping a half-defined watcher would be
# YAGNI. The option namespace below is kept minimal and easy to extend later.)
self:
{ config, lib, pkgs, ... }:

let
  cfg = config.programs.meetscribe;
in
{
  options.programs.meetscribe = {
    enable = lib.mkEnableOption "meetscribe, the offline meeting recorder + transcriber";

    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      defaultText = lib.literalExpression "meetscribe.packages.\${system}.default";
      description = "The meetscribe package to install.";
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = [ cfg.package ];
  };
}
