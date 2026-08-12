# The shippable meetscribe wrapper (what-we-build.md §7.3). Beyond Nix itself the user needs
# nothing preinstalled (except BlackHole on macOS): ffmpeg and the pinned models come from the
# store, wired in via makeWrapper.
# ``llama-cpp`` is optional: only the ``meetscribe-llm`` variant (paired with a models dir that
# carries llm/model.gguf) puts ``llama-server`` on PATH for the transcript-cleanup pass.
{ lib, stdenv, makeWrapper, ffmpeg, venv, models, llama-cpp ? null }:

stdenv.mkDerivation {
  pname = "meetscribe";
  version = "0.1.0";
  src = ../bin;

  nativeBuildInputs = [ makeWrapper ];
  dontBuild = true;

  installPhase = ''
    runHook preInstall
    install -Dm755 meetscribe $out/bin/meetscribe
    wrapProgram $out/bin/meetscribe \
      --prefix PATH : ${lib.makeBinPath ([ venv ffmpeg ] ++ lib.optional (llama-cpp != null) llama-cpp)} \
      --set MEETSCRIBE_MODELS ${models}
    runHook postInstall
  '';

  meta = {
    description = "Offline dual-track meeting recorder + diarized transcriber (CPU-only)";
    mainProgram = "meetscribe";
    platforms = lib.platforms.linux ++ lib.platforms.darwin;
  };
}
