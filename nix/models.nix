# Pinned model assets (what-we-build.md §7.2). Making the models part of the flake — rather
# than a download script — pins their hashes and makes a silent model swap (which would render
# all previously-computed embeddings incomparable) structurally impossible.
#
# URLs + hashes verified 2026-08-02; see docs/plans/2026-08-02-meetscribe-research.md §3.
#
# ``withLlm`` adds the transcript-cleanup GGUF under ``llm/model.gguf``. It is a SEPARATE,
# opt-in output (``packages.models-llm`` / ``meetscribe-llm``) so the default closure and CI
# stay lean — a machine without it just produces an uncleaned transcript (graceful fallback).
{ lib, fetchurl, runCommand, hunspellDicts, withLlm ? false }:

let
  asr = fetchurl {
    url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2";
    hash = "sha256-V5PQ/Tl8V3jSzyEmmU1Y6dVrG+fATRPHoVuxtOr7Fr8=";
  };
  # NOTE upstream tag is misspelled "speaker-recongition-models" — keep it.
  speaker = fetchurl {
    url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx";
    hash = "sha256-qjz8FpY6EFhqk5P1A11ta1fpjTWLNH+AwqML9PAM66I=";
  };
  segmentation = fetchurl {
    url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2";
    hash = "sha256-JGFe6ITIl9nSugm7TTDaa7GxXmhQZZYttbAuduSZZIg=";
  };
  vad = fetchurl {
    url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";
    hash = "sha256-niRJ4Qh0ltjUyrqQfyPgvT942R+lUkebucI6wJy7H9Y=";
  };
  # Cleanup LLM: Qwen2.5-7B-Instruct Q4_K_M (~4.68 GB, single file), pinned at a commit
  # revision (not `main`) to mirror the immutability of the release-tag pins above.
  llm = fetchurl {
    url = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/8911e8a47f92bac19d6f5c64a2e2095bd2f7d031/Qwen2.5-7B-Instruct-Q4_K_M.gguf";
    hash = "sha256-Zbj82Sr2tP76k1xiXRrCfqKdy27hRYnFWo8RXOqqFCM=";
  };
  # German dictionary for the broken-word flagger. Frami/igerman98 (from the LibreOffice
  # dictionaries repo, pinned by commit) — far better at compounds than nixpkgs' j3e de_DE
  # (measured ~18% fewer false flags on real audio). en_US still comes from nixpkgs.
  framiRev = "f2ff99058268502bdcf4cad25c1ca2935ad8aa7d";
  framiAff = fetchurl {
    url = "https://raw.githubusercontent.com/LibreOffice/dictionaries/${framiRev}/de/de_DE_frami.aff";
    hash = "sha256-ZGvzMzrGnCPp15RTPuUkHW91XDWej+EKZI+HYTdD1ZQ=";
  };
  framiDic = fetchurl {
    url = "https://raw.githubusercontent.com/LibreOffice/dictionaries/${framiRev}/de/de_DE_frami.dic";
    hash = "sha256-TKPJWLDlVFkQmZvCRvZohAv47ePfjl5nkNBe3VpYbDg=";
  };
in
# Assemble the layout the Python code expects under $MEETSCRIBE_MODELS:
#   asr/{encoder,decoder,joiner}.int8.onnx + tokens.txt   (Parakeet-TDT)
#   seg/model.int8.onnx                                    (pyannote segmentation-3.0, int8 for CPU)
#   spk/model.onnx                                         (CAM++, 192-dim)
#   vad/silero_vad.onnx
runCommand "meetscribe-models${lib.optionalString withLlm "-llm"}" { } ''
  mkdir -p $out/asr $out/seg $out/spk $out/vad
  tar xf ${asr}          -C $out/asr --strip-components=1
  tar xf ${segmentation} -C $out/seg --strip-components=1
  cp ${speaker} $out/spk/model.onnx
  cp ${vad}     $out/vad/silero_vad.onnx
  ${lib.optionalString withLlm ''
    mkdir -p $out/llm $out/hunspell
    cp ${llm} $out/llm/model.gguf
    # de (Frami/igerman98) + en_US dictionaries for the bilingual broken-word flagger
    # (lexicon.py); DICPATH points here at runtime.
    cp ${framiAff} $out/hunspell/de_DE.aff
    cp ${framiDic} $out/hunspell/de_DE.dic
    cp ${hunspellDicts.en_US}/share/hunspell/* $out/hunspell/
  ''}
''
