# Pinned model assets (what-we-build.md §7.2). Making the models part of the flake — rather
# than a download script — pins their hashes and makes a silent model swap (which would render
# all previously-computed embeddings incomparable) structurally impossible.
#
# URLs + hashes verified 2026-08-02; see docs/plans/2026-08-02-meetscribe-research.md §3.
{ fetchurl, runCommand }:

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
in
# Assemble the layout the Python code expects under $MEETSCRIBE_MODELS:
#   asr/{encoder,decoder,joiner}.int8.onnx + tokens.txt   (Parakeet-TDT)
#   seg/model.int8.onnx                                    (pyannote segmentation-3.0, int8 for CPU)
#   spk/model.onnx                                         (CAM++, 192-dim)
#   vad/silero_vad.onnx
runCommand "meetscribe-models" { } ''
  mkdir -p $out/asr $out/seg $out/spk $out/vad
  tar xf ${asr}          -C $out/asr --strip-components=1
  tar xf ${segmentation} -C $out/seg --strip-components=1
  cp ${speaker} $out/spk/model.onnx
  cp ${vad}     $out/vad/silero_vad.onnx
''
