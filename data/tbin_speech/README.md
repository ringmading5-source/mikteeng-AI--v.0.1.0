# TBIN real-speech pilot data

Put 16-bit PCM WAV recordings here using:

    <concept>/<speaker>/<recording>.wav

Minimum useful pilot:
- 5 concepts;
- at least 2 speakers per concept;
- 5-10 repetitions per speaker;
- recordings not reused between evaluation conditions.

Example:

    water/speaker01/001.wav
    water/speaker02/001.wav
    fire/speaker01/001.wav

Run:

    python experiments/tbin_real_speech.py data/tbin_speech

Primary falsification metric:

    margin = mean(different-concept distance) - mean(same-concept cross-speaker distance)

A positive margin is necessary but not sufficient evidence that TBIN's current
representation suppresses speaker variation while retaining concept structure.
Do not report SOTA, zero-shot semantic understanding, or noise robustness from
this pilot alone.
