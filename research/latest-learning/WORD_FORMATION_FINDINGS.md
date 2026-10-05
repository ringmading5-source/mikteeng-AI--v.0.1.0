# Word formation test

The original CharacterPredictionModel algorithm was used without modifications. Both context lengths (16 default, 3 local) were specified for comparison, not tuned iteratively. Thirty-six training words contained twelve roots and two of three regular endings per root. Twelve full inflected test words were withheld. The model received learned start/stop markers and generated one character at a time, greedily, without a complete-word candidate list.

Eight tests supplied a root plus the first character of a multi-character ending (ed or ing). These offer a partially specified formation, not generation of an entire unfamiliar word from nothing. Default context: 0/8 correct. Three-character context: 3/8 correct. Successful outputs were looki -> looking, worki -> working, cooki -> cooking; full target words were absent from fitting.

The other four tests supplied only a root and expected s, but multiple valid endings were possible. They are ambiguous and should not be treated as definitive morphological failures. The stored total scores are 0/12 and 3/12 but the eight guided cases are the more interpretable comparison.

From a start marker alone, default context generated clean and local context generated cleaning. Both appeared in training. Autonomous novel-word generation was not demonstrated. Several guided failures stopped prematurely or made invalid forms, such as raineg and callin.

This is English character formation, not spoken pronunciation, African-language learning, semantic understanding or creativity. The character head belongs to the original package but is not the prediction-history Self learner. No main model code or checkpoint was changed by this test. Fresh CPU training took approximately 0.02 seconds for default and 0.01 seconds for local context in this small run; energy use was not measured.

Run with PYTHONPATH=src python examples/test_word_formation.py. The JSON file includes every training word, held-out word, generation and control outcome.
