"""Unsupervised real-audio recurrence experiment.

Input: a user-provided audio recording converted to mono PCM WAV.
No transcript is used during clustering.

Protocol:
1. frame-level acoustic measurements,
2. adaptive activity segmentation,
3. temporal descriptors,
4. structural-distance clustering,
5. require recurrence at separated timestamps.

A resulting cluster is a candidate acoustic recurrence, NOT a word or meaning.
Use the existing tbin.audio frontend for production ingestion and validate
clusters against held-out recordings/speakers.
"""

def separation_margin(mean_different, mean_same):
    return mean_different - mean_same

def recurring_cluster_is_candidate(count, time_span_seconds,
                                   min_count=3, min_span_seconds=2.0):
    return count >= min_count and time_span_seconds >= min_span_seconds

if __name__ == "__main__":
    print("TBIN real-audio recurrence protocol.")
    print("No transcript labels are used during structural discovery.")
