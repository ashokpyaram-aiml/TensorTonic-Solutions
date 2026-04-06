import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    L = max_len or max(len(l) for l in seqs)
    r = np.full((N, L), pad_value)
    for i, s in enumerate(seqs):
         for j in range(min(len(s), L)):
            v = s[j]
            r[i, j] = v
    return r