"""Unicode letter/mark adapter, compatible with the previous ASCII word encoder."""
import hashlib
import unicodedata
import numpy as np
VERSION='mikteeng-unicode-letters-v1'
def tokenize(text):
 if not isinstance(text,str):raise TypeError('text must be a string')
 # Reject unpaired surrogates: input must be valid Unicode encodable as UTF-8.
 text.encode('utf-8',errors='strict')
 canonical=unicodedata.normalize('NFC',unicodedata.normalize('NFC',text).lower())
 words=[];current=[]
 for ch in canonical:
  kind=unicodedata.category(ch)
  if kind.startswith('L') or (kind.startswith('M') and current):current.append(ch)
  elif current:words.append(''.join(current));current=[]
 if current:words.append(''.join(current))
 return words
def encode_text(text,basis):
 basis=np.asarray(basis)
 if basis.ndim!=2 or basis.shape[0]!=basis.shape[1] or basis.shape[0]<160:raise ValueError('square basis with at least 160 dimensions required')
 words=tokenize(text)
 if not words:raise ValueError('input must contain at least one Unicode letter')
 vector=np.zeros(basis.shape[1])
 for pos,word in enumerate(words):
  seed=int.from_bytes(hashlib.sha256(f'{pos}:{word}'.encode('utf-8')).digest()[:8],'little')
  vector+=basis[np.random.default_rng(seed).choice(np.arange(16,160),8,replace=False)].sum(axis=0)
 return vector/np.linalg.norm(vector)
def metadata():
 return dict(version=VERSION,normalization='NFC',case='Unicode lowercase',token_categories=['L','M after letter'],hash='SHA256 of position:token encoded as UTF-8',basis_indices=[16,160],indices_per_token=8,unicode_database=unicodedata.unidata_version)
