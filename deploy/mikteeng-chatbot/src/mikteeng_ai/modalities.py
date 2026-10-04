"""CPU modality encoders and supervised shared-concept learning.

Acoustic/visual codebooks are fitted from examples, not semantic rules.
No transcription, speech synthesis, or image generation is implied.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

@dataclass(frozen=True)
class TokenStream:
    modality: str
    tokens: tuple
    positions: tuple
    source: str | None = None

class ModalityEncoder:
    def __init__(self, codebook_size=32, seed=42):
        if not isinstance(codebook_size, int) or codebook_size < 2:
            raise ValueError('codebook_size must be an integer >= 2')
        self.codebook_size = codebook_size
        self.seed = seed
        self.vocabulary = {}
        self.codebooks = {}
        self.scalers = {}

    @staticmethod
    def _words(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('text must be nonempty')
        return re.findall(r"\w+|[^\w\s]", text.casefold())

    @staticmethod
    def _audio(value):
        if isinstance(value, (str, Path)):
            rate, samples = wavfile.read(value)
        else:
            samples, rate = value
        samples = np.asarray(samples)
        if samples.ndim not in (1, 2) or samples.size == 0:
            raise ValueError('audio must contain mono or multichannel samples')
        if not isinstance(rate, (int, np.integer)) or rate <= 0:
            raise ValueError('sample rate must be positive')
        if samples.dtype.kind == 'u':
            midpoint = float(2 ** (samples.dtype.itemsize * 8 - 1))
            samples = (samples.astype(float) - midpoint) / midpoint
        elif samples.dtype.kind == 'i':
            samples = samples.astype(float) / float(2 ** (samples.dtype.itemsize * 8 - 1))
        else:
            samples = samples.astype(float)
        if not np.isfinite(samples).all():
            raise ValueError('audio contains nonfinite samples')
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        import math
        divisor = math.gcd(int(rate), 16000)
        samples = resample_poly(samples, 16000 // divisor, int(rate) // divisor)
        samples = np.pad(samples, (0, max(0, 400 - len(samples))))
        frames = np.stack([samples[i:i + 400] for i in range(0, len(samples) - 399, 160)])
        spectrum = np.abs(np.fft.rfft(frames * np.hanning(400))) ** 2
        # Fixed signal measurement; the clusters over these measurements are learned.
        bands = np.array_split(spectrum, 24, axis=1)
        return np.stack([np.log1p(b.mean(axis=1)) for b in bands], axis=1)

    @staticmethod
    def _image(value):
        from PIL import Image, ImageOps
        if isinstance(value, (str, Path)):
            with Image.open(value) as opened:
                image = ImageOps.exif_transpose(opened).convert('RGB')
        elif isinstance(value, Image.Image):
            image = ImageOps.exif_transpose(value).convert('RGB')
        else:
            arr = np.asarray(value)
            if arr.dtype != np.uint8 or arr.ndim not in (2, 3):
                raise ValueError('image arrays must be uint8 grayscale/RGB/RGBA')
            image = Image.fromarray(arr).convert('RGB')
        image = ImageOps.pad(image, (64, 64))
        pixels = np.asarray(image, dtype=float) / 255
        return np.stack([pixels[y:y + 8, x:x + 8].reshape(-1)
                         for y in range(0, 64, 8) for x in range(0, 64, 8)])

    def fit(self, modality, examples):
        examples = list(examples)
        if not examples:
            raise ValueError('training examples are empty')
        if modality == 'text':
            words = sorted({w for text in examples for w in self._words(text)})
            self.vocabulary = {word: i + 1 for i, word in enumerate(words)}
        elif modality in ('audio', 'image'):
            features = np.concatenate([getattr(self, '_' + modality)(v) for v in examples])
            scaler = StandardScaler().fit(features)
            scaler.scale_ = np.maximum(scaler.scale_, 0.05)
            normalized = scaler.transform(features)
            count = min(self.codebook_size, len(np.unique(normalized, axis=0)))
            model = KMeans(n_clusters=count, random_state=self.seed, n_init=10).fit(normalized)
            self.scalers[modality] = scaler
            self.codebooks[modality] = model
        else:
            raise ValueError('modality must be text, audio or image')
        return self

    def encode(self, modality, value):
        if modality == 'text':
            if not self.vocabulary:
                raise RuntimeError('fit the text vocabulary first')
            ids = [self.vocabulary.get(w, 0) for w in self._words(value)]
        elif modality in ('audio', 'image'):
            if modality not in self.codebooks:
                raise RuntimeError('fit the ' + modality + ' codebook first')
            features = getattr(self, '_' + modality)(value)
            ids = self.codebooks[modality].predict(self.scalers[modality].transform(features))
        else:
            raise ValueError('modality must be text, audio or image')
        return TokenStream(modality, tuple(int(i) for i in ids), tuple(range(len(ids))),
                           str(value) if modality != 'text' and isinstance(value, (str, Path)) else None)

    def decode_text(self, stream):
        if stream.modality != 'text':
            raise ValueError('only text tokens have a text decoder')
        inverse = {i: word for word, i in self.vocabulary.items()}
        return ' '.join(inverse.get(i, '<UNK>') for i in stream.tokens)

class MultimodalLearner:
    """Learn common labels across separately encoded modalities.

    Rows: {'modality': 'text'|'audio'|'image', 'data': value, 'label': str}.
    A prediction requires training examples for that modality. Shared labels
    do not establish zero-shot transfer or general semantic understanding.
    """
    def __init__(self, codebook_size=32, seed=42):
        self.encoder = ModalityEncoder(codebook_size, seed)
        self.classifier = None
        self.modalities = ()
        self.sizes = {}

    def _features(self, streams):
        rows = []
        for stream in streams:
            vector = []
            for modality in self.modalities:
                counts = np.zeros(self.sizes[modality])
                if modality == stream.modality:
                    counts = np.bincount(stream.tokens, minlength=self.sizes[modality]).astype(float)
                    counts /= max(1, counts.sum())
                vector.extend(counts)
                vector.append(float(modality == stream.modality))
            rows.append(vector)
        return np.asarray(rows)

    def fit(self, data):
        rows = list(data)
        if not rows or any(not isinstance(r, dict) or not {'modality', 'data', 'label'} <= r.keys() for r in rows):
            raise ValueError('rows require modality, data and label')
        if any(not isinstance(r['label'], str) or not r['label'].strip() for r in rows):
            raise ValueError('labels must be nonempty strings')
        if len({r['label'] for r in rows}) < 2:
            raise ValueError('at least two labels are required')
        # Build fresh state so a failed fit cannot corrupt an existing trained model.
        encoder = ModalityEncoder(self.encoder.codebook_size, self.encoder.seed)
        modalities = tuple(sorted({r['modality'] for r in rows}))
        for modality in modalities:
            encoder.fit(modality, [r['data'] for r in rows if r['modality'] == modality])
        candidate = MultimodalLearner(encoder.codebook_size, encoder.seed)
        candidate.encoder = encoder
        candidate.modalities = modalities
        candidate.sizes = {m: len(encoder.vocabulary) + 1 if m == 'text' else encoder.codebooks[m].n_clusters for m in modalities}
        streams = [encoder.encode(r['modality'], r['data']) for r in rows]
        candidate.classifier = LogisticRegression(max_iter=1000, C=10).fit(candidate._features(streams), [r['label'] for r in rows])
        self.__dict__.update(candidate.__dict__)
        return self

    def predict(self, modality, value):
        if self.classifier is None:
            raise RuntimeError('train the multimodal learner first')
        if modality not in self.modalities:
            raise ValueError('this modality was not trained')
        stream = self.encoder.encode(modality, value)
        probabilities = self.classifier.predict_proba(self._features([stream]))[0]
        index = int(probabilities.argmax())
        return {'label': str(self.classifier.classes_[index]), 'score': float(probabilities[index]),
                'modality': modality, 'token_count': len(stream.tokens)}
