import base64
import io
import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chatbot.server import make_server


class ChatbotHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(0)
        cls.url = 'http://127.0.0.1:' + str(cls.server.server_port)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, data=None, origin=None):
        headers = {'Content-Type': 'application/json'}
        if origin:
            headers['Origin'] = origin
        if isinstance(data, dict):
            data = json.dumps(data).encode()
        request = Request(self.url + path, data=data, headers=headers)
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def test_readiness_and_existing_text_modes(self):
        code, health = self.request('/health')
        self.assertEqual(code, 200)
        self.assertEqual(health['release'], '2026-10-05-learning')
        self.assertIn('waveform_continuation', health['capabilities'])
        for mode in ['generated', 'reference']:
            code, answer = self.request('/api/chat', {'message': 'What is chemistry?', 'mode': mode})
            self.assertEqual(code, 200)
            self.assertIn('chemistry', answer['text'].lower())
        code, answer = self.request('/api/chat', {'message': 'What happens next?', 'mode': 'continuation', 'context': 'The first seed was planted. It received water and sunlight.'})
        self.assertEqual(code, 200)
        self.assertEqual(answer['status'], 'experimental_continuation')
        self.assertTrue(answer['text'])

    def test_word_and_composition_use_integrated_heads(self):
        code, answer = self.request('/api/chat', {'message': 'looki', 'mode': 'word'})
        self.assertEqual(code, 200)
        self.assertEqual(answer['text'], 'looking')
        plan = {'state': {'a': 0, 'b': 0, 'c': 0}, 'goal': {'a': 1, 'b': 1, 'c': 1},
                'actions': [{'field': k, 'value': 1} for k in 'abc']}
        code, answer = self.request('/api/chat', {'message': json.dumps(plan), 'mode': 'composition'})
        self.assertEqual(code, 200)
        self.assertEqual(answer['status'], 'generated_path')
        result = json.loads(answer['text'])
        self.assertEqual(result['final_state'], plan['goal'])
        self.assertEqual(len(result['actions']), 3)
        plan['constraints'] = [{'a': 1}]
        code, answer = self.request('/api/chat', {'message': json.dumps(plan), 'mode': 'composition'})
        self.assertEqual(code, 200)
        self.assertEqual(answer['status'], 'no_solution_within_bounds')

    def test_audio_prediction_and_invalid_inputs(self):
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            t = np.arange(640) / 16000
            wav.writeframes((8000 * np.sin(2 * np.pi * 220 * t)).astype('<i2').tobytes())
        code, answer = self.request('/api/acoustic', buffer.getvalue())
        self.assertEqual(code, 200)
        with wave.open(io.BytesIO(base64.b64decode(answer['wav_base64'])), 'rb') as wav:
            self.assertEqual(wav.getnframes(), 2048)
            self.assertEqual(wav.getframerate(), 16000)
        for payload in [b'not a wav', buffer.getvalue()[:-20]]:
            self.assertEqual(self.request('/api/acoustic', payload)[0], 400)
        self.assertEqual(self.request('/api/chat', {'message': 'looki', 'mode': 'word'}, origin='https://other.example')[0], 403)
        self.assertEqual(self.request('/api/chat', {'message': 'hello 123', 'mode': 'word'})[0], 400)
        self.assertEqual(self.request('/api/chat', {'message': '{"state":[],"goal":{},"actions":[]}', 'mode': 'composition'})[0], 400)


if __name__ == '__main__':
    unittest.main()
