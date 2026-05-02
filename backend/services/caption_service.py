import json, time, torch
from PIL import Image
from torchvision import transforms                              # fix 1: was 'transform'
from .model_definitons import ImageCaptioner, ImageCaptionerV2


# fix 2: removed duplicate standalone _load_v1 / _load_v2 functions


def load_vocab(path):
    with open(path) as f:
        data = json.load(f)
    class Vocab:
        def __init__(self):
            self.word2idx = data['word2idx']
            self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        def decode(self, indices):
            words = []
            for idx in indices:
                word = self.idx2word.get(idx, '<unk>')         # fix 3: missing >
                if word in ('<pad>', '<start>'): continue
                if word == '<end>': break
                words.append(word)
            return ' '.join(words)                             # fix 4: outside loop
    return Vocab()


# ── Image preprocessing — must match training exactly ───────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# ── Model registry — loaded ONCE at startup ─────────────────────────
class CaptionService:
    def __init__(self, v1_path, v2_path, vocab_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab     = load_vocab(vocab_path)
        self.transform = get_transform()
        self.models    = {}

        print('Loading V1 model...')
        self.models['v1'] = self._load_v1(v1_path)

        print('Loading V2 model...')
        self.models['v2'] = self._load_v2(v2_path)

        print('Both models ready.')

    def _load_v1(self, path):
        # You will fill this in — import your ImageCaptioner class
        vocab_size = len(self.vocab.word2idx)
        model = ImageCaptioner(vocab_size)
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state'])
        model.to(self.device)
        model.eval()
        return model

    def _load_v2(self, path):
        # You will fill this in — import your ImageCaptionerV2 class
        vocab_size = len(self.vocab.word2idx)
        model = ImageCaptionerV2(vocab_size)
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state'])
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_bytes: bytes, model_name: str) -> dict:
        if model_name not in self.models:
            raise ValueError(f'Unknown model: {model_name}. Choose v1 or v2.')

        # Convert raw bytes → PIL → tensor
        from io import BytesIO
        image_pil    = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        model = self.models[model_name]
        model.eval()

        start = time.time()
        with torch.no_grad():
            caption = model.generate(image_tensor, self.vocab)
        elapsed = (time.time() - start) * 1000   # ms

        return {
            'caption' : caption,
            'model'   : model_name,
            'time_ms' : round(elapsed, 2)
        }