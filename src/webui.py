from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn.functional as F
import re

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Word2Vec Calculator</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #fff; margin-bottom: 10px; }
        .subtitle { color: #888; margin-bottom: 30px; }
        .input-row {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        input[type="text"] {
            flex: 1;
            padding: 15px;
            font-size: 18px;
            border: 2px solid #333;
            border-radius: 8px;
            background: #16213e;
            color: #fff;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #4a9eff;
        }
        button {
            padding: 15px 30px;
            font-size: 16px;
            background: #4a9eff;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        button:hover { background: #3a8eef; }
        .error {
            background: #ff4a4a22;
            border: 1px solid #ff4a4a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .results {
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
        }
        .result-item {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #333;
        }
        .result-item:last-child { border-bottom: none; }
        .result-word {
            font-size: 18px;
            font-weight: 500;
        }
        .result-word.top {
            color: #4a9eff;
            font-size: 24px;
        }
        .result-score {
            color: #888;
            font-family: monospace;
        }
        .examples {
            margin-top: 30px;
            padding: 20px;
            background: #16213e;
            border-radius: 8px;
        }
        .examples h3 { margin-top: 0; color: #888; }
        .example {
            display: inline-block;
            padding: 8px 12px;
            margin: 5px;
            background: #1a1a2e;
            border-radius: 4px;
            cursor: pointer;
            color: #4a9eff;
        }
        .example:hover { background: #252545; }
    </style>
</head>
<body>
    <h1>Word2Vec Calculator</h1>
    <p class="subtitle">Explore word relationships with vector arithmetic</p>

    <div class="input-row">
        <input type="text" id="expression" placeholder="king - man + woman" autofocus>
        <button onclick="calculate()">Calculate</button>
    </div>

    <div id="error" class="error" style="display: none;"></div>
    <div id="results" class="results" style="display: none;"></div>

    <div class="examples">
        <h3>Try these examples:</h3>
        <span class="example" onclick="tryExample(this)">king - man + woman</span>
        <span class="example" onclick="tryExample(this)">paris - france + germany</span>
        <span class="example" onclick="tryExample(this)">good - bad + terrible</span>
        <span class="example" onclick="tryExample(this)">bigger - big + small</span>
    </div>

    <script>
        const input = document.getElementById('expression');
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') calculate();
        });

        function tryExample(el) {
            input.value = el.textContent;
            calculate();
        }

        async function calculate() {
            const expr = input.value.trim();
            if (!expr) return;

            const errorDiv = document.getElementById('error');
            const resultsDiv = document.getElementById('results');

            errorDiv.style.display = 'none';
            resultsDiv.style.display = 'none';

            try {
                const response = await fetch('/calculate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ expression: expr })
                });

                const data = await response.json();

                if (data.error) {
                    errorDiv.textContent = data.error;
                    errorDiv.style.display = 'block';
                    return;
                }

                let html = '';
                data.results.forEach((item, i) => {
                    const cls = i === 0 ? 'result-word top' : 'result-word';
                    html += `
                        <div class="result-item">
                            <span class="${cls}">${item.word}</span>
                            <span class="result-score">${item.score.toFixed(4)}</span>
                        </div>
                    `;
                });

                resultsDiv.innerHTML = html;
                resultsDiv.style.display = 'block';
            } catch (err) {
                errorDiv.textContent = 'Failed to connect to server';
                errorDiv.style.display = 'block';
            }
        }
    </script>
</body>
</html>
"""


class WordCalculator:
    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.vocab = checkpoint['vocab']
        self.idx_to_word = checkpoint['idx_to_word']
        self.embedding_dim = checkpoint['embedding_dim']

        from word2vec_model import Word2VecModel
        model = Word2VecModel(vocab_size=len(self.vocab), embedding_dim=self.embedding_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.embeddings = model.center_embeddings.weight.detach()
        self.embeddings_norm = F.normalize(self.embeddings, dim=1)

    def get_vector(self, word):
        word = word.lower().strip()
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary")
        return self.embeddings[self.vocab[word]]

    def find_nearest(self, vector, top_n=10, exclude_words=None):
        exclude_words = exclude_words or set()
        vector_norm = F.normalize(vector.unsqueeze(0), dim=1)
        similarities = torch.mm(vector_norm, self.embeddings_norm.t()).squeeze(0)

        # Exclude input words from results
        for word in exclude_words:
            if word in self.vocab:
                similarities[self.vocab[word]] = -float('inf')

        top_indices = similarities.argsort(descending=True)[:top_n]
        return [(self.idx_to_word[idx.item()], similarities[idx].item()) for idx in top_indices]

    def calculate(self, expression):
        # Parse expression like "king - man + woman"
        # Tokenize by + and -, keeping the operators
        tokens = re.split(r'\s*([+\-])\s*', expression.strip())
        tokens = [t.strip() for t in tokens if t.strip()]

        if not tokens:
            raise ValueError("Empty expression")

        # First word is always positive
        result_vector = self.get_vector(tokens[0])
        used_words = {tokens[0].lower()}

        i = 1
        while i < len(tokens):
            if i + 1 >= len(tokens):
                raise ValueError(f"Invalid expression: operator '{tokens[i]}' has no operand")

            operator = tokens[i]
            word = tokens[i + 1]
            used_words.add(word.lower())

            if operator == '+':
                result_vector = result_vector + self.get_vector(word)
            elif operator == '-':
                result_vector = result_vector - self.get_vector(word)
            else:
                raise ValueError(f"Unknown operator: {operator}")

            i += 2

        return self.find_nearest(result_vector, top_n=10, exclude_words=used_words)


def create_app(checkpoint_path):
    app = Flask(__name__)
    calculator = WordCalculator(checkpoint_path)

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route('/calculate', methods=['POST'])
    def calculate():
        data = request.get_json()
        expression = data.get('expression', '')

        try:
            results = calculator.calculate(expression)
            return jsonify({
                'results': [{'word': word, 'score': score} for word, score in results]
            })
        except ValueError as e:
            return jsonify({'error': str(e)})

    return app


def run_webui(checkpoint_path, host='127.0.0.1', port=5000):
    app = create_app(checkpoint_path)
    print(f"Starting Word2Vec Calculator at http://{host}:{port}")
    app.run(host=host, port=port)
