import requests
import torch

def fetch_words() -> list[str]:
    # Read dataset
    url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
    content_in_bytes = requests.get(url).content
    words = content_in_bytes.decode('utf-8').split()
    return words    

def vocabulary_size(words: list[str]) -> int:
    stoi, _ = build_dicts(words)
    return len(stoi.keys())


def build_dicts(words: list[str]) -> tuple[dict, dict]:
    """
    Build dictionaries for characters to indices and vice versa
    """
    # Now we create index-mappings from chars to ints and back
    chars = set()
    for n in words:
        chars_from_name = [i for i in n]
        [chars.add(i) for i in chars_from_name]
    chars = sorted(list(chars))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {s:i for i,s in stoi.items()}
    return stoi, itos

def build_dataset(words: list[str], context_length: int = 3) -> tuple[torch.Tensor, torch.Tensor]:

    X, Y = [], []

    stoi,_ = build_dicts(words)

    for w in words:
        context = [0] * context_length
        # we want context to grow with every char
        for char in w + '.':
            prev_chars = context
            target = char
            X.append(prev_chars)
            Y.append(stoi[target])
            idx_char = stoi[char]
            context = context[1:] + [idx_char]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y