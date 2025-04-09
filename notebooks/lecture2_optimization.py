import marimo

__generated_with = "0.11.25"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import optuna
    import torch
    import requests
    import torch.nn.functional as F
    return F, mo, optuna, requests, torch


@app.cell
def _(F, optuna, requests, torch):
    # Assume you have your data loading and preprocessing code here
    # For example, reading names.txt and creating train/dev/test sets
    # vocab, stoi, itos, train_dataset, dev_dataset, test_dataset = ...

    # Assume you have a function to build your MLP model
    def build_model(vocab_size, embedding_dim, hidden_dim, block_size, generator):
        class StringGenerator(torch.nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, block_size):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
                self.linear1 = torch.nn.Linear(block_size * embedding_dim, hidden_dim)
                self.linear2 = torch.nn.Linear(hidden_dim, vocab_size)
                self.block_size = block_size

            def forward(self, idx):
                embed = self.embedding(idx)
                # Concatenate the embeddings of the context characters
                x = embed.view(embed.size(0), -1)
                h = torch.tanh(self.linear1(x))
                logits = self.linear2(h)
                return logits

        model = StringGenerator(vocab_size, embedding_dim, hidden_dim, block_size)
        # Initialize weights (you might have your own initialization)
        for p in model.parameters():
            if p.ndim == 2:
                torch.nn.init.kaiming_normal_(p, generator=generator)
            elif p.ndim == 1:
                torch.nn.init.zeros_(p)
        return model

    # Assume you have a function to calculate the loss on a dataset
    def calculate_loss(model, dataset_x, dataset_y):
        logits = model(dataset_x)
        loss = F.cross_entropy(logits, dataset_y)
        return loss.item()

    # Assume you have a function to train the model for one epoch (or a number of steps)
    def train_step(model, optimizer, train_x, train_y, batch_size):
        model.train()
        optimizer.zero_grad()
        # Create mini-batches (simplified for example)
        indices = torch.randperm(train_x.size(0))[:batch_size]
        batch_x = train_x[indices]
        batch_y = train_y[indices]
        logits = model(batch_x)
        loss = F.cross_entropy(logits, batch_y)
        loss.backward()
        optimizer.step()
        return loss.item()

    # Define the objective function for Optuna
    def objective(trial):
        # Suggest hyperparameters
        embedding_dim = trial.suggest_int('embedding_dim', 2, 30)
        hidden_dim = trial.suggest_int('hidden_dim', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 1e-2, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [1])
        # You could also suggest block_size if you want to optimize that

        # Fix the generator for reproducibility within each trial
        generator = torch.Generator().manual_seed(2147483647)
        vocab_size = 27 # Assuming your vocabulary size is fixed
        block_size = 3  # Assuming your block size is fixed for this optimization

        # Build the model with suggested hyperparameters
        model = build_model(vocab_size, embedding_dim, hidden_dim, block_size, generator)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Load your training and development data (replace with your actual loading)
        # For this example, let's assume you have these loaded as tensors:
        # train_x, train_y, dev_x, dev_y

        # Train the model for a certain number of steps
        n_steps = 5000 # You can adjust the number of training steps per trial
        for step in range(n_steps):
            train_loss = train_step(model, optimizer, train_x, train_y, batch_size)
            # You could potentially use the training loss as an early stopping signal

        # Evaluate the model on the development set
        dev_loss = calculate_loss(model, dev_x, dev_y)
        return dev_loss

    def build_dataset(words, block_size):
        X, Y = [], []
        for w in words:
            context = [0] * block_size
            for ch in w + '.':
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    # Load your data and create train_x, train_y, dev_x, dev_y
    # Example using the provided transcript logic (adapt to your actual data)
    # Read dataset
    url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
    content_in_bytes = requests.get(url).content
    words = content_in_bytes.decode('utf-8').split()
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    block_size = 3 # Consistent with W1 size calculation
    train_words = words[:int(0.8*len(words))]
    dev_words = words[int(0.8*len(words)):int(0.9*len(words))]
    test_words = words[int(0.9*len(words)):]

    train_x, train_y = build_dataset(train_words, block_size)
    dev_x, dev_y = build_dataset(dev_words, block_size)
    test_x, test_y = build_dataset(test_words, block_size)

    # Create an Optuna study
    study = optuna.create_study(direction='minimize')
    # Run the optimization
    study.optimize(objective, n_trials=10) # Adjust n_trials as needed

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: {}".format(trial.params))
    return (
        block_size,
        build_dataset,
        build_model,
        calculate_loss,
        chars,
        content_in_bytes,
        dev_words,
        dev_x,
        dev_y,
        itos,
        objective,
        stoi,
        study,
        test_words,
        test_x,
        test_y,
        train_step,
        train_words,
        train_x,
        train_y,
        trial,
        url,
        words,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
