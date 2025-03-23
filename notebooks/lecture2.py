import marimo

__generated_with = "0.11.25"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import requests
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt # for making figures
    return F, mo, plt, requests, torch


@app.cell
def _(mo):
    mo.md(r"""We follow the [lecture 2](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&ab_channel=AndrejKarpathy) of Andrej Karpathy's course and add some comments.""")
    return


@app.cell
def _(requests):
    # Read dataset
    url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
    content_in_bytes = requests.get(url).content
    words = content_in_bytes.decode('utf-8').split()
    return content_in_bytes, url, words


@app.cell
def _(words):
    # Now we create index-mappings from chars to ints and back
    chars = set()
    for n in words:
        chars_from_name = [i for i in n]
        [chars.add(i) for i in chars_from_name]
    chars = sorted(list(chars))
    # char to index
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    # {".": 0, "a": 1, etc}
    # index to str
    itos = {s:i for i,s in stoi.items()}
    # {0: ".", 1: "a", etc}
    return chars, chars_from_name, itos, n, stoi


@app.cell
def _():
    # Now we build the dataset.  Below the architecture we are targeting ([link](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf))
    return


@app.cell
def _(stoi, torch):
    context_length = 3

    def build_dataset(words: list[str], context_length: int = 3):

        X = []
        Y = []

        for w in words:
            context = [0]*context_length
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
    return build_dataset, context_length


@app.cell
def _(build_dataset, words):
    #X,Y = build_dataset()

    # splitting train-test datasets -  we could have used sklearn
    n1 = int(len(words)*0.8)
    n2 = int(len(words)*0.9)
    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])
    return Xdev, Xte, Xtr, Ydev, Yte, Ytr, n1, n2


@app.cell
def _(Xdev, Xte, Xtr):
    len(Xtr),len(Xdev),len(Xte)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        |From the above, we can see that we are building a context window of integers (that correspond to chars) that will be used to predict the target (again given in int, but also associated with a char.) For example,

        [0,0,0] -> e (first letter of emma)
        [e,m,m] -> a (final letter of emma)
        [m,m,a] -> . (after emma ends)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""![image.png](https://i.postimg.cc/Z5XhDGjn/image.png)](https://postimg.cc/21dJ1XsN)""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Now we proceed to build the Neural net, as illustrated above.
        Basic idea is to have one matrix C per character used in the context length, and fully connext those to neurons in the tanh layer, which finally go to a softmax and then are used to predict the target char.
        """
    )
    return


@app.cell
def _(torch):
    # g = torch.Generator().manual_seed(2147483647) # for reproducibility
    # C = torch.randn((27, 10), generator=g)
    # W1 = torch.randn((30, 200), generator=g)
    # b1 = torch.randn(200, generator=g)
    # W2 = torch.randn((200, 27), generator=g)
    # b2 = torch.randn(27, generator=g)
    # parameters = [C, W1, b1, W2, b2]

    g = torch.Generator().manual_seed(2147483647) # for reproducibility
    C = torch.randn((27, 10), generator=g)
    W1 = torch.randn((30, 200), generator=g)
    b1 = torch.randn(200, generator=g)
    W2 = torch.randn((200, 27), generator=g)
    b2 = torch.randn(27, generator=g)
    parameters = [C, W1, b1, W2, b2]

    for param3 in parameters:
        param3.requires_grad = True
    return C, W1, W2, b1, b2, g, param3, parameters


@app.cell
def _(mo):
    mo.md(r"""In essence, C acts as a table where you can "look up" the 10-dimensional feature vector (the embedding) associated with each of the 27 possible characters in your vocabulary. When an integer representing a character is fed into the model, this matrix is used to retrieve its corresponding 10-dimensional embedding vector, which then serves as input to the subsequent layers of the neural network""")
    return


@app.cell
def _(parameters):
    sum(p.nelement() for p in parameters) # number of parameters in total
    return


@app.cell
def _(parameters):
    for p in parameters:
      p.requires_grad = True
    return (p,)


@app.cell
def _(C, F, W1, W2, Xtr, Ytr, b1, b2, parameters, torch):
    print1 = False
    stepi = []
    lossi = []
    for i in range(5000):

        # mini batch
        ix = torch.randint(0, Xtr.shape[0], (32,))
        #ix = [i]
    
        # forward pass
        emb = C[Xtr[ix]]
        mult = emb.view(-1,30) @ W1 + b1
        h = torch.tanh(mult)
        logits = h@W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])
    
        for param in parameters:
            param.grad = None

        loss.backward()

        # update grads
        lr = 0.1
        for param2 in parameters:
            param2.data += -lr * param2.grad
    
        stepi.append(i)
        lossi.append(loss.log10().item())

        if print1 and i % 10:
            print('Xtr', Xtr[ix])
            print('emb', emb.shape)
            print('mult', mult.shape)
            print('h', h.shape)
            print('logits', logits.shape)
            print('Ytr', Ytr.shape)
            print('loss', loss)
        

    return (
        emb,
        h,
        i,
        ix,
        logits,
        loss,
        lossi,
        lr,
        mult,
        param,
        param2,
        print1,
        stepi,
    )


@app.cell
def _(lossi, plt):
    #plt.yscale("log")
    plt.plot(lossi)
    #print('a')
    return


@app.cell
def _(lossi):
    lossi[-1]
    return


@app.cell
def _(C, F, W1, W2, b1, b2, torch):
    # training loss -> calculate forward pass
    # forward pass
    def calculate_loss(X, Y):
        emb1 = C[X]
        mult1 = emb1.view(-1,30) @ W1 + b1
        h1 = torch.tanh(mult1)
        logits1 = h1@W2 + b2
        loss1 = F.cross_entropy(logits1, Y)
        #print(loss1)
        return loss1.item()
    return (calculate_loss,)


@app.cell
def _(Xdev, Xte, Xtr, Ydev, Yte, Ytr, calculate_loss):
    print('loss train', calculate_loss(Xtr, Ytr))
    print('loss val', calculate_loss(Xdev, Ydev))
    print('loss test', calculate_loss(Xte, Yte))
    return


@app.cell
def _(C, itos, plt):
    # visualize embeddings
    plt.figure(figsize=(8,8))
    plt.scatter(C[:,0].data, C[:,1].data, s=200)
    for ii in range(C.shape[0]):
        plt.text(C[ii,0].item(), C[ii,1].item(), itos[ii], ha="center", va="center", color='white')
    plt.grid('minor')
    plt.show()
    return (ii,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
