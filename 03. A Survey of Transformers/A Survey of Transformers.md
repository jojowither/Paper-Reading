# A Survey of Transformers
本篇探討在vanilla transformer提出之後各種transformer的改進

## Vanilla Transformer


Vanilla Transformer是目前NLP領域中幾乎佔主導的深度學習基本網路架構(相對於LSTM, CNN而言)，各種pretrained model幾乎都是由Transformer所組成，即便Vanilla Transformer取得了許多成功，但為人詬病的點仍然很明顯，就是他的計算與空間複雜度，所以許多針對改進Vanilla Transformer的論文也不斷出現，而這些Transformers也同樣成為新的pretrained model的核心架構。

先來聊聊Vanilla Transformer的複雜度吧

首先self-attention的計算複雜度為 $O(hdn^2)$，$h$為attention的head數，$d$為query和key的維度，$n$為句子長度， dot-product 公式為
$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V $$

由以上公式再乘上head數就可以推出計算複雜度。

同樣的空間複雜度也受到句子長度影響，為$O(hdn+hn^2)$，前者為存query和key所需的空間，後者為每個head產生的attention matrix。

我們以BERT-Base為例計算一下複雜度，BERT-Base句子長度為512，hidden size 768，12 heads，所以每個head維度為64 (768 / 12)。在這設定下，393216 floats(12 heads * 64 head size * 512 sequence length) = 393216 * 4 bytes = 1572864 bytes = 1572864/1024/1024 MB = 1.5 MB，1.5MB為query和key所需的空間。每個head產生的attention matrix所需空間為3145728 floats (12 * 512 * 512)～12MB，幾乎為前者的10倍大。BERT-Base為12層，所以每個example所需的memory接近(12+1.5)*12 = 162MB，當句子長度為1024時，$(12\times(1024^2)\times4/1024/1024+12\times64\times1024\times4/1024/1024)\times12 = 612$ MB，需要那麼多memory。

這也意味著在訓練時，我們只能用更小的batch size以及較差的平行運算能力，這也導致模型對長文本的推理能力較差。




