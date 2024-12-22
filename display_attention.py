# reference : https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb

import sys
sys.path.append("./Tellina")
from nlp_tools import tokenizer
from translate import translate
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


nl = 'copies "file.txt" to "null.txt"'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])

bash, attention_score = translate(nl)


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):

    sentence_list = sentence.split(' ')
    translation_list = translation.split(' ')

    assert n_rows * n_cols == n_heads

    # 출력할 그림 크기 조절
    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        # 어텐션(Attention) 스코어 확률 값을 이용해 그리기
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['<sos>'] + sentence_list + ['<eos>'], rotation=45)
        ax.set_yticklabels([''] + translation_list + ['<eos>'])

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

display_attention(nl, bash[0], attention_score[0])
