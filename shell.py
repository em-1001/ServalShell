"""해얄것
1. echo "~" 개선하기 위해 휴리스틱 실패 했을 시 다른 함수 써보기
2. 성능 테스트 더 해보기
3. cd 문제 해결
4. 결과가 없을 경우 RCS(recommended command structure) 보여주기
"""

import os
import subprocess
from translate import translate
import sys
sys.path.append("./Tellina")
from bashlint.data_tools import bash_tokenizer, bash_parser, ast2tokens, ast2command
from nlp_tools import tokenizer
from bashlint import data_tools
from encoder_decoder import slot_filling

import warnings
warnings.filterwarnings("ignore")


def servalshell():
    cat = """
       _
       \`*-.
        )  _`-.
       .  : `. .
       : _   '  \
       ; *` _.   `*-._
       `-.-'          `-.
         ;       `       `.
         :.       .        \
         . \  .   :   .-'   .
         '  `+.;  ;  '      :
         :  '  |    ;       ;-.
         ; '   : :`-:     _.`* ;
      .*' /  .*' ; .*`- +'  `*'
      `*-*   `*-*  `*-*'
    """
    print(cat)
    prompt = '\033[35m' + 'ServalShell' + '\033[30m' + ':~$ '
    while True:
        nl = input(prompt)
        nl_preprocess = ' '.join(tokenizer.ner_tokenizer(nl)[0])
        _bash = translate(nl_preprocess)

        node = bash_parser(_bash)
        _, _, nl_filler= tokenizer.ner_tokenizer(nl)[1]
        slot_filling.heuristic_slot_filling(node, nl_filler)

        bash = ast2command(node)
        bash2 = slot_filling.stupid_slot_matching(nl, _bash)

        if _bash == bash:
            bash = bash2
            #print("bash command: " + str(bash2))
        #else:
            #print("bash command: " + str(bash))

        try:
            output = subprocess.check_output(bash, shell=True, text=True)
            print(output)
        except subprocess.CalledProcessError as e:
            print("\033[31m" + "recommended command structure" + "\033[30m")
            print(_bash)
            print("\033[31m" + "recommended command" + "\033[30m")
            print("1. " + str(bash))
            print("2. " + str(bash2))

# servalshell()


servalshell()
