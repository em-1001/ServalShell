"""해얄것
1. echo "~" 개선하기 위해 휴리스틱 실패 했을 시 다른 함수 써보기
2. 성능 테스트 더 해보기
3. cd 문제 해결
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
        bash = translate(nl_preprocess)

        node = bash_parser(bash)
        _, _, nl_filler= tokenizer.ner_tokenizer(nl)[1]
        slot_filling.heuristic_slot_filling(node, nl_filler)

        bash = ast2command(node)
        print("command: " + str(bash))

        if bash.split()[0] == "cd":
            os.chdir("kernel")
            #try:
            #    os.chdir("kernel")
            #    continue
            #except:
            #    print(str(bash.split()[1]))
            #    continue

        try:
            output = subprocess.check_output(bash, shell=True, text=True)
            print(output, end='')
        except subprocess.CalledProcessError as e:
            print("\033[36m" + "recommended command structure" + "\033[30m")
            print(bash, end='')

# servalshell()


servalshell()
