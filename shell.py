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
    prompt = '\033[92m' + 'ServalShell' + '\033[30m' + ':~$ '
    while True:
        nl = input(prompt)
        print("compiling...")
        nl_preprocess = ' '.join(tokenizer.ner_tokenizer(nl)[0])
        _bash = translate(nl_preprocess)

        node = bash_parser(_bash)
        _, _, nl_filler= tokenizer.ner_tokenizer(nl)[1]
        slot_filling.heuristic_slot_filling(node, nl_filler)

        try:
            bash = ast2command(node)
        except:
            print("False", end="\n\n")
            continue    

        bash2 = slot_filling.stupid_slot_matching(nl, _bash)

        if _bash == bash:
            bash = bash2
            print("translated bash: " + str(bash2))
        else:
            print("translated bash: " + str(bash))

        try:
            output = subprocess.check_output(bash, shell=True, text=True)
            print(output)
        except subprocess.CalledProcessError as e:
            print("\nrecommended command structure")
            print("-> " + str(_bash), end="\n\n")
            print("recommended command")
            print("-> " + str(bash))
            print("-> " + str(bash2))
            print("")

servalshell()
