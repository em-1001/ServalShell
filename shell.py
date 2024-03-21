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


def post_processing(nl, _bash):
    node = bash_parser(_bash)
    _, _, nl_filler= tokenizer.ner_tokenizer(nl)[1]
    slot_filling.heuristic_slot_filling(node, nl_filler)

    bash = ast2command(node)

    bash2 = slot_filling.stupid_slot_matching(nl, _bash)

    if _bash == bash:
        bash = bash2

    return bash


def servalshell():
    serval_cat = """
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
    print(serval_cat)
    prompt = '\033[92m' + 'ServalShell' + '\033[30m' + ':~$ '
    while True:
        nl = input(prompt)

        nl_split = nl.split(' ')

        # directly enter bash commands: -d `command`
        if nl_split[0] == '-d':
          direct_bash = nl_split[1:]
          try:
            output = subprocess.check_output(direct_bash, shell=True, text=True)
            print(output)
          except subprocess.CalledProcessError as e:
            print(e)
          continue

        # quit the ServalShell
        elif nl_split[0] == '-q':
          break
            
        nl_preprocess = ' '.join(tokenizer.ner_tokenizer(nl)[0])
        _bash = translate(nl_preprocess)

        bash_list = []
        for cmd in _bash:
            try:
                bash = post_processing(nl, cmd)
                bash_list.append(bash)
            except:
                continue
            
        if len(bash_list) == 0:
            print("Failed at Abstract syntax tree...")
            print("\n")
            continue

        print("translated bash: " + str(bash_list[0]))
        try:
            output = subprocess.check_output(bash_list[0], shell=True, text=True)
            print(output)
        except subprocess.CalledProcessError as e:
            print("\033[91m" + "\nRecommended Command Structure" + "\033[30m")
            for rcs in _bash:
                print(rcs)
            print("\n")


if __name__ == '__main__':
    servalshell()
