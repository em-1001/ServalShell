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
        recommend_flag = 0

        nl = input(prompt)

        nl_split = nl.split(' ')

        # Execute bash command directly
        if nl_split[0] == '-d' or nl_split[0] == '--direct':
          direct_bash = ' '.join(nl_split[1:])
          try:
            output = subprocess.check_output(direct_bash, shell=True, text=True)
            print(output)
          except subprocess.CalledProcessError as e:
            print(e)
          continue

        # Describes usage and options
        elif nl_split[0] == '-h' or nl_split[0] == '--help':
          print("If you enter a command in natural language, the program automatically translates it into a bash command and executes it.")
          print("If execution fails because the bash command translated by the model is incorrect, It will recommend several command structures.")
          print("Additionally, the following options are available.\n")
          print("-d [cmd],  --direct [cmd]              Execute bash command directly")
          print("-r [nl],  --recommend [nl]             Even if the command execution is successful, Recommended Command Structure is displayed")
          print("-h,  --help                            Describes usage and options")
          print("-q,  --quit                            Quit Servalshell")
          print("")
          continue
        
        # Even if the command execution is successful, Recommended Command Structure is displayed
        elif nl_split[0] == '-r' or nl_split[0] == '--recommend':
            recommend_flag = 1
            nl = ' '.join(nl_split[1:])

        # Quit Servalshell
        elif nl_split[0] == '-q' or nl_split[0] == '--quit':
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
            print("")
            continue

        print("translated bash: " + str(bash_list[0]))
        try:
            output = subprocess.check_output(bash_list[0], shell=True, text=True)
            print(output)

            if recommend_flag == 1:
                print("\033[91m" + "Recommended Command Structure" + "\033[30m")
                for rcs in _bash:
                    print(rcs)
                print("")

        except subprocess.CalledProcessError as e:
            print("\033[91m" + "\nRecommended Command Structure" + "\033[30m")
            for rcs in _bash:
                print(rcs)
            print("")


if __name__ == '__main__':
    servalshell()
