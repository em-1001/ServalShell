# CompAiler
## minishell 
https://github.com/f0lio/minishell?tab=readme-ov-file  
https://hackernoon.com/search?query=shell    
os : https://austinhenley.com/blog/challengingprojects.html  

## compiler 
https://www.youtube.com/watch?v=WY3clwTLEXc&list=LL&index=1&t=212s

## CompAiler
eng -> shell : https://github.com/emcf/engshell

https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1224/reports/custom_116997097.pdf

https://paperswithcode.com/search?q_meta=&q_type=&q=NLC2CMD

DATASET? : https://ibm.ent.box.com/v/nl2bash-data  
https://github.com/magnumresearchgroup/bash_gen/tree/master  

`CompAiler:~$ download a picture of a galaxy...`

파이썬으로 bash 명령어 실행 : https://blankspace-dev.tistory.com/359

tokenizer : https://huggingface.co/docs/transformers/tokenizer_summary

load_dataset : https://huggingface.co/docs/datasets/loading -> json부분 읽어보기

load_dataset할 때 json으로 가져와서 원래 영상코드로 가져온 데이터셋이랑 형식이 어떻게 다른지 확인해야됨.  
get_all_sentences의 인자로 load_dataset의 결과가 들어가는데 원래 영상 코드처럼 주어진 lang의 문장만 내뱉을 수 있도록 만들어야 한다.

def get_all_sentences(ds, lang):
    for item in ds.values():
        yield item[lang]








