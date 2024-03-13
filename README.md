# ServalShell
```
/content# ./servalshell.sh
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
                                          
ServalShell:~$ 
```

# how to train 

# how to run 
1. **Install packages**   
```sh
$ pip install -r requirements.txt
```
2. **Create kaggle token**  
Download the kaggle token and put it in the ServalShell folder
```json
{"username":"?????","key":"????????????????????????????"}
```
3. **Download pretrained weights**  
```sh
$ mkdir -p ~/.kaggle
$ cp kaggle.json ~/.kaggle/
$ chmod 600 ~/.kaggle/kaggle.json
$ kaggle datasets download -d sj2129tommy/nlc2bash-21epoch
$ unzip  -qq /content/ServalShell/nlc2bash-21epoch.zip
```
4. **run servalshell**  
```sh
$ ./servalshell.sh
```

_______________________________________________________________________________________


$$Mask_{target} = 
\begin{pmatrix}
? & 0 & 0 & 0 & \cdots & 0 \\
? & ? & 0 & 0 & \cdots & 0 \\ 
? & ? & ? & 0 & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\   
? & ? & ? & ? & \cdots & ?
\end{pmatrix}$$

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


## Attention is all you need
## shell 
https://github.com/f0lio/minishell?tab=readme-ov-file  
https://hackernoon.com/search?query=shell    
os : https://austinhenley.com/blog/challengingprojects.html  


# Citation
If you used Tellina in your work, please cite
```
@techreport{LinWPVZE2017:TR, 
  author = {Xi Victoria Lin and Chenglong Wang and Deric Pang and Kevin Vu and Luke Zettlemoyer and Michael D. Ernst}, 
  title = {Program synthesis from natural language using recurrent neural networks}, 
  institution = {University of Washington Department of Computer Science and Engineering}, 
  number = {UW-CSE-17-03-01}, 
  address = {Seattle, WA, USA}, 
  month = mar, 
  year = {2017} 
}
```
```
@inproceedings{LinWZE2018:NL2Bash, 
  author = {Xi Victoria Lin and Chenglong Wang and Luke Zettlemoyer and Michael D. Ernst}, 
  title = {NL2Bash: A Corpus and Semantic Parser for Natural Language Interface to the Linux Operating System}, 
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources
               and Evaluation {LREC} 2018, Miyazaki (Japan), 7-12 May, 2018.},
  year = {2018} 
}
```


