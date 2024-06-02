# ServalShell

```
/content# ./servalshell.sh
Bashlint grammar set up (124 utilities)

2024-03-15 11:44:52.217874: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-03-15 11:44:52.217925: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-03-15 11:44:52.219259: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-03-15 11:44:52.226184: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-03-15 11:44:53.252487: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
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


ServalShell:~$ print current user name
translated bash: whoami
root

ServalShell:~$ ▯ 
```
**ServalShell** is an nlc2bash program created by learning the Transformer model proposed in **"Attention Is All You Need"** 2017 research paper. I implemented the Transformer model myself by referring to the **paper** and **Umar Jamil's video**.
For model learning, refer to the **"NL2CMD: An Updated Workflow for Natural
Language to Bash Commands Translation"** paper. **Tellina-Tool** was used for pre-processing and post-processing, and **Hugging Face** was used for tokenizer and dataset.


# How to train 
1. **Install packages**   
```sh
$ pip install -r requirements.txt
```
2. **Data preprocessing**
```sh
$ python3 preprocess.py
```
3. **Set config.py values and Train**
```sh
$ python3 train.py
```

# How to run 
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
4. **Run servalshell**  
```sh
$ ./servalshell.sh
```

# Options 
|option|description|
|-|-|
|-d [cmd] , <br> --direct [cmd]|Execute bash command directly|
|-r [nl] , <br> --recommend [nl]|Even if the command execution is successful, <br> Recommended Command Structure is displayed|
|-h, --help|Describes usage and options|
|-q, --quit|Quit Servalshell| 

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

<img src="https://github.com/em-1001/ServalShell/blob/master/Data/servalcat.jpg">

