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
                                          
ServalShell:~$ Add variable 'v' with value '5' to a temporary environment, list this environment using 'less' to interactively view it.
translated bash: env | uniq | sort -n
CGROUP_MEMORY_EVENTS=/sys/fs/cgroup/memory.events /var/colab/cgroup/jupyter-children/memory.events
CLICOLOR=1
CLOUDSDK_CONFIG=/content/.config
CLOUDSDK_PYTHON=python3
COLAB_BACKEND_VERSION=next
COLAB_DEBUG_ADAPTER_MUX_PATH=/usr/local/bin/dap_multiplexer
COLAB_FILE_HANDLER_ADDR=localhost:3453
COLAB_GPU=1
COLAB_JUPYTER_IP=172.28.0.12
COLAB_JUPYTER_TOKEN=
COLAB_JUPYTER_TRANSPORT=ipc
COLAB_KERNEL_MANAGER_PROXY_HOST=172.28.0.12
COLAB_KERNEL_MANAGER_PROXY_PORT=6000
...

ServalShell:~$ print current user name
translated bash: whoami
root
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


