Using device cuda
Max length of source sentence: 56
Max length of target sentence: 62
[0.0001]
Processing epoch 00: 100%|██████████| 1164/1164 [01:16<00:00, 15.16it/s, loss=2.596]
--------------------------------------------------------------------------------
    SOURCE: search for first match of regex _REGEX in all _FILE file under current directori and print file name
    TARGET: find Path -name Regex -exec awk Program {} \;
 PREDICTED: find Path -name Regex | xargs -I {} grep Regex {}
--------------------------------------------------------------------------------
    SOURCE: print which file differ in _REGEX and _REGEX recurs and sort output
    TARGET: diff -q -r File File | sort
 PREDICTED: cat File | sed Program
--------------------------------------------------------------------------------
[9.998286624877786e-05]
Processing epoch 01: 100%|██████████| 1164/1164 [01:21<00:00, 14.21it/s, loss=1.733]
--------------------------------------------------------------------------------
    SOURCE: chang ownership of all regular file in current directori
    TARGET: find Path -type f -exec chown Regex {} \;
 PREDICTED: find Path -type f | xargs -I {} chown Regex {}
--------------------------------------------------------------------------------
    SOURCE: print each line in _REGEX and _REGEX separ by a space
    TARGET: paste -d Regex File File | sed Program
 PREDICTED: echo Regex | sed Program | column -t
--------------------------------------------------------------------------------
[9.99314767377287e-05]
Processing epoch 02: 100%|██████████| 1164/1164 [01:18<00:00, 14.80it/s, loss=2.167]
--------------------------------------------------------------------------------
    SOURCE: find file associ with an inod
    TARGET: find Path -inum Quantity -exec ls -l {} \;
 PREDICTED: find Path -inum Quantity -exec ls -l {} \;
--------------------------------------------------------------------------------
    SOURCE: print all filenam in _FILE except for those that are of form _REGEX or _FILE
    TARGET: find Path -not \( -name Regex -or -name Regex \) {} Path -print
 PREDICTED: find Path -name Regex | grep -v Regex | grep -v Regex | cut -d Regex -f Number | rev
--------------------------------------------------------------------------------
[9.984586668665642e-05]
Processing epoch 03: 100%|██████████| 1164/1164 [01:17<00:00, 14.98it/s, loss=1.988]
--------------------------------------------------------------------------------
    SOURCE: display all regular file in folder image-fold
    TARGET: find Path -type f
 PREDICTED: find Path -type f
--------------------------------------------------------------------------------
    SOURCE: find all _FILE directori under _FILE directori
    TARGET: find Path -type d \( -name Regex -or -name Regex -or -name Regex \) -prune -or -name Regex -print
 PREDICTED: find Path -type d -name Regex
--------------------------------------------------------------------------------
[9.972609476841367e-05]
Processing epoch 04: 100%|██████████| 1164/1164 [01:15<00:00, 15.42it/s, loss=1.644]
--------------------------------------------------------------------------------
    SOURCE: find all file under _REGEX not match regex _FILE and execut hashcod on each of them with file path as it argument
    TARGET: find Path -type f -not -regex Regex -exec  \;
 PREDICTED: find Path -type f -name Regex -exec file {} \; -exec echo {} \; -exec echo {} \; -exec echo {} \; | grep Regex
--------------------------------------------------------------------------------
    SOURCE: search system for file and directori own by user _REGEX
    TARGET: find Path -user Regex -print
 PREDICTED: find Path -user Regex -name Regex
--------------------------------------------------------------------------------
[9.957224306869053e-05]
Processing epoch 05: 100%|██████████| 1164/1164 [01:14<00:00, 15.60it/s, loss=1.479]
--------------------------------------------------------------------------------
    SOURCE: find file and directori in entir file system that were access in less than _TIMESPAN ago
    TARGET: find Path -atime -Timespan
 PREDICTED: find Path -atime Timespan
--------------------------------------------------------------------------------
    SOURCE: print shell option _REGEX with indic of it status
    TARGET: shopt -p globstar
 PREDICTED: shopt -s nullglob
--------------------------------------------------------------------------------
[9.93844170297569e-05]
Processing epoch 06: 100%|██████████| 1164/1164 [01:14<00:00, 15.57it/s, loss=1.528]
--------------------------------------------------------------------------------
    SOURCE: make directori to file _FILE as need
    TARGET: mkdir -p $( dirname Regex )
 PREDICTED: mkdir Directory Directory Directory Directory Directory
--------------------------------------------------------------------------------
    SOURCE: find and copi all _FILE file in current directori tree to _FILE
    TARGET: find Path -name Regex -print0 | xargs -I {} -0 cp -v {} File
 PREDICTED: find Path -name Regex | xargs -I {} cp -a --target-directory Directory --parents {}
--------------------------------------------------------------------------------
[9.916274537819775e-05]
Processing epoch 07: 100%|██████████| 1164/1164 [01:15<00:00, 15.46it/s, loss=1.580]
--------------------------------------------------------------------------------
    SOURCE: print which file differ in _DIRECTORY and _DIRECTORY recurs exclud ani file that match ani pattern in _REGEX
    TARGET: diff File File -r -q -X File
 PREDICTED: diff -r -q File File File
--------------------------------------------------------------------------------
    SOURCE: print shell option _REGEX with indic of it status
    TARGET: shopt -p globstar
 PREDICTED: shopt -s extglob
--------------------------------------------------------------------------------
[9.890738003669029e-05]
Processing epoch 08: 100%|██████████| 1164/1164 [01:15<00:00, 15.41it/s, loss=1.501]
--------------------------------------------------------------------------------
    SOURCE: format file _FILE as new-lin separ column
    TARGET: column -t -s Regex File
 PREDICTED: column -t -s Regex File
--------------------------------------------------------------------------------
    SOURCE: search current directori tree for file with extens _REGEX and remov them if are more than _TIMESPAN old
    TARGET: find Path -name Regex -ctime +Timespan -exec rm {} \;
 PREDICTED: find Path -mtime +Timespan -type d -exec rm -f {} \;
--------------------------------------------------------------------------------
[9.861849601988383e-05]
Processing epoch 09: 100%|██████████| 1164/1164 [01:15<00:00, 15.48it/s, loss=1.549]
--------------------------------------------------------------------------------
    SOURCE: find all file in file system whose size is bigger than _SIZE
    TARGET: find Path -size +Size
 PREDICTED: find Path -size +Size
--------------------------------------------------------------------------------
    SOURCE: print out name and type of all file in current directori tree
    TARGET: find Path -printf "%y %p\n"
 PREDICTED: find Path -exec echo {} \;
--------------------------------------------------------------------------------
[9.82962913144534e-05]
Processing epoch 10: 100%|██████████| 1164/1164 [01:15<00:00, 15.42it/s, loss=1.508]
--------------------------------------------------------------------------------
    SOURCE: print comma separ gap in file _REGEX that contain new line separ order number
    TARGET: seq $( tail - Quantity File ) | diff File File | grep -P -o Regex
 PREDICTED: cat File | paste -d Regex File File
--------------------------------------------------------------------------------
    SOURCE: find all file and directori that are own by user _REGEX and are newer than _REGEX by modif time in entir filesystem
    TARGET: find Path -newer File -user Regex -print
 PREDICTED: find Path -newer File
--------------------------------------------------------------------------------
[9.794098674340965e-05]
Processing epoch 11: 100%|██████████| 1164/1164 [01:15<00:00, 15.36it/s, loss=1.459]
--------------------------------------------------------------------------------
    SOURCE: delet all file with inod number _REGEX under current directori tree
    TARGET: find Path -inum Quantity -exec rm {} \;
 PREDICTED: find Path -inum Quantity -exec rm -i {} \;
--------------------------------------------------------------------------------
    SOURCE: find all regular file in _FILE which been access in _TIMESPAN
    TARGET: find Path -amin -Quantity -type f
 PREDICTED: find Path -type f -atime -Timespan
--------------------------------------------------------------------------------
[9.755282581475767e-05]
Processing epoch 12: 100%|██████████| 1164/1164 [01:15<00:00, 15.34it/s, loss=1.200]
--------------------------------------------------------------------------------
    SOURCE: find all file with extens _FILE in _FILE directori tree
    TARGET: find Path -type f -name Regex
 PREDICTED: find Path -name Regex
--------------------------------------------------------------------------------
    SOURCE: creat an empti file _REGEX in each directori under current directori contain a file name _REGEX
    TARGET: find Path -name Regex -execdir touch File \;
 PREDICTED: find Path -type f -name Regex -exec touch File \;
--------------------------------------------------------------------------------
[9.713207455460892e-05]
Processing epoch 13: 100%|██████████| 1164/1164 [01:16<00:00, 15.31it/s, loss=1.459]
--------------------------------------------------------------------------------
    SOURCE: execut _REGEX everi _TIMESPAN
    TARGET: watch -n Quantity du -s File
 PREDICTED: watch ls -l
--------------------------------------------------------------------------------
    SOURCE: recurs add _FILE to all file without an extens in directori tree _FILE
    TARGET: find Path -type f -not -name Regex -exec mv {} File \;
 PREDICTED: find Path -type f -name Regex -exec mv {} File \;
--------------------------------------------------------------------------------
[9.667902132486008e-05]
Processing epoch 14: 100%|██████████| 1164/1164 [01:16<00:00, 15.16it/s, loss=1.418]
--------------------------------------------------------------------------------
    SOURCE: delet all _REGEX directori under maximum _NUMBER level down current directori tree
    TARGET: find Path -maxdepth Quantity -type d -name Regex -print0 | xargs -0 -I {} rm -r -f {}
 PREDICTED: find Path -maxdepth Quantity -type d -name Regex -delete
--------------------------------------------------------------------------------
    SOURCE: find all file under current directori and replac match of regex _FILE with _REGEX in everi line of output
    TARGET: find Path -type f -print | sed Program
 PREDICTED: find Path -type f -exec sed -i Program {} \;
--------------------------------------------------------------------------------
[9.619397662556433e-05]
Processing epoch 15: 100%|██████████| 1164/1164 [01:15<00:00, 15.36it/s, loss=1.482]
--------------------------------------------------------------------------------
    SOURCE: recurs search current directori for all file with name end with _FILE chang to _FILE
    TARGET: find Path -name Regex -exec rename Regex {} \;
 PREDICTED: find Path -name Regex -exec chown -R Regex {} \;
--------------------------------------------------------------------------------
    SOURCE: set variabl _REGEX to base name of first argument to script or function that is part follow last slash
    TARGET: basename Regex
 PREDICTED: basename Regex
--------------------------------------------------------------------------------
[9.567727288213003e-05]
Processing epoch 16: 100%|██████████| 1164/1164 [01:15<00:00, 15.44it/s, loss=1.310]
--------------------------------------------------------------------------------
    SOURCE: remov all core file in file system
    TARGET: find Path -name Regex | xargs -I {} rm {}
 PREDICTED: find Path -name Regex -exec rm {} \;
--------------------------------------------------------------------------------
    SOURCE: find all _FILE file under build direcotri except _FILE and _FILE directori
    TARGET: find Path -not \( -path Regex -prune \) -not \( -path Regex -prune \) -name Regex
 PREDICTED: find Path -name Regex -prune -or -name Regex -print
--------------------------------------------------------------------------------
[9.512926421749304e-05]
Processing epoch 17: 100%|██████████| 1164/1164 [01:15<00:00, 15.45it/s, loss=1.434]
--------------------------------------------------------------------------------
    SOURCE: compress everi file in current directori tree that match _FILE and keep origin file
    TARGET: find Path -type f -name Regex -exec gzip -k {} \;
 PREDICTED: gzip -k -r Regex File
--------------------------------------------------------------------------------
    SOURCE: show who is _FILE on
    TARGET: who
 PREDICTED: who Regex
--------------------------------------------------------------------------------
[9.455032620941839e-05]
Processing epoch 18: 100%|██████████| 1164/1164 [01:15<00:00, 15.44it/s, loss=1.448]
--------------------------------------------------------------------------------
    SOURCE: recurs find and compress all file in a current folder
    TARGET: find Path -type f -exec bzip2 {} +
 PREDICTED: find Path -type f -exec bzip2 {} \;
--------------------------------------------------------------------------------
    SOURCE: find all file in home folder which been modifi exact _TIMESPAN befor
    TARGET: find Path -mtime Timespan -daystart
 PREDICTED: find Path -mtime Timespan
--------------------------------------------------------------------------------
[9.394085563309827e-05]
Processing epoch 19: 100%|██████████| 1164/1164 [01:15<00:00, 15.33it/s, loss=1.389]
--------------------------------------------------------------------------------
    SOURCE: find all _FILE file under current directori and list path with name
    TARGET: find Path -iname Regex -printf "%p %f\n"
 PREDICTED: find Path -type f -name Regex
--------------------------------------------------------------------------------
    SOURCE: find command will list of all file _REGEX directori from current directori befor list echo command will display _REGEX
    TARGET: find Path -exec echo Regex {} \;
 PREDICTED: find Path -type f -exec echo Regex Regex Regex {} \; -or -exec echo {} \;
--------------------------------------------------------------------------------
[9.330127018922194e-05]
Processing epoch 20: 100%|██████████| 1164/1164 [01:16<00:00, 15.31it/s, loss=1.345]
--------------------------------------------------------------------------------
    SOURCE: display all regular file in current directori
    TARGET: find Path -type f -print0
 PREDICTED: find Path -type f
--------------------------------------------------------------------------------
    SOURCE: verbos compress all file on third and fourth depth level keep origin file in place
    TARGET: bzip2 -k -v File
 PREDICTED: bzip2 -k -v File
--------------------------------------------------------------------------------
[9.263200821770461e-05]
Processing epoch 21: 100%|██████████| 1164/1164 [01:16<00:00, 15.24it/s, loss=1.354]
--------------------------------------------------------------------------------
    SOURCE: make all bugzilla subdirectori permiss _NUMBER
    TARGET: find Path -type d -exec chmod Permission {} \;
 PREDICTED: find Path -type d -exec chmod Permission {} \;
--------------------------------------------------------------------------------
    SOURCE: print a minim set of differ between file in directori _REGEX and _REGEX ignor first _NUMBER line of output and print ani line start with _REGEX with first charact remov
    TARGET: diff File File File | tail -n +Quantity | grep Regex | cut -c Number
 PREDICTED: diff File File | grep Regex | sed Program
--------------------------------------------------------------------------------
[9.19335283972712e-05]
Processing epoch 22: 100%|██████████| 1164/1164 [01:16<00:00, 15.27it/s, loss=1.209]
--------------------------------------------------------------------------------
    SOURCE: find file and directori under current directori that match regex _FILE in path
    TARGET: find Path | grep -q -i Regex
 PREDICTED: find Path -regex Regex
--------------------------------------------------------------------------------
    SOURCE: remov filetyp suffix from filenam
    TARGET: echo Regex | rev | cut -f Number -d Regex | rev
 PREDICTED: echo Regex | rev | cut -d Regex -f Number | rev
--------------------------------------------------------------------------------
[9.120630943110079e-05]
Processing epoch 23: 100%|██████████| 1164/1164 [01:15<00:00, 15.35it/s, loss=1.310]
--------------------------------------------------------------------------------
    SOURCE: find all file and directori with _FILE extens that are less than _SIZE in size under _FILE directori tree
    TARGET: find Path -name Regex -size -Size
 PREDICTED: find Path -size +Size -name Regex
--------------------------------------------------------------------------------
    SOURCE: copi all file under current directori like _REGEX to _FILE directori
    TARGET: find Path -name Regex -print0 | xargs -0 -I {} cp -t Directory {}
 PREDICTED: find Path -name Regex | sed -e Program -e Program -e Program | xargs -I {} cp File {}
--------------------------------------------------------------------------------
[9.045084971874738e-05]
Processing epoch 24: 100%|██████████| 1164/1164 [01:16<00:00, 15.31it/s, loss=1.200]
--------------------------------------------------------------------------------
    SOURCE: search for all regular _FILE file in file system and move them to folder _FILE
    TARGET: find Path -iname Regex -type f -exec mv {} File \;
 PREDICTED: find Path -iname Regex -type f -exec mv {} File \;
--------------------------------------------------------------------------------
    SOURCE: remov all _FILE file in _FILE directori tree
    TARGET: find Path -type f -name Regex | tr Regex Regex | xargs -0 -I {} rm -f {}
 PREDICTED: find Path -type f -name Regex -print0 | xargs -0 -I {} rm -f {}
--------------------------------------------------------------------------------
[8.966766701456177e-05]
Processing epoch 25: 100%|██████████| 1164/1164 [01:16<00:00, 15.23it/s, loss=1.206]
--------------------------------------------------------------------------------
    SOURCE: print all ns server of domain _FILE
    TARGET: dig Regex Regex | awk Program
 PREDICTED: dig Regex Regex ns
--------------------------------------------------------------------------------
    SOURCE: append all _FILE file modifi within _TIMESPAN to tar archiv _FILE
    TARGET: find Path -name Regex Path Path -mtime -Timespan -print0 | xargs -0 -I {} tar -r -v -f File {}
 PREDICTED: find Path -mtime -Timespan -type f -exec tar -r -v -f File {} \;
--------------------------------------------------------------------------------
[8.885729807284856e-05]
Processing epoch 26: 100%|██████████| 1164/1164 [01:16<00:00, 15.29it/s, loss=1.182]
--------------------------------------------------------------------------------
    SOURCE: locat symlink in directori tree _REGEX and _FILE
    TARGET: find Path Path -type l
 PREDICTED: find Path -type l -lname Regex
--------------------------------------------------------------------------------
    SOURCE: find all file and directori in level _NUMBER down _REGEX directori with all posit paramet append with find command
    TARGET: echo Regex | xargs -I {} find {} -mindepth Quantity -maxdepth Quantity Path
 PREDICTED: find Path -maxdepth Quantity -name Regex
--------------------------------------------------------------------------------
[8.802029828000156e-05]
Processing epoch 27: 100%|██████████| 1164/1164 [01:17<00:00, 15.05it/s, loss=1.218]
--------------------------------------------------------------------------------
    SOURCE: monitor _NUMBER specif process id _NUMBER _REGEX and _NUMBER
    TARGET: top -p Regex -p Regex -p Regex
 PREDICTED: pstree -a Regex
--------------------------------------------------------------------------------
    SOURCE: find all directori under _FILE and chang permiss to _NUMBER
    TARGET: find Path -type d -exec chmod Permission {} +
 PREDICTED: find Path -type d -exec chmod Permission {} +
--------------------------------------------------------------------------------
[8.715724127386972e-05]
Processing epoch 28: 100%|██████████| 1164/1164 [01:16<00:00, 15.31it/s, loss=1.333]
--------------------------------------------------------------------------------
    SOURCE: connect to host _REGEX as ssh user _REGEX to copi remot file _FILE to current directori on local host
    TARGET: scp -v File
 PREDICTED: scp -v File File
--------------------------------------------------------------------------------
    SOURCE: remov all _FILE file in and below _FILE
    TARGET: find Path -name Regex | xargs -I {} rm {}
 PREDICTED: find Path -name Regex -print0 | xargs -0 -I {} rm {}
--------------------------------------------------------------------------------
[8.626871855061438e-05]
Processing epoch 29: 100%|██████████| 1164/1164 [01:16<00:00, 15.22it/s, loss=1.139]
--------------------------------------------------------------------------------
    SOURCE: find all file name _REGEX in current directori tree not descend into _REGEX directori
    TARGET: find Path -name Regex -prune
 PREDICTED: find Path -name Regex -prune -or -name Regex -print
--------------------------------------------------------------------------------
    SOURCE: print text file path that match _REGEX in content under _REGEX recurs
    TARGET: grep -r -l Regex File | tr Regex Regex | xargs -r -0 -I {} file {} | grep -e Regex | grep -v -e Regex
 PREDICTED: grep -r Regex File
--------------------------------------------------------------------------------
[8.535533905932738e-05]
Processing epoch 30: 100%|██████████| 1164/1164 [01:15<00:00, 15.34it/s, loss=1.141]
--------------------------------------------------------------------------------
    SOURCE: search current directori recurs for file last modifi within past _TIMESPAN ignor _FILE file and path _FILE and _FILE
    TARGET: find Path -mtime Timespan -not \( -name Regex -or -regex Regex -or -regex Regex \)
 PREDICTED: find Path -mtime Timespan | grep -v Regex | grep -v Regex
--------------------------------------------------------------------------------
    SOURCE: get disk space use by all _FILE file under _FILE directori
    TARGET: find Path -type f -name Regex -printf "%s\n" | awk Program
 PREDICTED: find Path -iname Regex -print0 | du --files0-from File -c -s | tail - Quantity
--------------------------------------------------------------------------------
[8.44177287846877e-05]
Processing epoch 31: 100%|██████████| 1164/1164 [01:15<00:00, 15.47it/s, loss=1.159]
--------------------------------------------------------------------------------
    SOURCE: remov all subdirectori name _FILE under current dir
    TARGET: find Path -type d -name Regex -exec rm -r {} \;
 PREDICTED: find Path -type d -name Regex -exec rm -r -f {} \;
--------------------------------------------------------------------------------
    SOURCE: chang owner to _REGEX and group to _REGEX of _REGEX
    TARGET: chown -- Regex File
 PREDICTED: chown Regex File
--------------------------------------------------------------------------------
[8.345653031794292e-05]
Processing epoch 32: 100%|██████████| 1164/1164 [01:15<00:00, 15.48it/s, loss=1.121]
--------------------------------------------------------------------------------
    SOURCE: find all file and directori that are own by user _REGEX and are newer than _REGEX by modif time in entir filesystem
    TARGET: find Path -newer File -user Regex -print
 PREDICTED: find Path -newer File
--------------------------------------------------------------------------------
    SOURCE: print file type of command _REGEX
    TARGET: file -L $( which Regex )
 PREDICTED: file $( which Regex )
--------------------------------------------------------------------------------
[8.247240241650919e-05]
Processing epoch 33: 100%|██████████| 1164/1164 [01:16<00:00, 15.30it/s, loss=1.256]
--------------------------------------------------------------------------------
    SOURCE: print a record for domain _REGEX from _FILE nameserv
    TARGET: dig Regex Regex a
 PREDICTED: dig Regex Regex a
--------------------------------------------------------------------------------
    SOURCE: list each uniqu charact in _REGEX prefix by number of occurr
    TARGET: grep -o Regex File | sort | uniq -c
 PREDICTED: grep -o Regex File | tr Regex Regex | sort | uniq -c | sort -n -r
--------------------------------------------------------------------------------
[8.146601955249188e-05]
Processing epoch 34: 100%|██████████| 1164/1164 [01:16<00:00, 15.28it/s, loss=1.131]
--------------------------------------------------------------------------------
    SOURCE: verbos compress all file on third and fourth depth level keep origin file in place
    TARGET: bzip2 -k -v File
 PREDICTED: bzip2 -k -v File
--------------------------------------------------------------------------------
    SOURCE: find file and directori under _FILE smaller than _SIZE
    TARGET: find Path -size -Size
 PREDICTED: find Path -size -Size
--------------------------------------------------------------------------------
[8.043807145043604e-05]
Processing epoch 35: 100%|██████████| 1164/1164 [01:16<00:00, 15.30it/s, loss=1.208]
--------------------------------------------------------------------------------
    SOURCE: chang permiss of _REGEX to _NUMBER
    TARGET: chmod Permission File
 PREDICTED: chmod Permission File
--------------------------------------------------------------------------------
    SOURCE: find all _FILE file under current directori and print content
    TARGET: find Path -name Regex -exec cat {} \;
 PREDICTED: cat $( find Path -name Regex )
--------------------------------------------------------------------------------
[7.938926261462367e-05]
Processing epoch 36: 100%|██████████| 1164/1164 [01:16<00:00, 15.22it/s, loss=1.179]
--------------------------------------------------------------------------------
    SOURCE: chang permiss of all regular file in current folder
    TARGET: find Path -type f -exec chmod Permission {} +
 PREDICTED: find Path -type f -exec chmod Permission {} \;
--------------------------------------------------------------------------------
    SOURCE: list all environ variabl whose name start with goroot
    TARGET: env | grep Regex
 PREDICTED: env | awk -F Regex Program
--------------------------------------------------------------------------------
[7.832031184624166e-05]
Processing epoch 37: 100%|██████████| 1164/1164 [01:16<00:00, 15.25it/s, loss=1.081]
--------------------------------------------------------------------------------
    SOURCE: print sort uniqu list of folder in compress archiv _FILE
    TARGET: tar -t -f File | xargs -I {} dirname {} | sort | uniq
 PREDICTED: find Path -type d | tar -x -z -v -f File | cut -f Number -d Regex
--------------------------------------------------------------------------------
    SOURCE: list all regular file match name pattern _REGEX under _FILE _FILE _FILE and _FILE directori tree
    TARGET: find Path Path Path Path -name Regex -type f -ls
 PREDICTED: find Path Path -name Regex -type f
--------------------------------------------------------------------------------
[7.723195175075138e-05]
Processing epoch 38: 100%|██████████| 1164/1164 [01:16<00:00, 15.30it/s, loss=1.154]
--------------------------------------------------------------------------------
    SOURCE: find all file under _FILE and redirect sort list to myfil
    TARGET: find Path -type f | sort | tee File
 PREDICTED: find Path -type f -name Regex | sort -t Regex -k Number
--------------------------------------------------------------------------------
    SOURCE: find disk use space of onli target directori
    TARGET: du --max-depth Quantity File
 PREDICTED: du -h File
--------------------------------------------------------------------------------
[7.612492823579746e-05]
Processing epoch 39: 100%|██████████| 1164/1164 [01:16<00:00, 15.30it/s, loss=1.089]
--------------------------------------------------------------------------------
    SOURCE: print line number of each match _REGEX in _REGEX
    TARGET: nl -b a File | grep Regex | awk Program
 PREDICTED: wc -l File
--------------------------------------------------------------------------------
    SOURCE: search for _REGEX in all _FILE file under current directori
    TARGET: find Path -name Regex | xargs -I {} grep -E Regex {}
 PREDICTED: find Path -name Regex | xargs -I {} grep -E Regex {}
--------------------------------------------------------------------------------
[7.500000000000002e-05]
Processing epoch 40: 100%|██████████| 1164/1164 [01:16<00:00, 15.18it/s, loss=1.148]
--------------------------------------------------------------------------------
    SOURCE: find all regular file in current folder and display total line in them
    TARGET: find Path -type f -print0 | xargs -0 -I {} wc -l {}
 PREDICTED: find Path -type f -exec wc -l {} +
--------------------------------------------------------------------------------
    SOURCE: find all file that been use more than _TIMESPAN sinc status was last chang
    TARGET: find Path Path
 PREDICTED: find Path -ctime +Timespan
--------------------------------------------------------------------------------
[7.385793801298045e-05]
Processing epoch 41: 100%|██████████| 1164/1164 [01:15<00:00, 15.44it/s, loss=1.129]
--------------------------------------------------------------------------------
    SOURCE: read a line from standard input into variabl _REGEX with prompt _FILE
    TARGET: read -p Regex Regex
 PREDICTED: read -p Regex
--------------------------------------------------------------------------------
    SOURCE: display process tree of a process with id _REGEX show parent process and process id
    TARGET: pstree -p -s Regex
 PREDICTED: pstree -s Regex
--------------------------------------------------------------------------------
[7.269952498697736e-05]
Processing epoch 42: 100%|██████████| 1164/1164 [01:15<00:00, 15.41it/s, loss=1.184]
--------------------------------------------------------------------------------
    SOURCE: find and remov _SIZE file from user 's directori
    TARGET: find Path -size Size -exec rm {} \;
 PREDICTED: find Path -size Size -exec rm {} \;
--------------------------------------------------------------------------------
    SOURCE: find all _FILE file exclud _FILE file under _FILE with null charact as delimit
    TARGET: find Path -name Regex ! -name Regex -print0
 PREDICTED: find Path -name Regex -print0
--------------------------------------------------------------------------------
[7.152555484041477e-05]
Processing epoch 43: 100%|██████████| 1164/1164 [01:15<00:00, 15.33it/s, loss=1.175]
--------------------------------------------------------------------------------
    SOURCE: find all file in current directori which size _SIZE in current disk partit
    TARGET: find Path -size -Size -xdev -print
 PREDICTED: find Path -xdev -size +Size -print
--------------------------------------------------------------------------------
    SOURCE: find all file name _REGEX start from _DIRECTORY
    TARGET: find Path -name Regex
 PREDICTED: find Path -name Regex
--------------------------------------------------------------------------------
[7.033683215379002e-05]
Processing epoch 44: 100%|██████████| 1164/1164 [01:16<00:00, 15.28it/s, loss=1.119]
--------------------------------------------------------------------------------
    SOURCE: display all _FILE file and header file in path _FILE and not search in sub directori
    TARGET: find Path -maxdepth Quantity \( -name Regex -or -name Regex \) -print
 PREDICTED: find Path -maxdepth Quantity -mindepth Quantity -iname Regex -type f
--------------------------------------------------------------------------------
    SOURCE: find all regular file in _REGEX directori tree which not been modifi in _TIMESPAN and delet them
    TARGET: find Path -type f -mtime +Timespan -exec rm {} \;
 PREDICTED: find Path -type f -mtime +Timespan -exec rm {} \;
--------------------------------------------------------------------------------
[6.91341716182545e-05]
Processing epoch 45: 100%|██████████| 1164/1164 [01:16<00:00, 15.21it/s, loss=1.105]
--------------------------------------------------------------------------------
    SOURCE: search in current directori downward all file whose owner is _REGEX and grep is grp
    TARGET: find Path \( -user Regex Path Path Path \) -print
 PREDICTED: find Path \( -user Regex -or -name Regex \) -print
--------------------------------------------------------------------------------
    SOURCE: request mx record of _FILE domain and filter out all comment string
    TARGET: dig Regex Regex | grep -v Regex | grep Regex
 PREDICTED: dig Regex Regex dragon-architect.com | awk Program Program
--------------------------------------------------------------------------------
[6.791839747726503e-05]
Processing epoch 46: 100%|██████████| 1164/1164 [01:16<00:00, 15.24it/s, loss=1.093]
--------------------------------------------------------------------------------
    SOURCE: time stamp everi ping request to _FILE in unix epoch format
    TARGET: ping Regex -n Regex -i Quantity -W Quantity Regex
 PREDICTED: ping Regex | awk Program
--------------------------------------------------------------------------------
    SOURCE: save date of first sunday in month _REGEX of year _REGEX in _REGEX variabl
    TARGET: cal -m DateTime DateTime | awk Program
 PREDICTED: set $( cal DateTime DateTime | tr -s Regex | tail - Quantity )
--------------------------------------------------------------------------------
[6.669034296168855e-05]
Processing epoch 47: 100%|██████████| 1164/1164 [01:16<00:00, 15.23it/s, loss=1.198]
--------------------------------------------------------------------------------
    SOURCE: find _FILE file omit result contain _FILE
    TARGET: find Path ! -path Regex -type f -name Regex
 PREDICTED: find Path ! -path Regex -type f -name Regex
--------------------------------------------------------------------------------
    SOURCE: replac last occurr of _REGEX with _FILE in file
    TARGET: tac File | sed Program | tac
 PREDICTED: tac File | awk Program | tac
--------------------------------------------------------------------------------
[6.545084971874738e-05]
Processing epoch 48: 100%|██████████| 1164/1164 [01:16<00:00, 15.25it/s, loss=1.123]
--------------------------------------------------------------------------------
    SOURCE: save hexadecim byte _NUMBER in binari file _FILE to variabl _REGEX
    TARGET: od -t x1 --skip-bytes Size --read-bytes Size File | head - Quantity | awk Program
 PREDICTED: od -t x1 --skip-bytes Size --read-bytes Size File | head - Quantity | awk Program
--------------------------------------------------------------------------------
    SOURCE: output _FILE omit all contain directori _DIRECTORY
    TARGET: basename Regex
 PREDICTED: find Path ! -path Regex -type d -exec ls -l -r -t {} \;
--------------------------------------------------------------------------------
[6.420076723519615e-05]
Processing epoch 49: 100%|██████████| 1164/1164 [01:16<00:00, 15.22it/s, loss=1.114]
--------------------------------------------------------------------------------
    SOURCE: remov all _FILE file in and below _FILE
    TARGET: find Path -name Regex | xargs -I {} rm {}
 PREDICTED: find Path -name Regex -print0 | xargs -0 -I {} rm {}
--------------------------------------------------------------------------------
    SOURCE: extract data from _FILE tabl in _FILE
    TARGET: paste -d Regex >( grep Regex File | sed -e Program ) >( grep Regex File | sed -e Program )
 PREDICTED: tail -f Regex | grep --line-buffered Regex | read -t Quantity
--------------------------------------------------------------------------------
[6.294095225512605e-05]
Processing epoch 50: 100%|██████████| 1164/1164 [01:15<00:00, 15.39it/s, loss=1.156]
--------------------------------------------------------------------------------
    SOURCE: display all _FILE file and header file in path _FILE and not search in sub directori
    TARGET: find Path -maxdepth Quantity \( -name Regex -or -name Regex \) -print
 PREDICTED: find Path -name Regex -maxdepth Quantity
--------------------------------------------------------------------------------
    SOURCE: send current job to background
    TARGET: bg Regex
 PREDICTED: bg
--------------------------------------------------------------------------------
[6.167226819279528e-05]
Processing epoch 51: 100%|██████████| 1164/1164 [01:15<00:00, 15.46it/s, loss=1.094]
--------------------------------------------------------------------------------
    SOURCE: format space separ field in _REGEX as a tabl
    TARGET: column -t -s Regex File
 PREDICTED: column -t -s Regex File
--------------------------------------------------------------------------------
    SOURCE: find all file in _FILE whose name begin with current user 's name follow by _REGEX
    TARGET: find Path -maxdepth Quantity -name Regex
 PREDICTED: find Path -regex Regex
--------------------------------------------------------------------------------
[6.0395584540887963e-05]
Processing epoch 52: 100%|██████████| 1164/1164 [01:15<00:00, 15.36it/s, loss=1.117]
--------------------------------------------------------------------------------
    SOURCE: find all _REGEX file and directori under current directori and copi them to _FILE
    TARGET: find Path -name Regex | sed Program | xargs -I {} cp File {}
 PREDICTED: find Path -name Regex | sed -e Program -e Program -e Program | xargs -I {} cp File {}
--------------------------------------------------------------------------------
    SOURCE: find file and directori name blah under current directori
    TARGET: find Path -name Regex
 PREDICTED: find Path -iname Regex
--------------------------------------------------------------------------------
[5.9111776274607377e-05]
Processing epoch 53: 100%|██████████| 1164/1164 [01:17<00:00, 14.98it/s, loss=1.111]
--------------------------------------------------------------------------------
    SOURCE: find file and directori under current directori and print them as null termin string
    TARGET: find Path -print0
 PREDICTED: find Path -print0
--------------------------------------------------------------------------------
    SOURCE: search bla directori recurs for _FILE file
    TARGET: find Path -name Regex
 PREDICTED: find Path -name Regex
--------------------------------------------------------------------------------
[5.7821723252011545e-05]
Processing epoch 54: 100%|██████████| 1164/1164 [01:17<00:00, 14.96it/s, loss=1.076]
--------------------------------------------------------------------------------
    SOURCE: find all _NUMBER permiss file under _FILE directori
    TARGET: find Path -type f -perm Permission
 PREDICTED: find Path -type f -perm Permission -print
--------------------------------------------------------------------------------
    SOURCE: find all file that were modifi within _TIMESPAN
    TARGET: find Path -mtime -Timespan
 PREDICTED: find Path -type f -mtime -Timespan
--------------------------------------------------------------------------------
[5.6526309611002574e-05]
Processing epoch 55: 100%|██████████| 1164/1164 [01:16<00:00, 15.23it/s, loss=1.096]
--------------------------------------------------------------------------------
    SOURCE: find all file name _REGEX start from _DIRECTORY
    TARGET: find Path -name Regex
 PREDICTED: find Path -name Regex
--------------------------------------------------------------------------------
    SOURCE: output _FILE omit all contain directori _DIRECTORY
    TARGET: basename Regex
 PREDICTED: find Path ! -path Regex -type d -exec compress File {} \;
--------------------------------------------------------------------------------
[5.5226423163382674e-05]
Processing epoch 56: 100%|██████████| 1164/1164 [01:16<00:00, 15.29it/s, loss=1.037]
--------------------------------------------------------------------------------
    SOURCE: renam all _REGEX directori to _REGEX in current directori tree
    TARGET: find Path -type d | awk -F Regex Program | sort -k Number -n -r | awk Program | sed Program | xargs -n Quantity -I {} mv {}
 PREDICTED: find Path -type d -exec rename Regex {} \;
--------------------------------------------------------------------------------
    SOURCE: enabl shell option _REGEX
    TARGET: shopt -s nocaseglob
 PREDICTED: shopt -s expand_aliases
--------------------------------------------------------------------------------
[5.392295478639225e-05]
Processing epoch 57: 100%|██████████| 1164/1164 [01:16<00:00, 15.19it/s, loss=1.045]
--------------------------------------------------------------------------------
    SOURCE: display all file in current directori exclud those that are in _REGEX directori
    TARGET: find Path -name Regex -prune -or -print
 PREDICTED: find Path -maxdepth Quantity -name Regex -prune -or -print
--------------------------------------------------------------------------------
    SOURCE: find all directori name _REGEX under current directori and set read-write-execut permiss for owner and group and no permiss for other for those directori
    TARGET: find Path -type d -name Regex -exec chmod Permission {} \;
 PREDICTED: find Path -type d -name Regex -exec chmod Permission {} \;
--------------------------------------------------------------------------------
[5.261679781214718e-05]
Processing epoch 58: 100%|██████████| 1164/1164 [01:16<00:00, 15.25it/s, loss=1.068]
--------------------------------------------------------------------------------
    SOURCE: replac all newlin with space in standard input
    TARGET: sed -z Program
 PREDICTED: sed -e Program
--------------------------------------------------------------------------------
    SOURCE: delet all file and directori name test under maximum _NUMBER level down current directori
    TARGET: find Path -maxdepth Quantity -name Regex -exec rm -r -f {} \;
 PREDICTED: find Path -maxdepth Quantity -name Regex -delete
--------------------------------------------------------------------------------
[5.130884741539365e-05]
Processing epoch 59: 100%|██████████| 1164/1164 [01:16<00:00, 15.22it/s, loss=1.057]
--------------------------------------------------------------------------------
    SOURCE: find all file in _FILE whose name begin with current user 's name follow by _REGEX
    TARGET: find Path -maxdepth Quantity -name Regex
 PREDICTED: find Path -regex Regex
--------------------------------------------------------------------------------
    SOURCE: search directori _DIRECTORY recurs for regular file
    TARGET: find Path -type f
 PREDICTED: find Path -type f -print
--------------------------------------------------------------------------------
