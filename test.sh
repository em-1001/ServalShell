nl = 'print current user name'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
translate(nl)

nl = 'copies "file.txt" to "null.txt"'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
translate(nl)

nl = 'finds all files with a ".txt" extension in the current directory'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
translate(nl)

nl = 'prints "Hello, World!" on the terminal'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
translate(nl)

nl = 'list current dictory files'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
translate(nl)

nl = 'Prints the current working directory.'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
translate(nl)

nl = 'gives execute permission to "script.sh"'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
translate(nl)

nl = 'changes the owner and group of "file.txt" to "user:group"'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
translate(nl)

nl = 'moves "file.txt" to "./bin"'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
translate(nl)

nl = 'deletes a file named "file.txt"'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
translate(nl)

nl = 'creates a directory named "my_folder"'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
translate(nl)

nl = 'changes to the "Documents" directory'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
translate(nl)

nl = 'displays the content of "file.txt"'
nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
translate(nl)
