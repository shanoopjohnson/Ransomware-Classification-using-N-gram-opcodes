# Ransomware-Classification-using-N-gram-opcodes
This code was developed as part of my master Thesis on System security. It make use of Ensemble learning model to classify Ransomware samples.


Ransomware Classification
--------------------------------------------------
I: Disassembling
--------------------------------------------------
Install and Use IDA Pro for disassembling. 
1) File -> Load file (Allow IDA to connect to internet and download necessary files to unpack)
2) File -> Produce File -> Create ASM File
3) The disassembled ASM file will be saved in the location of your chosing.
4) In case if you have many files to disassemble, Use a batch script in below format to do so:

	
"D:\appp\IDA 7.3\ida64.exe" -B D:\Sem3\final_year_project\dataset\VirusShare\extracted\VirusShare_0a06fbd7930744b8a6b24c92007913ce
"D:\appp\IDA 7.3\ida64.exe" -B D:\Sem3\final_year_project\dataset\VirusShare\extracted\VirusShare_0b49ddd6a619da3bc14c289497ab859d
"D:\appp\IDA 7.3\ida64.exe" -B D:\Sem3\final_year_project\dataset\VirusShare\extracted\VirusShare_0b69eeab38b6a00ae262cc3b52cbedbb
"D:\appp\IDA 7.3\ida64.exe" -B D:\Sem3\final_year_project\dataset\VirusShare\extracted\VirusShare_0b6812de6a6f9209f56ad67a45647d80

5) Remove unwanted intermediate files. We only need asm files.

--------------------------------------------------
II: Extracting Opcodes
--------------------------------------------------
1) Open extract_opcodes.py file and give location where the asm files are stored in "malware_directory"
2) Make sure you have trainLabels.csv file in the source folder. It contains the label information.
3) Run the script. 
	It will read each file one by one and generate a "dataset.txt" file with time sorted opcode sequence.

--------------------------------------------------
III. Train model on dataset
--------------------------------------------------
1) open train_model.py
2) enter the location of created dataset in Step II in "data"
3) This will also create 
	i) saved model with all weights
	ii) a feature N-gram CSV file.

-----------------------------------------------------
IV. Test with unknown file samples
------------------------------------------------------
1) Run steps I and II with the unknown file samples and create dataset.
2) open "test_model_unknown_files.py" file and give dataset location in data
3) Make sure filename in "loaded_model" is same as the filename saved in Step III.
4) Run the file.
