# CSF Guidelines
## Writing the Job Script

**Still a work in progress**

General notes on the structure of the runs so far:
```
~
├── scratch/ #this is the folder from which the slurm batch can be submitted
│      ├── logs/
│      │     └─ # all log files go here as per name in bash script
│      └── results/
│         └── tasks/ 
│               └# all the results go here as per name in running script 
└── rev_mrgp/ # this is where all the code is stored
        └── Cooperative-Object-Transportation
              ├── cot
              ├── docs
              ├── scripts/
                     ├── csf/ # this is where the bash scripts and the code will be stored
                           ├── results/ #  results stored longer term
        # the rest of this folder is the same as usual, the only addition is the CSF folder 
                     
```
Once the job has run, the merge script will be run to combine all the data in the tasks folder and both the combined data and the raw data will be copied into the `rev_mrgp/Cooperative-Object-Transportation/scripts/csf/results` folder and will be uploaded onto GitHub. The results are currently stored under different folders named with job name and taskID for now (the naming system can be worked out)

The `scratch/logs` and `scratch/results` folder will be cleared out before every run (or every 3 runs - this can be finalised as we go along) just to prevent confusion.