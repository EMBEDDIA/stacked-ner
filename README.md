#### Run the code

BERT models need to be dowloaded (with the exception of CamemBERT)

Training:

```

CUDA_VISIBLE_DEVICES=1,2,3 python main.py 
--directory TEMP_MODEL # path to save the model; predictions on test/dev will be automatically saved here at the end of training
--pre_trained_model PRETRAINED_MODEL_NAME #bert-base-cased
--train_dataset train.tsv 
--test_dataset test.tsv 
--dev_dataset valid.tsv 
--batch_size 4 
--do_train 
--no_cpu 5
--language french #for CamemBERT; english for other models
--model stacked # or bert 
--num_layers 2 #2 Transformer layers
```

Predicting:
```

python main.py 
--directory TEMP_MODEL #same param as train.py
--pre_trained_model PRETRAINED_MODEL_NAME #same param as main.py
--train_dataset train.tsv #same param as main.py
--test_dataset test.tsv #same param as main.py
--dev_dataset valid.tsv #same param as main.py
--dataset_dir DIR_DATA_TEST #directory with .tsv to be predicted
--output_dir DIR_DATA_TEST_PREDICTIONS #directory where predictions will be saved
--batch_size 4 
--do_eval 
--saved_model TEMP_MODEL/best/best_ #best model after training
--no_cpu 5
--language french #for CamemBERT; english for other; same as main.py
--model stacked # or bert; same as main.py
--num_layers 2 #2 Transformer layers; same as main.py


```
##### Dataset Annotation


```
TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC
Wienstrasse	I-LOC	O	O	O	O	O	null	O	SpaceAfter

```

#### Requirements
```
pip install -r requirements.txt
```

#### How to citate:

```
@inproceedings{boros2020robust,
  title={Robust named entity recognition and linking on historical multilingual documents},
  author={Boros, Emanuela and Pontes, Elvys Linhares and Cabrera-Diego, Luis Adri{\'a}n and Hamdi, Ahmed and Moreno, Jos{\'e} and Sid{\`e}re, Nicolas and Doucet, Antoine},
  booktitle={Conference and Labs of the Evaluation Forum (CLEF 2020)},
  volume={2696},
  number={Paper 171},
  pages={1--17},
  year={2020},
  organization={CEUR-WS Working Notes}
}
```

```
@inproceedings{borocs2020alleviating,
  title={Alleviating digitization errors in named entity recognition for historical documents},
  author={Boro{\c{s}}, Emanuela and Hamdi, Ahmed and Pontes, Elvys Linhares and Cabrera-Diego, Luis-Adri{\'a}n and Moreno, Jose G and Sidere, Nicolas and Doucet, Antoine},
  booktitle={Proceedings of the 24th Conference on Computational Natural Language Learning},
  pages={431--441},
  year={2020}
}
```

