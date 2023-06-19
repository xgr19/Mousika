# Empowering In-Network Classification in Programmable Switches by Binary Decision Tree and Knowledge Distillation
**More information about us** [https://xgr19.github.io](https://xgr19.github.io)  

This is Mousikav2 published in ToN 2023. We provides the whole distillation code in this branch.

We hope that our another In-network intelligence work [Soter](https://github.com/xgr19/Soter) would be also helpful for you.

## Code Architecture
```
-- Distillation
  -- train_teacher.py (train the teacher model)
  -- bdt_test.py (an example for distilling the model from the teacher model to BDT)
  -- SoftTree.py (the class of soft BDT, which supports tree distillation)
  -- utils.py (include some used functions)
  -- models/ (teacher model structures)
  -- mousika_v2/ (rules of BDT)
  -- params/ (parameters of the teacher model)
  -- Dataset/ (save training data and test data)
```

## Dataset
The dataset we use is UNIV, You can download it at [UNIV](https://pages.cs.wisc.edu/~tbenson/IMC10_Data.html).

You need to convert the data features to binary features and then save them in the folder Dataset/ in CSV format.

## Run Binary Decision Trees
Training the BDT model, this script outputs the results and rules of soft BDT (BDT with distillation):
```
python bdt_test.py
```

The rules in folder mousika_v2/ can be installed into Tofino switches through the P4 code in the ["main"](https://github.com/xgr19/Mousika/tree/main) branch.

You can modify the settings in bdt_test.py and train_teacher.py to specify the teacher model that you need.
