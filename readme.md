# Empowering In-Network Classification in Programmable Switches by Binary Decision Tree and Knowledge Distillation
**More information about us** [https://xgr19.github.io](https://xgr19.github.io)  

This is Mousikav2 published in ToN 2023. We provides the whole distillation codes (5 teachers of RF, GBDT, GRU, LSTM, and MLP) in this branch.

We hope that our another In-network intelligence work [Soter](https://github.com/xgr19/Soter) would be also helpful for you.

## Code Architecture
```
-- Distillation
  -- train_teacher.py (train the teacher model)
  -- bdt_test.py (an example for distilling the model from teacher model to BDT)
  -- SoftTree.py (the class of soft DT, which supports tree distillation)
  -- utils.py (include some used functions)
  -- models/ (teacher model structure)
  -- mousika_v2/ (rules of BDT)
```

## Run Decision Trees
Training the BDT model, this script outputs the results and rules of SDT (BDT with distillation):
```
python bdt_test.py
```
