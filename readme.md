# Mousika of In-Network Intelligence in IEEE INFOCOM22 
**More information about us** https://xgr19.github.io  
**News**: Our extended draft of the journal version is https://github.com/xgr19/Mousika/tree/Mousikav2

This repository provides part of the code used to run a demo, and all the code will be released after we extend this paper into a journal version.

## Code Architecture

```
-- Distillation
	-- code
		-- train_teacher.py(train the teacher model)
		-- bdt_test.py(an example for distilling the model from teacher model to BDT)
		-- dt_test.py(train DT model)
		-- SoftTree.py(the class of soft DT, which supports tree distillation)
		-- Tree.py(the original DT)
		-- utils.py(include the load-data function)
		-- Dataset/(univ.zip is the flow size prediction task dataset, both 
			the continuous feature 'univ_C' and binary feature 'univ' are stored here)
		-- models/(teacher model)
		-- result_acc/
		-- rule_tree/(store rules.txt of bdt_test.py)
		
-- rule2entry
	-- code
		-- rule2entry.py(convert rule of DT to ternary entry of P4 table)
		-- entry2dataplane.py(load ternary entry and send to P4 dataplane)
	-- output
	    -- rule_tree/(copy Distillation/rule_tree/*.txt to here)
	    -- ternary_entry/(generated P4 tale entries from rules.txt)

-- P4
    -- flowcontrol.p4(implementation of P4 data plane)
```

## Run Decision Trees
#### (Responsibility of Yutao Dong, dyt20@mails.tsinghua.edu.cn)  
Training the BDT model, this script outputs the results of sdt(BDT with distillation) and bdt(BDT without distillation):
```
python bdt_test.py

```
Training the DT model, this script outputs the results of DT:

```
python dt_test.py

```
## Run P4 Program
#### (Responsibility of Guanglin Duan, imbaplayer@163.com)  
compile P4 code: flowcontrol.p4 under your p4 path of a Barefoot tofino switch

```
cd $SDE/pkgsrc/p4-build
./configure --prefix=$SDE_INSTALL --with-tofino --with-bf-runtime P4_NAME=flowcontrol P4_PATH= <your p4 path> P4_VERSION=p4-16 P4C=p4c --enable-thrift
make
make install
```

run the compiled P4 program

```
cd $SDE
./run_switchd.sh -p flowcontrol
```

convert rule of BDT/distilled BDT to ternary entry of P4 table

```
cd rule2entry/code
python rule2entry.py
```

load ternary entry and send to P4 data plane

```
mv rule2entry/code/entry2dataplane.py /root/static_entry/
cd $SDE
./run_p4_tests.sh -t /root/static_entry/
```

