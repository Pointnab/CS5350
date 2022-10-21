echo warning: code takes a long time to test 500 iterations
python3 AdaBoostTest.py bank/train.csv bank/test.csv 500
echo warning: code takes a long time to test 500 trees
python3 BagTest.py bank/train.csv bank/teset.csv 500 1000
echo warning: code takes a long time to test 500 trees
python3 BagBiasVar.py bank/train.csv bank/teset.csv