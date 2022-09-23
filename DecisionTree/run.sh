echo car - training accuracy
python3 ID3.py car/train.csv car/train.csv H 1
python3 ID3.py car/train.csv car/train.csv H 2
python3 ID3.py car/train.csv car/train.csv H 3
python3 ID3.py car/train.csv car/train.csv H 4
python3 ID3.py car/train.csv car/train.csv H 5
python3 ID3.py car/train.csv car/train.csv H 6
python3 ID3.py car/train.csv car/train.csv ME 1
python3 ID3.py car/train.csv car/train.csv ME 2
python3 ID3.py car/train.csv car/train.csv ME 3
python3 ID3.py car/train.csv car/train.csv ME 4
python3 ID3.py car/train.csv car/train.csv ME 5
python3 ID3.py car/train.csv car/train.csv ME 6
python3 ID3.py car/train.csv car/train.csv GI 1
python3 ID3.py car/train.csv car/train.csv GI 2
python3 ID3.py car/train.csv car/train.csv GI 3
python3 ID3.py car/train.csv car/train.csv GI 4
python3 ID3.py car/train.csv car/train.csv GI 5
python3 ID3.py car/train.csv car/train.csv GI 6
echo car - test accuracy
python3 ID3.py car/train.csv car/test.csv H 1
python3 ID3.py car/train.csv car/test.csv H 2
python3 ID3.py car/train.csv car/test.csv H 3
python3 ID3.py car/train.csv car/test.csv H 4
python3 ID3.py car/train.csv car/test.csv H 5
python3 ID3.py car/train.csv car/test.csv H 6
python3 ID3.py car/train.csv car/test.csv ME 1
python3 ID3.py car/train.csv car/test.csv ME 2
python3 ID3.py car/train.csv car/test.csv ME 3
python3 ID3.py car/train.csv car/test.csv ME 4
python3 ID3.py car/train.csv car/test.csv ME 5
python3 ID3.py car/train.csv car/test.csv ME 6
python3 ID3.py car/train.csv car/test.csv GI 1
python3 ID3.py car/train.csv car/test.csv GI 2
python3 ID3.py car/train.csv car/test.csv GI 3
python3 ID3.py car/train.csv car/test.csv GI 4
python3 ID3.py car/train.csv car/test.csv GI 5
python3 ID3.py car/train.csv car/test.csv GI 6
echo bank - unkown category - training accuracy
python3 ID3bin.py bank/train.csv bank/train.csv H 1 False
python3 ID3bin.py bank/train.csv bank/train.csv H 2 False
python3 ID3bin.py bank/train.csv bank/train.csv H 3 False
python3 ID3bin.py bank/train.csv bank/train.csv H 4 False
python3 ID3bin.py bank/train.csv bank/train.csv H 5 False
python3 ID3bin.py bank/train.csv bank/train.csv H 6 False
python3 ID3bin.py bank/train.csv bank/train.csv ME 1 False
python3 ID3bin.py bank/train.csv bank/train.csv ME 2 False
python3 ID3bin.py bank/train.csv bank/train.csv ME 3 False
python3 ID3bin.py bank/train.csv bank/train.csv ME 4 False
python3 ID3bin.py bank/train.csv bank/train.csv ME 5 False
python3 ID3bin.py bank/train.csv bank/train.csv ME 6 False
python3 ID3bin.py bank/train.csv bank/train.csv GI 1 False
python3 ID3bin.py bank/train.csv bank/train.csv GI 2 False
python3 ID3bin.py bank/train.csv bank/train.csv GI 3 False
python3 ID3bin.py bank/train.csv bank/train.csv GI 4 False
python3 ID3bin.py bank/train.csv bank/train.csv GI 5 False
python3 ID3bin.py bank/train.csv bank/train.csv GI 6 False
echo bank - unkown category - test accuracy
python3 ID3bin.py bank/train.csv bank/test.csv H 1 False
python3 ID3bin.py bank/train.csv bank/test.csv H 2 False
python3 ID3bin.py bank/train.csv bank/test.csv H 3 False
python3 ID3bin.py bank/train.csv bank/test.csv H 4 False
python3 ID3bin.py bank/train.csv bank/test.csv H 5 False
python3 ID3bin.py bank/train.csv bank/test.csv H 6 False
python3 ID3bin.py bank/train.csv bank/test.csv ME 1 False
python3 ID3bin.py bank/train.csv bank/test.csv ME 2 False
python3 ID3bin.py bank/train.csv bank/test.csv ME 3 False
python3 ID3bin.py bank/train.csv bank/test.csv ME 4 False
python3 ID3bin.py bank/train.csv bank/test.csv ME 5 False
python3 ID3bin.py bank/train.csv bank/test.csv ME 6 False
python3 ID3bin.py bank/train.csv bank/test.csv GI 1 False
python3 ID3bin.py bank/train.csv bank/test.csv GI 2 False
python3 ID3bin.py bank/train.csv bank/test.csv GI 3 False
python3 ID3bin.py bank/train.csv bank/test.csv GI 4 False
python3 ID3bin.py bank/train.csv bank/test.csv GI 5 False
python3 ID3bin.py bank/train.csv bank/test.csv GI 6 False
echo bank - unkown missing - training accuracy
python3 ID3bin.py bank/train.csv bank/train.csv H 1 True
python3 ID3bin.py bank/train.csv bank/train.csv H 2 True
python3 ID3bin.py bank/train.csv bank/train.csv H 3 True
python3 ID3bin.py bank/train.csv bank/train.csv H 4 True
python3 ID3bin.py bank/train.csv bank/train.csv H 5 True
python3 ID3bin.py bank/train.csv bank/train.csv H 6 True
python3 ID3bin.py bank/train.csv bank/train.csv ME 1 True
python3 ID3bin.py bank/train.csv bank/train.csv ME 2 True
python3 ID3bin.py bank/train.csv bank/train.csv ME 3 True
python3 ID3bin.py bank/train.csv bank/train.csv ME 4 True
python3 ID3bin.py bank/train.csv bank/train.csv ME 5 True
python3 ID3bin.py bank/train.csv bank/train.csv ME 6 True
python3 ID3bin.py bank/train.csv bank/train.csv GI 1 True
python3 ID3bin.py bank/train.csv bank/train.csv GI 2 True
python3 ID3bin.py bank/train.csv bank/train.csv GI 3 True
python3 ID3bin.py bank/train.csv bank/train.csv GI 4 True
python3 ID3bin.py bank/train.csv bank/train.csv GI 5 True
python3 ID3bin.py bank/train.csv bank/train.csv GI 6 True
echo bank - unkown missing - test accuracy
python3 ID3bin.py bank/train.csv bank/test.csv H 1 True
python3 ID3bin.py bank/train.csv bank/test.csv H 2 True
python3 ID3bin.py bank/train.csv bank/test.csv H 3 True
python3 ID3bin.py bank/train.csv bank/test.csv H 4 True
python3 ID3bin.py bank/train.csv bank/test.csv H 5 True
python3 ID3bin.py bank/train.csv bank/test.csv H 6 True
python3 ID3bin.py bank/train.csv bank/test.csv ME 1 True
python3 ID3bin.py bank/train.csv bank/test.csv ME 2 True
python3 ID3bin.py bank/train.csv bank/test.csv ME 3 True
python3 ID3bin.py bank/train.csv bank/test.csv ME 4 True
python3 ID3bin.py bank/train.csv bank/test.csv ME 5 True
python3 ID3bin.py bank/train.csv bank/test.csv ME 6 True
python3 ID3bin.py bank/train.csv bank/test.csv GI 1 True
python3 ID3bin.py bank/train.csv bank/test.csv GI 2 True
python3 ID3bin.py bank/train.csv bank/test.csv GI 3 True
python3 ID3bin.py bank/train.csv bank/test.csv GI 4 True
python3 ID3bin.py bank/train.csv bank/test.csv GI 5 True
python3 ID3bin.py bank/train.csv bank/test.csv GI 6 True