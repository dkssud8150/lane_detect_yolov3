curl https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip --create-dirs ./datasets/tusimple/train -o ./datasets/tusimple/train/img_tusimple.zip
cd ./datasets/tusimple/train && unzip -d . img_tusimple.zip

curl https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip --create-dirs ./datasets/tusimple/test -o ./datasets/tusimple/test/img_tusimple.zip
cd ./datasets/tusimple/test/ && unzip -d img_tusimple.zip

curl https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json -o ./datasets/tusimple/test/test_label.json
curl https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_baseline.json -o ./datasets/tusimple/test/test_baseline.json

