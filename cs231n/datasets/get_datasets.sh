# cd cs231n/datasets
if [ ! -d "cifar-10-batches-py" ]; then
  wget https://storage.yandexcloud.net/dl-course/cifar-10-python.zip -O cifar-10-python.zip
  tar -xvf cifar-10-python.zip
  rm cifar-10-python.tar.gz
  wget https://storage.yandexcloud.net/dl-course/imagenet_val_25.npz
fi
# cd ../..