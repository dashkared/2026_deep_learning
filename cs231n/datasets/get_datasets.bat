@echo off
setlocal

pushd "%~dp0"

if exist "cifar-10-batches-py\" (
  echo CIFAR-10 is already available.
  goto :end
)

echo Downloading CIFAR-10...
curl.exe -L --fail "https://storage.yandexcloud.net/dl-course/cifar-10-python.zip" -o "cifar-10-python.zip"
if errorlevel 1 goto :download_error

echo Extracting CIFAR-10...
tar.exe -xf "cifar-10-python.zip"
if errorlevel 1 goto :extract_error

del "cifar-10-python.zip"

echo Downloading ImageNet validation data...
curl.exe -L --fail "https://storage.yandexcloud.net/dl-course/imagenet_val_25.npz" -o "imagenet_val_25.npz"
if errorlevel 1 goto :download_error

echo Dataset download complete.
goto :end

:extract_error
echo Failed to extract cifar-10-python.zip.
goto :end

:download_error
echo Failed to download a dataset file.

:end
popd
endlocal