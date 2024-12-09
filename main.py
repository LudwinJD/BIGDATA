import kagglehub

# Download latest version
path = kagglehub.dataset_download("eimadevyni/car-model-variants-and-images-dataset")

print("Path to dataset files:", path)