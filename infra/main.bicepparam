using './main.bicep'

param location = 'swedencentral'
param namePrefix = 'proteinai'

// Set this to your ACR image, e.g. myacr.azurecr.io/proteinai-demo:latest
param image = 'proteinai010616d1bf.azurecr.io/proteinai-demo:fix180335'

// ACR name for pulling the image via managed identity
param acrName = 'proteinai010616d1bf'

// Optional: set if your Hugging Face account/token is required to download the models
@secure()
param hfToken = ''
