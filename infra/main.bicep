targetScope = 'resourceGroup'

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Name prefix for created resources')
param namePrefix string = 'proteinai'

@description('Container image for the demo (should be in ACR for best cold start)')
param image string

@description('Azure Container Registry name (without .azurecr.io), must be in the same resource group')
param acrName string

@description('Workload profile name (serverless GPU T4)')
param gpuWorkloadProfileName string = 'Consumption-GPU-NC8as-T4'

@description('Workload profile name for premium ingress (must NOT be Consumption/Flex)')
param ingressWorkloadProfileName string = 'Ingress-D4'

@description('Ingress target port')
param targetPort int = 7860

@secure()
@description('Optional Hugging Face token. Leave empty if not required for the models.')
param hfToken string = ''

var envName = '${namePrefix}-aca-env'
var appName = '${namePrefix}-demo'
var registryServer = '${acrName}.azurecr.io'

resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' existing = {
  name: acrName
}

resource env 'Microsoft.App/managedEnvironments@2025-10-02-preview' = {
  name: envName
  location: location
  properties: {
    workloadProfiles: [
      {
        name: gpuWorkloadProfileName
        workloadProfileType: gpuWorkloadProfileName
      }
      {
        name: ingressWorkloadProfileName
        workloadProfileType: 'D4'
        minimumCount: 2
        maximumCount: 2
      }
    ]
    ingressConfiguration: {
      workloadProfileName: ingressWorkloadProfileName
      headerCountLimit: 100
      requestIdleTimeout: 30
      terminationGracePeriodSeconds: 480
    }
    publicNetworkAccess: 'Enabled'
  }
}

resource app 'Microsoft.App/containerApps@2025-10-02-preview' = {
  name: appName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: env.id
    workloadProfileName: gpuWorkloadProfileName
    configuration: {
      ingress: {
        external: true
        targetPort: targetPort
        transport: 'auto'
      }
      registries: [
        {
          server: registryServer
          identity: 'system'
        }
      ]
      secrets: !empty(hfToken) ? [
        {
          name: 'hf-token'
          value: hfToken
        }
      ] : []
    }
    template: {
      containers: [
        {
          name: 'demo'
          image: image
          resources: {
            cpu: 8
            memory: '56Gi'
            gpu: 1
          }
          env: !empty(hfToken) ? [
            {
              name: 'PORT'
              value: string(targetPort)
            }
            {
              name: 'HF_TOKEN'
              secretRef: 'hf-token'
            }
            {
              name: 'HF_HOME'
              value: '/tmp/huggingface'
            }
            {
              name: 'TRANSFORMERS_CACHE'
              value: '/tmp/huggingface'
            }
            {
              name: 'BIOEMU_OUTPUT_BASE_DIR'
              value: '/tmp/bioemu_runs'
            }
          ] : [
            {
              name: 'PORT'
              value: string(targetPort)
            }
            {
              name: 'HF_HOME'
              value: '/tmp/huggingface'
            }
            {
              name: 'TRANSFORMERS_CACHE'
              value: '/tmp/huggingface'
            }
            {
              name: 'BIOEMU_OUTPUT_BASE_DIR'
              value: '/tmp/bioemu_runs'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
}

// Allow the Container App to pull the private image from ACR using its system-assigned identity.
resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, app.identity.principalId, 'acrpull')
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: app.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output containerAppUrl string = 'https://${app.properties.configuration.ingress.fqdn}'

output containerAppPrincipalId string = app.identity.principalId
