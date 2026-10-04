// Copyright Epic Games, Inc. All Rights Reserved.

#include "SatAIBridge.h"

#define LOCTEXT_NAMESPACE "FSatAIBridgeModule"

void FSatAIBridgeModule::StartupModule()
{
	UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: hello world!"));
}

void FSatAIBridgeModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FSatAIBridgeModule, SatAIBridge)