#include "SatAISubsystem.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "EngineUtils.h"  

TWeakObjectPtr<ASatAISubsystem> ASatAISubsystem::Instance;

void ASatAISubsystem::BeginPlay()
{
    Super::BeginPlay();
    Instance = this;
    UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: subsystem started"));
}

void ASatAISubsystem::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (Instance.Get() == this)
    {
        Instance.Reset();
    }
    Super::EndPlay(EndPlayReason);
}

ASatAISubsystem* ASatAISubsystem::Get(const UObject* WorldContext)
{
    UWorld* World = GEngine->GetWorldFromContextObject(WorldContext, EGetWorldErrorMode::ReturnNull);
    if (!World) return nullptr;

    if (Instance.IsValid() && Instance->GetWorld() == World)
    {
        return Instance.Get();
    }

    for (TActorIterator<ASatAISubsystem> It(World); It; ++It)
    {
        Instance = *It;
        return *It;
    }
    return nullptr;
}

FString ASatAISubsystem::HandlePing()
{
    PingCount++;
    return FString::Printf(TEXT("SatAIBridge: pong (v0.1, ping #%d)"), PingCount);
}