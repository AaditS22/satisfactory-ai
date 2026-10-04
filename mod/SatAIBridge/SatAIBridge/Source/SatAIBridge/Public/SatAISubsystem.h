#pragma once

#include "CoreMinimal.h"
#include "Subsystem/ModSubsystem.h"
#include "SatAISubsystem.generated.h"

UCLASS(Abstract, Blueprintable)
class SATAIBRIDGE_API ASatAISubsystem : public AModSubsystem
{
    GENERATED_BODY()

public:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    static ASatAISubsystem* Get(const UObject* WorldContext);

    FString HandlePing();

private:
    int32 PingCount = 0;
    static TWeakObjectPtr<ASatAISubsystem> Instance;
};