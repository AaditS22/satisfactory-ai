#pragma once

#include "CoreMinimal.h"
#include "Command/ChatCommandInstance.h"
#include "SatAIChatCommand.generated.h"

UCLASS()
class SATAIBRIDGE_API ASatAIChatCommand : public AChatCommandInstance
{
    GENERATED_BODY()

public:
    ASatAIChatCommand();

    virtual EExecutionStatus ExecuteCommand_Implementation(
        UCommandSender* Sender,
        const TArray<FString>& Arguments,
        const FString& Label) override;
};