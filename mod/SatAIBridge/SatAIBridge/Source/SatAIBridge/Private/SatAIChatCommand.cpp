#include "SatAIChatCommand.h"
#include "Command/CommandSender.h"
#include "SatAISubsystem.h"
#include "GameFramework/Pawn.h"
#include "Engine/World.h"
#include "Buildables/FGBuildable.h"
#include "FGPlayerController.h"

namespace
{
    const TMap<FString, FString> KnownBuildings = {
        { TEXT("constructor"), TEXT("/Game/FactoryGame/Buildable/Factory/ConstructorMk1/Build_ConstructorMk1.Build_ConstructorMk1_C") },
        { TEXT("smelter"),     TEXT("/Game/FactoryGame/Buildable/Factory/SmelterMk1/Build_SmelterMk1.Build_SmelterMk1_C") },
    };

    const TCHAR* RecipeIronIngot = TEXT("/Game/FactoryGame/Recipes/Smelter/Recipe_IngotIron.Recipe_IngotIron_C");
    const TCHAR* RecipeIronPlate = TEXT("/Game/FactoryGame/Recipes/Constructor/Recipe_IronPlate.Recipe_IronPlate_C");
    const TCHAR* BeltMk1 = TEXT("/Game/FactoryGame/Buildable/Factory/ConveyorBeltMk1/Build_ConveyorBeltMk1.Build_ConveyorBeltMk1_C");

    FTransform GroundTransformAhead(APawn* Pawn, float DistanceCm)
    {
        UWorld* World = Pawn->GetWorld();
        FVector Ahead = Pawn->GetActorLocation() + Pawn->GetActorForwardVector() * DistanceCm;

        FHitResult Hit;
        FCollisionQueryParams Params;
        Params.AddIgnoredActor(Pawn);
        FVector Location = Ahead;
        if (World->LineTraceSingleByChannel(Hit, Ahead + FVector(0, 0, 500.f), Ahead - FVector(0, 0, 5000.f),
            ECC_Visibility, Params))
        {
            Location = Hit.ImpactPoint;
        }

        float Yaw = FMath::RoundToFloat(Pawn->GetActorRotation().Yaw / 90.f) * 90.f;
        return FTransform(FRotator(0.f, Yaw, 0.f), Location);
    }
}

ASatAIChatCommand::ASatAIChatCommand()
{
    CommandName = TEXT("ai");
    MinNumberOfArguments = 1;
    Usage = NSLOCTEXT("SatAIBridge", "AIUsage", "/ai ping");
}

EExecutionStatus ASatAIChatCommand::ExecuteCommand_Implementation(
    UCommandSender* Sender, const TArray<FString>& Arguments, const FString& Label)
{
    const FString& SubCommand = Arguments[0];

    if (SubCommand == TEXT("ping"))
    {
        ASatAISubsystem* Subsystem = ASatAISubsystem::Get(this);
        if (!Subsystem)
        {
            Sender->SendChatMessage(TEXT("SatAIBridge: subsystem not found (registered in RootGameWorld?)"));
            return EExecutionStatus::UNCOMPLETED;
        }
        Sender->SendChatMessage(Subsystem->HandlePing());
        return EExecutionStatus::COMPLETED;
    }

    if (SubCommand == TEXT("spawn"))
    {
        if (Arguments.Num() < 2 || !KnownBuildings.Contains(Arguments[1]))
        {
            Sender->SendChatMessage(TEXT("Usage: /ai spawn <constructor|smelter>"));
            return EExecutionStatus::BAD_ARGUMENTS;
        }

        AFGPlayerController* PC = Sender->GetPlayer();
        APawn* Pawn = PC ? PC->GetPawn() : nullptr;
        if (!Pawn)
        {
            Sender->SendChatMessage(TEXT("SatAIBridge: spawn must be run by a player in the world"));
            return EExecutionStatus::UNCOMPLETED;
        }

        ASatAISubsystem* Subsystem = ASatAISubsystem::Get(this);
        if (!Subsystem)
        {
            Sender->SendChatMessage(TEXT("SatAIBridge: subsystem not found"));
            return EExecutionStatus::UNCOMPLETED;
        }

        FTransform Transform = GroundTransformAhead(Pawn, 1500.f);

        FString Error;
        AFGBuildable* Built = Subsystem->SpawnBuildable(KnownBuildings[Arguments[1]], Transform, Error);
        if (!Built)
        {
            Sender->SendChatMessage(TEXT("SatAIBridge: spawn failed: ") + Error);
            return EExecutionStatus::UNCOMPLETED;
        }

        Sender->SendChatMessage(FString::Printf(TEXT("SatAIBridge: spawned %s"), *Built->GetName()));
        return EExecutionStatus::COMPLETED;
    }

    if (SubCommand == TEXT("line"))
    {
        AFGPlayerController* PC = Sender->GetPlayer();
        APawn* Pawn = PC ? PC->GetPawn() : nullptr;
        ASatAISubsystem* Subsystem = ASatAISubsystem::Get(this);
        if (!Pawn || !Subsystem)
        {
            Sender->SendChatMessage(TEXT("SatAIBridge: need a player and the subsystem"));
            return EExecutionStatus::UNCOMPLETED;
        }

        FTransform SmelterT = GroundTransformAhead(Pawn, 1500.f);
        FTransform ConstructorT = SmelterT;
        ConstructorT.AddToTranslation(SmelterT.GetRotation().GetRightVector() * 1500.f);

        FString Error;
        AFGBuildable* Smelter = Subsystem->SpawnBuildable(KnownBuildings[TEXT("smelter")], SmelterT, Error);
        AFGBuildable* Constructor = Smelter
            ? Subsystem->SpawnBuildable(KnownBuildings[TEXT("constructor")], ConstructorT, Error)
            : nullptr;
        if (!Constructor)
        {
            Sender->SendChatMessage(TEXT("SatAIBridge: spawn failed: ") + Error);
            return EExecutionStatus::UNCOMPLETED;
        }

        if (!Subsystem->SetMachineRecipe(Smelter, RecipeIronIngot, Error) ||
            !Subsystem->SetMachineRecipe(Constructor, RecipeIronPlate, Error))
        {
            Sender->SendChatMessage(TEXT("SatAIBridge: recipe failed: ") + Error);
            return EExecutionStatus::UNCOMPLETED;
        }

        if (!Subsystem->ConnectWithBelt(Smelter, Constructor, BeltMk1, Error))
        {
            Sender->SendChatMessage(TEXT("SatAIBridge: belt failed: ") + Error);
            return EExecutionStatus::UNCOMPLETED;
        }

        Sender->SendChatMessage(TEXT("SatAIBridge: smelter -> belt -> constructor built"));
        return EExecutionStatus::COMPLETED;
    }

    Sender->SendChatMessage(FString::Printf(TEXT("Unknown subcommand '%s'. Try: /ai ping"), *SubCommand));
    return EExecutionStatus::BAD_ARGUMENTS;
}