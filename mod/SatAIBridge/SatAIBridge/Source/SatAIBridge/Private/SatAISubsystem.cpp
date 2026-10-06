#include "SatAISubsystem.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "EngineUtils.h"  
#include "Buildables/FGBuildable.h"
#include "Buildables/FGBuildableManufacturer.h"
#include "FGRecipe.h"
#include "FGFactoryConnectionComponent.h"
#include "Buildables/FGBuildableConveyorBelt.h"
#include "Tests/FGTestBlueprintFunctionLibrary.h"

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

AFGBuildable* ASatAISubsystem::SpawnBuildable(const FString& ClassPath, const FTransform& Transform, FString& OutError)
{
    UClass* BuildClass = LoadClass<AFGBuildable>(nullptr, *ClassPath);
    if (!BuildClass)
    {
        OutError = FString::Printf(TEXT("class not found: %s"), *ClassPath);
        return nullptr;
    }

    AFGBuildable* Buildable = GetWorld()->SpawnActorDeferred<AFGBuildable>(
        BuildClass, Transform, nullptr, nullptr,
        ESpawnActorCollisionHandlingMethod::AlwaysSpawn);
    if (!Buildable)
    {
        OutError = TEXT("SpawnActorDeferred returned null");
        return nullptr;
    }

    Buildable->FinishSpawning(Transform);

    UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: spawned %s at %s"),
        *Buildable->GetName(), *Transform.GetLocation().ToString());
    return Buildable;
}

bool ASatAISubsystem::SetMachineRecipe(AFGBuildable* Buildable, const FString& RecipePath, FString& OutError)
{
    AFGBuildableManufacturer* Machine = Cast<AFGBuildableManufacturer>(Buildable);
    if (!Machine)
    {
        OutError = TEXT("buildable is not a manufacturer");
        return false;
    }

    TSubclassOf<UFGRecipe> Recipe = LoadClass<UFGRecipe>(nullptr, *RecipePath);
    if (!Recipe)
    {
        OutError = FString::Printf(TEXT("recipe not found: %s"), *RecipePath);
        return false;
    }

    Machine->SetRecipe(Recipe);
    return true;
}

void ASatAISubsystem::LogPorts(AFGBuildable* Buildable) const
{
    TArray<UFGFactoryConnectionComponent*> Ports;
    Buildable->GetComponents<UFGFactoryConnectionComponent>(Ports);

    const FTransform& BuildingTransform = Buildable->GetActorTransform();
    UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: %s has %d belt ports"),
        *Buildable->GetClass()->GetName(), Ports.Num());

    for (UFGFactoryConnectionComponent* Port : Ports)
    {
        const EFactoryConnectionDirection Dir = Port->GetDirection();
        const TCHAR* DirText =
            Dir == EFactoryConnectionDirection::FCD_INPUT ? TEXT("IN") :
            Dir == EFactoryConnectionDirection::FCD_OUTPUT ? TEXT("OUT") : TEXT("OTHER");

        FVector LocalPos = BuildingTransform.InverseTransformPosition(Port->GetComponentLocation()) / 100.f;
        FVector LocalFacing = BuildingTransform.InverseTransformVectorNoScale(Port->GetConnectorNormal());

        UE_LOG(LogTemp, Warning, TEXT("SatAIBridge:   %-20s %-5s pos_m=%s facing=%s"),
            *Port->GetName(), DirText, *LocalPos.ToString(), *LocalFacing.ToString());
    }
}

UFGFactoryConnectionComponent* ASatAISubsystem::FindFreePort(AFGBuildable* Buildable, bool bOutput) const
{
    const EFactoryConnectionDirection Wanted =
        bOutput ? EFactoryConnectionDirection::FCD_OUTPUT : EFactoryConnectionDirection::FCD_INPUT;

    TArray<UFGFactoryConnectionComponent*> Ports;
    Buildable->GetComponents<UFGFactoryConnectionComponent>(Ports);
    for (UFGFactoryConnectionComponent* Port : Ports)
    {
        if (Port->GetDirection() == Wanted && !Port->IsConnected())
        {
            return Port;
        }
    }
    return nullptr;
}

AFGBuildableConveyorBelt* ASatAISubsystem::ConnectWithBelt(AFGBuildable* From, AFGBuildable* To,
    const FString& BeltClassPath, FString& OutError)
{
    UFGFactoryConnectionComponent* OutPort = FindFreePort(From, true);
    UFGFactoryConnectionComponent* InPort = FindFreePort(To, false);
    if (!OutPort || !InPort)
    {
        OutError = TEXT("no free output on source or no free input on target");
        return nullptr;
    }

    TSubclassOf<AFGBuildable> BeltClass = LoadClass<AFGBuildableConveyorBelt>(nullptr, *BeltClassPath);
    if (!BeltClass)
    {
        OutError = FString::Printf(TEXT("belt class not found: %s"), *BeltClassPath);
        return nullptr;
    }

    AFGBuildableConveyorBelt* Belt = Cast<AFGBuildableConveyorBelt>(
        UFGTestBlueprintFunctionLibrary::SpawnSplineBuildable(BeltClass, OutPort, InPort));
    if (!Belt)
    {
        OutError = TEXT("SpawnSplineBuildable returned null");
        return nullptr;
    }

    UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: belt %s from %s.%s to %s.%s, length %.1f m"),
        *Belt->GetName(),
        *From->GetName(), *OutPort->GetName(),
        *To->GetName(), *InPort->GetName(),
        Belt->GetLength() / 100.f);
    return Belt;
}

FString ASatAISubsystem::HandlePing()
{
    PingCount++;
    return FString::Printf(TEXT("SatAIBridge: pong (v0.1, ping #%d)"), PingCount);
}