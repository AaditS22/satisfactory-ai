#pragma once

#include "CoreMinimal.h"
#include "Subsystem/ModSubsystem.h"
#include "HttpRouteHandle.h"
#include "SatAISubsystem.generated.h"

class AFGBuildable;
class UFGFactoryConnectionComponent;
class AFGBuildableConveyorBelt;
class IHttpRouter;

struct FSatAISpawnResult
{
    FString Name;
    FTransform Transform;
    FBox LocalBounds = FBox(ForceInit);
    bool bLightweight = false;
};

struct FSatAITrackedPiece
{
    TWeakObjectPtr<AActor> Actor;
    UClass* LightweightClass = nullptr;
    int32 LightweightIndex = INDEX_NONE;
    FVector Location = FVector::ZeroVector;
};

UCLASS(Abstract, Blueprintable)
class SATAIBRIDGE_API ASatAISubsystem : public AModSubsystem
{
    GENERATED_BODY()

public:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    static ASatAISubsystem* Get(const UObject* WorldContext);
    AFGBuildable* SpawnBuildable(const FString& ClassPath, const FTransform& Transform, FString& OutError,
        const FString& BuiltWithRecipePath = FString());
    bool SetMachineRecipe(AFGBuildable* Buildable, const FString& RecipePath, FString& OutError);
    void LogPorts(AFGBuildable* Buildable) const;
    UFGFactoryConnectionComponent* FindFreePort(AFGBuildable* Buildable, bool bOutput) const;
    AFGBuildableConveyorBelt* ConnectWithBelt(AFGBuildable* From, AFGBuildable* To,
        const FString& BeltClassPath, FString& OutError);

    FString HandlePing();

    bool SpawnTracked(const FString& BuildId, const FString& ClassPath, const FString& BuiltWithRecipePath,
        const FTransform& Transform, FSatAISpawnResult& Out, FString& OutError);
    int32 ClearBuild(const FString& BuildId);

private:
    int32 FindLightweightIndex(UClass* Class, const FVector& LocationCm) const;

    void StartHttpServer();
    void StopHttpServer();

    static constexpr uint32 HttpPort = 18642;
    TSharedPtr<IHttpRouter> HttpRouter;
    TArray<FHttpRouteHandle> HttpRoutes;

    TMap<FString, TArray<FSatAITrackedPiece>> Builds;

    int32 PingCount = 0;
    static TWeakObjectPtr<ASatAISubsystem> Instance;
};