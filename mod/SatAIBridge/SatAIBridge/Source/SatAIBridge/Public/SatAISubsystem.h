#pragma once

#include "CoreMinimal.h"
#include "Subsystem/ModSubsystem.h"
#include "HttpRouteHandle.h"
#include "SatAISubsystem.generated.h"

class AFGBuildable;
class UFGFactoryConnectionComponent;
class AFGBuildableConveyorBelt;
class IHttpRouter;

UCLASS(Abstract, Blueprintable)
class SATAIBRIDGE_API ASatAISubsystem : public AModSubsystem
{
    GENERATED_BODY()

public:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    static ASatAISubsystem* Get(const UObject* WorldContext);
    AFGBuildable* SpawnBuildable(const FString& ClassPath, const FTransform& Transform, FString& OutError);
    bool SetMachineRecipe(AFGBuildable* Buildable, const FString& RecipePath, FString& OutError);
    void LogPorts(AFGBuildable* Buildable) const;
    UFGFactoryConnectionComponent* FindFreePort(AFGBuildable* Buildable, bool bOutput) const;
    AFGBuildableConveyorBelt* ConnectWithBelt(AFGBuildable* From, AFGBuildable* To,
        const FString& BeltClassPath, FString& OutError);

    FString HandlePing();

private:
    void StartHttpServer();
    void StopHttpServer();

    static constexpr uint32 HttpPort = 18642;
    TSharedPtr<IHttpRouter> HttpRouter;
    TArray<FHttpRouteHandle> HttpRoutes;

    int32 PingCount = 0;
    static TWeakObjectPtr<ASatAISubsystem> Instance;
};